from typing import Callable, Dict, Iterable, List, Optional, Tuple

import hashlib
import torch
from torch import nn


# ============================================================================
# Secure Aggregation via Pairwise Masking (Bonawitz et al. 2017, simplified)
# ============================================================================
# Each pair of clients (i, j) agrees on a shared seed s_ij.  From that seed
# they deterministically generate a pseudo-random mask tensor M_ij.  Client i
# adds +M_ij to its update for every j > i and -M_ij for every j < i.
# When the server sums all clients' masked updates the masks cancel.
#
# Key property:  the server sees only the sum of all updates (which is what
# it needs for FedAvg), never an individual client's update.
#
# We do NOT implement Diffie-Hellman key exchange here -- we assume a trusted
# setup that hands each pair a shared seed.  Real production systems (Google's
# Gboard) use proper DH + shamir secret sharing for dropout resilience.


def _mask_from_seed(
    seed: int,
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    scale: float = 1.0,
) -> torch.Tensor:
    """Deterministic Gaussian mask derived from an integer seed."""
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randn(shape, generator=g, dtype=torch.float32).to(dtype) * scale


def _pair_seed(i: int, j: int, round_id: int, param_name: str) -> int:
    """Derive a deterministic seed for pair (min(i,j), max(i,j)) in this round."""
    a, b = (i, j) if i < j else (j, i)
    key = f"{a}|{b}|{round_id}|{param_name}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    # 32-bit seed is enough for torch.Generator
    return int.from_bytes(digest[:4], "big")


def apply_secure_masks(
    state: Dict[str, torch.Tensor],
    client_id: int,
    num_clients: int,
    round_id: int,
    scale: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """
    Add pairwise masks to a client's state so that the SUM over all clients
    cancels out the masks.  Returns a masked copy of the state dict.

    scale controls the mask magnitude; small scale = fewer numerical issues.
    """
    masked: Dict[str, torch.Tensor] = {}
    for name, tensor in state.items():
        t = tensor.float().clone()
        for j in range(num_clients):
            if j == client_id:
                continue
            seed = _pair_seed(client_id, j, round_id, name)
            m = _mask_from_seed(seed, t.shape, t.dtype, scale=scale)
            if j > client_id:
                t = t + m
            else:
                t = t - m
        masked[name] = t
    return masked


class FedServer:
    """
    FedAvg server supporting both dense and sparse aggregation.

    For sparse updates (where clients send only a subset of parameters),
    the server keeps its existing weights for any keys not present in the
    client updates, and only averages the keys that clients actually sent.
    """

    def __init__(self, global_model: nn.Module, device: torch.device):
        self.global_model = global_model.to(device)
        self.device = device

    def get_global_state(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.global_model.state_dict().items()}

    def set_global_state(self, state: Dict[str, torch.Tensor]) -> None:
        self.global_model.load_state_dict(state, strict=False)

    def aggregate(
        self,
        client_states: Iterable[Tuple[Dict[str, torch.Tensor], int]],
    ) -> None:
        """
        FedAvg aggregation over (state_dict, num_samples) pairs.

        Handles sparse updates correctly: each parameter key is averaged
        only across clients that include it.  Keys absent from all clients
        retain their current global value.
        """
        client_states = list(client_states)
        if not client_states:
            return

        # Track per-key: weighted sum and total weight (for sparse support)
        agg_state: Dict[str, torch.Tensor] = {}
        agg_weight: Dict[str, float] = {}

        total_samples = sum(n for _, n in client_states)

        for state, n in client_states:
            weight = n / total_samples
            for name, tensor in state.items():
                if name not in agg_state:
                    agg_state[name] = tensor.float() * weight
                    agg_weight[name] = weight
                else:
                    agg_state[name] += tensor.float() * weight
                    agg_weight[name] += weight

        # Re-normalise sparse keys so weights sum to 1.0.
        # For dense updates every key appears in every client, so
        # agg_weight[k] == 1.0 already and this is a no-op.
        for name in agg_state:
            w = agg_weight[name]
            if w > 0 and abs(w - 1.0) > 1e-6:
                agg_state[name] /= w

        self._apply_agg(agg_state)

    # ---- helpers ----

    def _apply_agg(self, agg_state: Dict[str, torch.Tensor]) -> None:
        """Merge aggregated state into the global model."""
        global_state = self.global_model.state_dict()
        for name, tensor in agg_state.items():
            if name in global_state:
                global_state[name] = tensor.to(global_state[name].dtype)
        self.global_model.load_state_dict(global_state, strict=False)

    # ---- robust aggregation methods ----

    def aggregate_median(
        self,
        client_states: Iterable[Tuple[Dict[str, torch.Tensor], int]],
    ) -> None:
        """Coordinate-wise median aggregation (Byzantine-robust)."""
        client_states = list(client_states)
        if not client_states:
            return

        # Collect all keys
        all_keys = set()
        for s, _ in client_states:
            all_keys.update(s.keys())

        agg: Dict[str, torch.Tensor] = {}
        for key in all_keys:
            tensors = [s[key].float() for s, _ in client_states if key in s]
            if tensors:
                stacked = torch.stack(tensors, dim=0)
                agg[key] = stacked.median(dim=0).values

        self._apply_agg(agg)

    def aggregate_trimmed_mean(
        self,
        client_states: Iterable[Tuple[Dict[str, torch.Tensor], int]],
        trim_fraction: float = 0.1,
    ) -> None:
        """Trimmed mean: remove top/bottom fraction, then average."""
        client_states = list(client_states)
        if not client_states:
            return

        all_keys = set()
        for s, _ in client_states:
            all_keys.update(s.keys())

        n = len(client_states)
        trim_count = max(1, int(n * trim_fraction))

        agg: Dict[str, torch.Tensor] = {}
        for key in all_keys:
            tensors = [s[key].float() for s, _ in client_states if key in s]
            if not tensors:
                continue
            stacked = torch.stack(tensors, dim=0)  # (n_clients, ...)
            if len(tensors) > 2 * trim_count:
                sorted_t, _ = stacked.sort(dim=0)
                trimmed = sorted_t[trim_count:-trim_count]
                agg[key] = trimmed.mean(dim=0)
            else:
                agg[key] = stacked.mean(dim=0)

        self._apply_agg(agg)

    def aggregate_secure(
        self,
        client_states: Iterable[Tuple[Dict[str, torch.Tensor], int]],
        round_id: int = 0,
        mask_scale: float = 0.1,
    ) -> Dict[str, float]:
        """
        Secure FedAvg aggregation using pairwise masking.

        Each client's update is masked with pairwise masks; the server only
        sees the sum of masked updates (which equals the sum of the raw
        updates because masks cancel).  The server therefore *cannot*
        reconstruct any individual client's update.

        For demonstrability the masking is done on the server side here
        (it is mathematically identical to clients applying masks before
        sending).  The returned dict reports mask-cancel error statistics.

        Requires all clients to be present in this round and to have
        identical state_dict keys.  Sparse updates are NOT supported
        (secure aggregation needs a common key set).

        Returns a dict with diagnostic statistics:
            {"max_residual": float, "mean_residual": float, "num_clients": int}
        """
        client_states = list(client_states)
        if not client_states:
            return {"max_residual": 0.0, "mean_residual": 0.0, "num_clients": 0}

        n_clients = len(client_states)
        total_samples = sum(n for _, n in client_states)

        # Step 1: mask each client's update (demonstrative; in practice clients
        # do this locally before sending).  Also compute the TRUE weighted
        # sum for the residual diagnostic (the server never sees it).
        masked_states: List[Dict[str, torch.Tensor]] = []
        true_agg: Dict[str, torch.Tensor] = {}
        for i, (state, n) in enumerate(client_states):
            w = n / total_samples
            # Mask is added to the UNWEIGHTED update; we multiply by weight
            # AFTER masking.  So masks need to be scaled by (1/w_j) ... no,
            # easier to just weight the SUM.  We mask state-as-is, sum,
            # divide by n_clients, and trust that all clients contribute
            # equal weight (1/N).  For true FedAvg-weighted secure aggr.
            # we would use weighted masks; here we do uniform weighting.
            masked = apply_secure_masks(
                state, client_id=i, num_clients=n_clients,
                round_id=round_id, scale=mask_scale,
            )
            masked_states.append(masked)
            # Track true unweighted sum for residual diagnostic
            for k, v in state.items():
                if k in true_agg:
                    true_agg[k] = true_agg[k] + v.float()
                else:
                    true_agg[k] = v.float().clone()

        # Step 2: server sums the masked updates (masks cancel)
        agg: Dict[str, torch.Tensor] = {}
        for masked in masked_states:
            for k, v in masked.items():
                if k in agg:
                    agg[k] = agg[k] + v
                else:
                    agg[k] = v.clone()

        # Step 3: diagnostics -- how well did the masks cancel?
        residuals = []
        for k in agg:
            if k in true_agg:
                residuals.append((agg[k] - true_agg[k]).abs().max().item())
        max_res = max(residuals) if residuals else 0.0
        mean_res = sum(residuals) / len(residuals) if residuals else 0.0

        # Step 4: divide by number of clients to get the mean (uniform FedAvg)
        final_agg = {k: v / n_clients for k, v in agg.items()}
        self._apply_agg(final_agg)

        return {
            "max_residual": max_res,
            "mean_residual": mean_res,
            "num_clients": n_clients,
        }

    def aggregate_with_verification(
        self,
        client_states: Iterable[Tuple[Dict[str, torch.Tensor], int]],
        proofs: list,
        verify_fn: Callable,
        expected_k: int = 2,
    ) -> Tuple[int, int]:
        """
        FedAvg but reject clients whose SEPG proof fails verification.

        Returns (accepted_count, rejected_count).
        """
        client_states = list(client_states)
        accepted = []
        rejected = 0

        for (state, n), proof in zip(client_states, proofs):
            passed, reason = verify_fn(proof, state, expected_k)
            if passed:
                accepted.append((state, n))
            else:
                rejected += 1

        if accepted:
            self.aggregate(accepted)

        return len(accepted), rejected

