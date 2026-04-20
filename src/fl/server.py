from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn


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

