from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def _is_expert_param(name: str) -> Tuple[bool, int]:
    """
    Heuristic: parameters under moe.experts.<idx>.* are expert parameters.
    Returns (is_expert, expert_index or -1).
    """
    parts = name.split(".")
    for i in range(len(parts) - 2):
        if parts[i] == "moe" and parts[i + 1] == "experts":
            try:
                return True, int(parts[i + 2])
            except ValueError:
                return True, -1
    return False, -1


def local_train(
    model: nn.Module,
    dataset: Dataset,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    top_k_sparse: int = 2,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], int, int, int, List[int], torch.Tensor]:
    """
    Local training loop for one client.

    All parameters (including embedding) are trained.  The dense update
    contains the full model state.  The sparse update excludes expert
    parameters that were *not* among the Top-K most-routed experts,
    demonstrating the MoE communication advantage.

    Returns:
        full_state    -- complete model state (dense update)
        sparse_state  -- model state excluding non-Top-K expert params
        n_samples     -- number of training samples processed
        dense_bytes   -- bytes for the dense update
        sparse_bytes  -- bytes for the sparse update
        top_k_indices -- which expert indices were selected as Top-K
        expert_usage  -- (num_experts,) tensor of accumulated routing probs
    """
    model = model.to(device)
    model.train()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Track expert usage via router outputs
    expert_usage = None
    n_samples = 0

    for _ in range(epochs):
        for batch in loader:
            input_ids, labels = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            n_samples += labels.size(0)

            optimizer.zero_grad()

            # Forward through submodules to capture router probs
            x = model.embedding(input_ids)
            x = x.mean(dim=1)
            x, router_probs = model.moe(x)
            logits = model.classifier(x)

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            # Accumulate expert usage (sum of routing probabilities)
            with torch.no_grad():
                batch_usage = router_probs.sum(dim=0)  # (num_experts,)
                if expert_usage is None:
                    expert_usage = batch_usage
                else:
                    expert_usage += batch_usage

    # Select Top-K most-used experts based on actual router statistics
    if expert_usage is not None:
        top_k_indices = expert_usage.topk(top_k_sparse).indices.cpu().tolist()
    else:
        top_k_indices = list(range(top_k_sparse))
    top_k_set = set(top_k_indices)

    # Build full (dense) and sparse states
    full_state: Dict[str, torch.Tensor] = {}
    sparse_state: Dict[str, torch.Tensor] = {}
    dense_bytes = 0
    sparse_bytes = 0

    for name, tensor in model.state_dict().items():
        t = tensor.detach().cpu().clone()
        num_bytes = t.numel() * t.element_size()

        full_state[name] = t
        dense_bytes += num_bytes

        is_expert, idx = _is_expert_param(name)
        # Include in sparse: everything except non-selected experts
        if not is_expert or idx in top_k_set:
            sparse_state[name] = t
            sparse_bytes += num_bytes

    # Normalise expert usage; fallback to zeros if no data was processed
    if expert_usage is None:
        expert_usage = torch.zeros(top_k_sparse)
    expert_usage = expert_usage.detach().cpu()

    return full_state, sparse_state, n_samples, dense_bytes, sparse_bytes, top_k_indices, expert_usage

