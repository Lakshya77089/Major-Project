from typing import Dict, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def local_train(
    model: nn.Module,
    dataset: Dataset,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], int, int, int]:
    """
    Simple local training loop for one client.

    Returns:
        state_dict (only trainable params) and number of samples used.
    """
    model = model.to(device)
    model.train()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    n_samples = 0
    for _ in range(epochs):
        for batch in loader:
            input_ids, labels = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            n_samples += labels.size(0)

            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

    # For Phase 1/2, return the full model state plus a sparse (Top-K experts) view
    # and communication statistics.
    full_state = {name: p.detach().cpu().clone() for name, p in model.state_dict().items()}

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

    dense_bytes = 0
    sparse_bytes = 0
    top_k_sparse = 2  # fixed K for Phase 2 comm accounting / sparse view

    sparse_state: Dict[str, torch.Tensor] = {}

    for name, tensor in full_state.items():
        num_bytes = tensor.numel() * tensor.element_size()
        dense_bytes += num_bytes

        is_expert, idx = _is_expert_param(name)
        if not is_expert or (0 <= idx < top_k_sparse):
            sparse_bytes += num_bytes
            sparse_state[name] = tensor

    return full_state, sparse_state, n_samples, dense_bytes, sparse_bytes

