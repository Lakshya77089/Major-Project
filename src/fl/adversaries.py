"""
Adversary client simulations for zkFedMoE.

Three attack types:
  - Poisoning: flip labels during local training
  - Free-rider: return stale/random updates without real training
  - Sybil: one attacker poses as multiple clients
"""

from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def poisoning_train(
    model: nn.Module,
    dataset: Dataset,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    num_classes: int = 4,
    flip_fraction: float = 1.0,
) -> Tuple[Dict[str, torch.Tensor], int]:
    """
    Poisoning attack: randomly flip a fraction of labels during training.

    Returns (state_dict, n_samples) — same interface as honest training.
    """
    model = model.to(device)
    model.train()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n_samples = 0
    for _ in range(epochs):
        for input_ids, labels in loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            n_samples += labels.size(0)

            # Flip labels: assign random classes
            mask = torch.rand(labels.size(0), device=device) < flip_fraction
            if mask.any():
                labels[mask] = torch.randint(0, num_classes, (mask.sum().item(),), device=device)

            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

    state = {n: p.detach().cpu().clone() for n, p in model.state_dict().items()}
    return state, n_samples


def freerider_train(
    global_state: Dict[str, torch.Tensor],
    n_samples: int = 100,
    noise_scale: float = 1e-4,
) -> Tuple[Dict[str, torch.Tensor], int]:
    """
    Free-rider attack: return the global state with tiny random noise,
    pretending to have trained without actually doing real work.

    Returns (state_dict, fake_n_samples).
    """
    state = {}
    for k, v in global_state.items():
        state[k] = v.clone() + torch.randn_like(v.float()) * noise_scale
    return state, n_samples


def sybil_clones(
    base_state: Dict[str, torch.Tensor],
    base_n_samples: int,
    num_clones: int = 3,
) -> List[Tuple[Dict[str, torch.Tensor], int]]:
    """
    Sybil attack: duplicate a single client's update multiple times
    to amplify its influence during aggregation.

    Returns a list of (state_dict, n_samples) tuples, each a clone.
    """
    clones = []
    for _ in range(num_clones):
        cloned = {k: v.clone() for k, v in base_state.items()}
        clones.append((cloned, base_n_samples))
    return clones
