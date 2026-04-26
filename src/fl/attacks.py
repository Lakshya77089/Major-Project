"""
Privacy attack evaluations for zkFedMoE.

Implements a black-box loss-threshold Membership Inference Attack (MIA)
that tells whether a given sample was part of the training set of a model.

This is the standard way to *measure* whether a privacy defense (DP-SGD)
actually works.  A well-defended model should give MIA AUC close to 0.5
(random guessing), while an undefended model often gives AUC >= 0.7.

Interpretation:
  * AUC = 0.5  --> attacker cannot distinguish train from non-train
                   (DP is working)
  * AUC = 1.0  --> attacker perfectly identifies training members
                   (privacy catastrophe)
"""

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class MIAResult:
    """Result of a membership inference attack evaluation."""
    auc: float
    attack_accuracy: float
    threshold: float
    train_loss_mean: float
    nonmember_loss_mean: float
    num_members: int
    num_nonmembers: int

    def summary(self) -> str:
        interp = (
            "DP is effective" if self.auc < 0.55
            else "Weak defense" if self.auc < 0.65
            else "Significant leakage"
        )
        return (
            f"MIA AUC={self.auc:.3f}, attack acc={self.attack_accuracy:.1%} "
            f"(thr={self.threshold:.4f}) -- {interp}"
        )


def _per_sample_losses(
    model: nn.Module,
    dataset: Dataset,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute per-sample cross-entropy loss. Returns a 1-D tensor."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    criterion = nn.CrossEntropyLoss(reduction="none")
    model.to(device).eval()
    losses = []
    with torch.no_grad():
        for ids, lbl in loader:
            ids = ids.to(device)
            lbl = lbl.to(device)
            logits = model(ids)
            loss = criterion(logits, lbl)
            losses.append(loss.cpu())
    return torch.cat(losses)


def _roc_auc(scores_pos: torch.Tensor, scores_neg: torch.Tensor) -> float:
    """
    Compute AUC of a binary classifier via the Mann-Whitney U statistic.

    scores_pos = scores assigned to positive (member) samples
    scores_neg = scores assigned to negative (non-member) samples

    A higher score should mean "more likely to be a member".
    """
    p = scores_pos.numpy()
    n = scores_neg.numpy()
    # AUC = P(score_pos > score_neg).  Compute by rank sum.
    import numpy as np
    all_scores = np.concatenate([p, n])
    labels = np.concatenate([np.ones_like(p), np.zeros_like(n)])
    order = np.argsort(all_scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(all_scores) + 1)
    # Average ranks for ties
    from collections import defaultdict
    tie_groups = defaultdict(list)
    for i, s in enumerate(all_scores):
        tie_groups[s].append(i)
    for idxs in tie_groups.values():
        if len(idxs) > 1:
            avg = ranks[idxs].mean()
            for i in idxs:
                ranks[i] = avg
    rank_sum_pos = ranks[labels == 1].sum()
    n_pos = len(p)
    n_neg = len(n)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def membership_inference_attack(
    model: nn.Module,
    members: Dataset,
    nonmembers: Dataset,
    device: torch.device = torch.device("cpu"),
    batch_size: int = 64,
    max_samples: int = 500,
) -> MIAResult:
    """
    Run a loss-threshold Membership Inference Attack.

    The attack: compute per-sample loss for both a set the model was trained
    on (members) and a disjoint set (non-members).  Members tend to have
    LOWER loss because the model memorised them.  We convert to a score
    where higher = more likely a member: score = -loss.  Then measure AUC.

    Args:
        model: the trained global model (FL output)
        members: subset of clients' training data
        nonmembers: held-out samples the model has never seen
        device: torch device
        batch_size: loader batch size
        max_samples: cap each side to this many samples for speed

    Returns:
        MIAResult with AUC, best threshold accuracy, and diagnostics.
    """
    # Subsample to keep the attack fast
    from torch.utils.data import Subset
    m_idx = list(range(min(len(members), max_samples)))
    n_idx = list(range(min(len(nonmembers), max_samples)))
    members_sub = Subset(members, m_idx)
    nonmembers_sub = Subset(nonmembers, n_idx)

    loss_members = _per_sample_losses(model, members_sub, batch_size, device)
    loss_nonmembers = _per_sample_losses(model, nonmembers_sub, batch_size, device)

    # Score = -loss  (higher score = more likely member)
    auc = _roc_auc(-loss_members, -loss_nonmembers)

    # Find best threshold on member loss (predict "member" if loss <= threshold)
    all_losses = torch.cat([loss_members, loss_nonmembers])
    labels = torch.cat([
        torch.ones_like(loss_members),
        torch.zeros_like(loss_nonmembers),
    ])
    sorted_losses, sort_idx = torch.sort(all_losses)
    sorted_labels = labels[sort_idx]
    # Sweep threshold at each unique loss value; count accuracy
    best_acc = 0.5
    best_thr = float(sorted_losses[0].item())
    total = len(sorted_losses)
    # Cumulative counts help us evaluate thresholds efficiently
    n_mem_leq = 0
    n_non_leq = 0
    for i in range(total):
        if sorted_labels[i].item() == 1.0:
            n_mem_leq += 1
        else:
            n_non_leq += 1
        # If we predict "member" when loss <= sorted_losses[i]:
        correct = n_mem_leq + (len(loss_nonmembers) - n_non_leq)
        acc = correct / total
        if acc > best_acc:
            best_acc = acc
            best_thr = float(sorted_losses[i].item())

    return MIAResult(
        auc=auc,
        attack_accuracy=best_acc,
        threshold=best_thr,
        train_loss_mean=float(loss_members.mean().item()),
        nonmember_loss_mean=float(loss_nonmembers.mean().item()),
        num_members=len(loss_members),
        num_nonmembers=len(loss_nonmembers),
    )
