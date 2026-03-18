import math
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    """
    Minimal LoRA wrapper around a Linear layer.

    Only the low-rank adapters are trained; the base weight is frozen by the caller.
    """

    def __init__(self, in_features: int, out_features: int, r: int = 8, alpha: float = 1.0):
        super().__init__()
        self.base = nn.Linear(in_features, out_features)
        self.r = r
        self.scaling = alpha / r if r > 0 else 0.0

        if r > 0:
            self.A = nn.Linear(in_features, r, bias=False)
            self.B = nn.Linear(r, out_features, bias=False)
            # Initialize LoRA weights small
            nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.B.weight)
        else:
            self.register_parameter("A", None)
            self.register_parameter("B", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        if self.r > 0:
            lora_out = self.B(self.A(x)) * self.scaling
            return base_out + lora_out
        return base_out


class MoEExpert(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MoELayer(nn.Module):
    """
    Simple MoE layer with softmax router and Top-K gating.
    """

    def __init__(self, dim: int, num_experts: int = 4, hidden_dim: int = 128, k: int = 2):
        super().__init__()
        assert 1 <= k <= num_experts
        self.dim = dim
        self.num_experts = num_experts
        self.k = k

        self.experts = nn.ModuleList([MoEExpert(dim, hidden_dim) for _ in range(num_experts)])
        self.router = nn.Linear(dim, num_experts)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, dim)

        Returns:
            y: (batch, dim)
            router_probs: (batch, num_experts) softmax probabilities
        """
        logits = self.router(x)  # (B, E)
        probs = F.softmax(logits, dim=-1)

        # Top-K per example
        topk_vals, topk_idx = torch.topk(probs, self.k, dim=-1)  # (B, K)

        expert_outputs = []
        for e_idx, expert in enumerate(self.experts):
            mask = (topk_idx == e_idx)  # (B, K)
            if not mask.any():
                expert_outputs.append(torch.zeros_like(x))
                continue
            # Any position selected for this expert?
            selected = mask.any(dim=-1)  # (B,)
            if not selected.any():
                expert_outputs.append(torch.zeros_like(x))
                continue
            x_e = x[selected]
            out_e = expert(x_e)
            # Sum of routing weights for this expert per example
            weights = topk_vals[selected][mask[selected]].view(-1, 1)
            out_weighted = torch.zeros_like(x)
            out_weighted[selected] = out_e * weights
            expert_outputs.append(out_weighted)

        y = torch.stack(expert_outputs, dim=0).sum(dim=0)
        return y, probs


class MoETextClassifier(nn.Module):
    """
    Small text classifier:
        token embedding -> mean pooling -> MoE layer -> LoRA classifier head

    This keeps the model tiny for Phase 1 experiments.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_classes: int,
        num_experts: int = 4,
        expert_hidden_dim: int = 128,
        k: int = 2,
        lora_r: int = 8,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.moe = MoELayer(embed_dim, num_experts=num_experts, hidden_dim=expert_hidden_dim, k=k)
        self.classifier = LoRALinear(embed_dim, num_classes, r=lora_r, alpha=1.0)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len) token indices
        """
        x = self.embedding(input_ids)  # (B, L, D)
        x = x.mean(dim=1)  # simple mean pooling
        x, _ = self.moe(x)
        logits = self.classifier(x)
        return logits

