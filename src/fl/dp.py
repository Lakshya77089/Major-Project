"""
Differential Privacy utilities for zkFedMoE.

Provides gradient clipping, Gaussian noise injection, and a simple
privacy accountant for tracking cumulative (epsilon, delta) budget.
"""

import math
from typing import Dict, Tuple

import torch


def clip_update(state: Dict[str, torch.Tensor], max_norm: float) -> Dict[str, torch.Tensor]:
    """Clip a model update (state dict) so its overall L2 norm <= max_norm."""
    # Flatten all tensors to compute global norm
    all_params = [t.float().flatten() for t in state.values()]
    flat = torch.cat(all_params)
    total_norm = flat.norm(2).item()

    clip_factor = min(1.0, max_norm / (total_norm + 1e-8))
    if clip_factor < 1.0:
        return {k: v * clip_factor for k, v in state.items()}
    return state


def add_noise(state: Dict[str, torch.Tensor], noise_scale: float) -> Dict[str, torch.Tensor]:
    """Add Gaussian noise N(0, noise_scale^2) to every tensor in a state dict."""
    noisy = {}
    for k, v in state.items():
        noise = torch.randn_like(v.float()) * noise_scale
        noisy[k] = v.float() + noise
    return noisy


def apply_dp(
    state: Dict[str, torch.Tensor],
    clip_norm: float,
    noise_multiplier: float,
) -> Dict[str, torch.Tensor]:
    """Clip update to norm C, then add Gaussian noise with scale sigma = noise_multiplier * C."""
    clipped = clip_update(state, clip_norm)
    sigma = noise_multiplier * clip_norm
    return add_noise(clipped, sigma)


class PrivacyAccountant:
    """
    Simple privacy accountant using the Gaussian mechanism formula.

    For each round of DP-SGD with noise_multiplier sigma and sampling
    probability q = batch_size / dataset_size, the per-step epsilon at
    a given delta is approximated by:

        epsilon_step = q * sqrt(2 * ln(1.25 / delta)) / sigma

    This is the basic composition bound.  For tighter bounds, use
    Renyi DP (not implemented here for simplicity).
    """

    def __init__(self, target_delta: float = 1e-5):
        self.target_delta = target_delta
        self.steps: int = 0
        self._per_step_eps: float = 0.0

    def accumulate(
        self,
        noise_multiplier: float,
        sample_rate: float,
        num_steps: int = 1,
    ) -> None:
        """Record num_steps of DP-SGD with given parameters."""
        if noise_multiplier <= 0:
            return
        # Basic Gaussian mechanism bound per step
        eps_step = sample_rate * math.sqrt(2 * math.log(1.25 / self.target_delta)) / noise_multiplier
        self._per_step_eps = eps_step
        self.steps += num_steps

    def get_privacy_spent(self) -> Tuple[float, float]:
        """Return (epsilon, delta) under basic composition."""
        if self.steps == 0:
            return (0.0, 0.0)
        # Basic composition: epsilon grows with sqrt(steps)
        total_eps = self._per_step_eps * math.sqrt(self.steps)
        return (total_eps, self.target_delta)
