from typing import Dict, Iterable, List, Tuple

import torch
from torch import nn


class FedServer:
    """
    Minimal FedAvg server for Phase 1.
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
        """
        client_states = list(client_states)
        if not client_states:
            return

        total_samples = sum(n for _, n in client_states)
        # Initialize with zeros
        agg_state: Dict[str, torch.Tensor] = {}

        for state, n in client_states:
            weight = n / total_samples
            for name, tensor in state.items():
                if name not in agg_state:
                    agg_state[name] = tensor.float() * weight
                else:
                    agg_state[name] += tensor.float() * weight

        # Update only keys present in agg_state
        global_state = self.global_model.state_dict()
        for name, tensor in agg_state.items():
            if name in global_state:
                global_state[name] = tensor.to(global_state[name].dtype)
        self.global_model.load_state_dict(global_state, strict=False)

