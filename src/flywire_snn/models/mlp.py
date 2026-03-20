from typing import Tuple

import torch
import torch.nn as nn


class BaselineMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.net(x)
        # MLP has no spikes; return zero as placeholder.
        spike_sparsity = torch.tensor(0.0, device=x.device)
        return logits, spike_sparsity

