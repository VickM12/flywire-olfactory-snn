"""Fully connected ReLU MLP; can match parameter count of connectome SNN."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


def count_dense_mlp_params(input_dim: int, hidden_dim: int, num_classes: int) -> int:
    return (
        input_dim * hidden_dim
        + hidden_dim
        + hidden_dim * hidden_dim
        + hidden_dim
        + hidden_dim * num_classes
        + num_classes
    )


def hidden_dim_for_param_target(
    input_dim: int,
    num_classes: int,
    target_params: int,
    max_hidden: int = 8192,
) -> int:
    """Pick hidden width so total MLP params are closest to target_params."""

    def nparams(h: int) -> int:
        return count_dense_mlp_params(input_dim, h, num_classes)

    lo, hi = 1, max_hidden
    best_h, best_diff = 1, abs(nparams(1) - target_params)
    while lo <= hi:
        mid = (lo + hi) // 2
        if nparams(mid) < target_params:
            lo = mid + 1
        else:
            hi = mid - 1
    for h in range(max(1, lo - 2), min(max_hidden, lo + 2) + 1):
        d = abs(nparams(h) - target_params)
        if d < best_diff:
            best_diff = d
            best_h = h
    return best_h


class DenseMLP(nn.Module):
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
        return logits, torch.tensor(0.0, device=x.device)

    @classmethod
    def matched_to_connectome_snn(
        cls,
        input_dim: int,
        num_classes: int,
        connectome_snn_param_count: int,
    ) -> "DenseMLP":
        h = hidden_dim_for_param_target(input_dim, num_classes, connectome_snn_param_count)
        return cls(input_dim, h, num_classes)
