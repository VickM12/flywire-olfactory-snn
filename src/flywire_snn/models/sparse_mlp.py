"""MLP with fixed random sparse masks at a target density (no spiking)."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        sparsity_ratio: float,
        seed: int,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, num_classes)

        rng = np.random.default_rng(seed)
        d1 = float(np.clip(sparsity_ratio, 0.0, 1.0))
        m1 = (rng.random((hidden_dim, input_dim)) < d1).astype(np.float32)
        m2 = (rng.random((hidden_dim, hidden_dim)) < d1).astype(np.float32)

        self.register_buffer("mask1", torch.from_numpy(m1))
        self.register_buffer("mask2", torch.from_numpy(m2))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        w1 = self.lin1.weight * self.mask1
        w2 = self.lin2.weight * self.mask2
        h = F.linear(x, w1, self.lin1.bias)
        h = F.relu(h)
        h = F.linear(h, w2, self.lin2.bias)
        h = F.relu(h)
        logits = self.lin3(h)
        return logits, torch.tensor(0.0, device=x.device)


def recurrent_sparsity_ratio(adjacency) -> float:
    """Fraction of possible directed recurrent weights that are nonzero."""
    adj = adjacency.tocsr()
    h = adj.shape[0]
    if h == 0:
        return 0.0
    return float(adj.nnz) / float(h * h)
