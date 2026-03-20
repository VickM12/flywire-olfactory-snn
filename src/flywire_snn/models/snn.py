from typing import Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from norse.torch import LIFCell, LIFParameters


class MaskedRecurrentLIFSNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        adjacency: sp.csr_matrix,
        steps: int = 20,
        alpha: float = 100.0,
    ) -> None:
        super().__init__()
        self.steps = steps
        self.input_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.recurrent = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        nn.init.kaiming_uniform_(self.recurrent, a=np.sqrt(5.0))

        adj_dense = adjacency.toarray().astype(np.float32)
        if adj_dense.shape[0] != hidden_dim:
            raise ValueError(
                f"Adjacency shape {adj_dense.shape} does not match hidden_dim={hidden_dim}."
            )
        mask = (adj_dense != 0).astype(np.float32)
        sign = np.sign(adj_dense).astype(np.float32)
        self.register_buffer("rec_mask", torch.from_numpy(mask))
        self.register_buffer("rec_sign", torch.from_numpy(sign))

        params = LIFParameters(method="super", alpha=torch.as_tensor(alpha))
        self.lif = LIFCell(p=params)
        self.readout = nn.Linear(hidden_dim, num_classes)

    def _rate_encode(self, x: torch.Tensor) -> torch.Tensor:
        # Map standardized receptor values to [0,1] firing probabilities.
        x_min = x.min(dim=1, keepdim=True).values
        x_max = x.max(dim=1, keepdim=True).values
        denom = (x_max - x_min).clamp_min(1e-6)
        rates = (x - x_min) / denom
        return rates.clamp(0.0, 1.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        device = x.device
        rates = self._rate_encode(x)
        state = None
        spikes_acc = torch.zeros(batch_size, self.recurrent.shape[0], device=device)
        spk_prev = torch.zeros(batch_size, self.recurrent.shape[0], device=device)

        signed_masked_rec = self.recurrent * self.rec_mask * self.rec_sign
        input_current = self.input_proj(rates)

        for _ in range(self.steps):
            poisson = torch.bernoulli(rates)
            current = input_current * poisson + torch.matmul(spk_prev, signed_masked_rec.T)
            spk, state = self.lif(current, state)
            spikes_acc = spikes_acc + spk
            spk_prev = spk

        spike_rate = spikes_acc / float(self.steps)
        logits = self.readout(spike_rate)
        spike_sparsity = (spikes_acc == 0).float().mean()
        return logits, spike_sparsity

