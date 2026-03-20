"""Topology-randomized SNN with identical dynamics to MaskedRecurrentLIFSNN."""

from __future__ import annotations

import scipy.sparse as sp

from flywire_snn.connectome.degree_shuffle import degree_preserving_shuffle
from flywire_snn.models.snn import MaskedRecurrentLIFSNN


class ShuffledSNN(MaskedRecurrentLIFSNN):
    """Same as connectome SNN but with degree-preserving shuffled recurrent mask."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        adjacency: sp.csr_matrix,
        shuffle_seed: int,
        steps: int = 20,
        alpha: float = 100.0,
    ) -> None:
        shuffled = degree_preserving_shuffle(adjacency, seed=shuffle_seed)
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            adjacency=shuffled,
            steps=steps,
            alpha=alpha,
        )
