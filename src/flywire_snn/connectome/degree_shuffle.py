"""Directed degree-preserving edge shuffles (does not modify FlyWire loading)."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def degree_preserving_shuffle(
    adjacency: sp.spmatrix,
    seed: int,
    n_swaps: int | None = None,
) -> sp.csr_matrix:
    """
    Randomize directed edges while preserving each node's in-degree and out-degree.
    Uses repeated valid 2-edge swaps on an edge list.
    """
    adj = adjacency.tocsr()
    n = int(adj.shape[0])
    coo = adj.tocoo()
    rows = coo.row.tolist()
    cols = coo.col.tolist()
    data = coo.data.astype(np.float32).tolist()
    m = len(rows)
    if m == 0:
        return adj

    edge_set = set(zip(rows, cols))
    if n_swaps is None:
        n_swaps = max(5000, 50 * m)

    rng = np.random.default_rng(seed)

    for _ in range(n_swaps):
        ei = int(rng.integers(0, m))
        ej = int(rng.integers(0, m))
        if ei == ej:
            continue
        a, b = rows[ei], cols[ei]
        c, d = rows[ej], cols[ej]
        if len({a, b, c, d}) < 4:
            continue
        if (a, d) in edge_set or (c, b) in edge_set:
            continue
        wi, wj = data[ei], data[ej]
        edge_set.remove((a, b))
        edge_set.remove((c, d))
        edge_set.add((a, d))
        edge_set.add((c, b))
        rows[ei], cols[ei] = a, d
        rows[ej], cols[ej] = c, b
        data[ei], data[ej] = wi, wj

    out = sp.coo_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32).tocsr()
    return out
