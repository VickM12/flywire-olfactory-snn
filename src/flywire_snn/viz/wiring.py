"""Interactive wiring-style graph: FlyWire L2 centroids (nm) when cached, else spring layout."""

from __future__ import annotations

from collections import deque
from typing import List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import plotly.graph_objects as go
import scipy.sparse as sp


def _expansion_nodes(A: sp.csr_matrix, center: int, max_nodes: int) -> List[int]:
    """Breadth-first expansion using both outgoing and incoming edges (weak neighborhood)."""
    n = A.shape[0]
    if n == 0 or max_nodes <= 0:
        return []
    center = int(np.clip(center, 0, n - 1))
    At = A.transpose().tocsr()
    order: List[int] = []
    seen: set[int] = set()
    q: deque[int] = deque([center])
    seen.add(center)
    while q and len(order) < max_nodes:
        u = q.popleft()
        order.append(u)
        nbrs: List[int] = []
        lo, hi = A.indptr[u], A.indptr[u + 1]
        nbrs.extend(A.indices[lo:hi].tolist())
        lo2, hi2 = At.indptr[u], At.indptr[u + 1]
        nbrs.extend(At.indices[lo2:hi2].tolist())
        for v in nbrs:
            if v not in seen:
                seen.add(v)
                q.append(v)
    return order


def _induced_edges(
    A: sp.csr_matrix,
    global_indices: Sequence[int],
    max_edges: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """Edges inside the node set; columns are local 0..k-1 indices. Subsample if too many."""
    idx = list(global_indices)
    sub = A[np.ix_(idx, idx)].tocsr()
    coo = sub.tocoo()
    rows = coo.row.astype(np.int32)
    cols = coo.col.astype(np.int32)
    data = coo.data.astype(np.float32)
    m = rows.size
    if m > max_edges:
        pick = rng.choice(m, size=max_edges, replace=False)
        rows, cols, data = rows[pick], cols[pick], data[pick]
    return rows, cols, data, idx


def make_wiring_figure(
    A: sp.csr_matrix,
    center_idx: int,
    max_nodes: int = 48,
    max_edges: int = 500,
    dim: int = 3,
    layout_seed: int = 0,
    rng: Optional[np.random.Generator] = None,
    positions_nm: Optional[np.ndarray] = None,
    prefer_brain_layout: bool = True,
) -> Optional[go.Figure]:
    """
    Wiring diagram: **FlyWire L2 bounding-box centers** (nm) when `positions_nm` is aligned with `A`
    and `prefer_brain_layout`, otherwise **spring** layout in 2D/3D.
    """
    if rng is None:
        rng = np.random.default_rng(layout_seed + 17)
    dim = 2 if dim == 2 else 3
    nodes_global = _expansion_nodes(A, center_idx, max_nodes)
    if not nodes_global:
        return None

    loc_r, loc_c, weights, idx_order = _induced_edges(A, nodes_global, max_edges, rng)
    if loc_r.size == 0:
        return None

    k = len(idx_order)
    G = nx.DiGraph()
    G.add_nodes_from(range(k))
    for r, c, w in zip(loc_r, loc_c, weights):
        G.add_edge(int(r), int(c), weight=float(abs(w)), sign=float(np.sign(w)))

    pos: dict[int, np.ndarray]
    layout_mode = "spring"
    if (
        prefer_brain_layout
        and positions_nm is not None
        and positions_nm.shape[0] == A.shape[0]
        and positions_nm.shape[1] >= 3
    ):
        gix = np.asarray(idx_order, dtype=int)
        pts = np.asarray(positions_nm[gix, :3], dtype=np.float64)
        finite = np.isfinite(pts).all(axis=1)
        # If some neurons are missing positions, still use the brain layout by
        # placing them at the centroid + tiny jitter. This keeps the plot in
        # brain-aligned coordinates while avoiding a full fallback to spring
        # layout.
        if finite.any():
            centroid = np.nanmean(pts[finite], axis=0)
            missing = ~finite
            if missing.any():
                jitter = (np.random.default_rng(layout_seed + 991).normal(0.0, 1.0, size=(missing.sum(), 3))) * 5_000.0
                pts[missing] = centroid + jitter
            layout_mode = "brain"
            pos = {}
            if dim == 3:
                for i in range(k):
                    pos[i] = np.array([pts[i, 0], pts[i, 1], pts[i, 2]], dtype=float)
            else:
                for i in range(k):
                    pos[i] = np.array([pts[i, 0], pts[i, 1]], dtype=float)

    if layout_mode == "spring":
        UG = nx.Graph()
        UG.add_nodes_from(G.nodes())
        for u, v in G.edges():
            UG.add_edge(u, v)
        pos = nx.spring_layout(
            UG,
            dim=dim,
            seed=int(layout_seed),
            k=2.0 / np.sqrt(max(len(UG), 1)),
            iterations=60,
        )

    tot_deg = np.array([G.degree(n) for n in range(k)], dtype=np.float32)
    node_colors = tot_deg

    def append_edge(
        r: int, c: int, xs: List[Optional[float]], ys: List[Optional[float]], zs: List[Optional[float]]
    ) -> None:
        pr, pc = pos[r], pos[c]
        xs.extend([float(pr[0]), float(pc[0]), None])
        ys.extend([float(pr[1]), float(pc[1]), None])
        if dim == 3:
            zs.extend([float(pr[2]), float(pc[2]), None])

    xp, yp, zp = [], [], []
    xn, yn, zn = [], [], []
    for r, c, w in zip(loc_r, loc_c, weights):
        if w >= 0:
            append_edge(int(r), int(c), xp, yp, zp)
        else:
            append_edge(int(r), int(c), xn, yn, zn)

    traces: List = []
    line_kw_p = dict(color="rgba(40, 160, 90, 0.55)", width=2)
    line_kw_n = dict(color="rgba(200, 60, 80, 0.55)", width=2)
    if dim == 3:
        if xp:
            traces.append(
                go.Scatter3d(
                    x=xp,
                    y=yp,
                    z=zp,
                    mode="lines",
                    line=line_kw_p,
                    name="excitatory (+)",
                    hoverinfo="skip",
                    showlegend=True,
                )
            )
        if xn:
            traces.append(
                go.Scatter3d(
                    x=xn,
                    y=yn,
                    z=zn,
                    mode="lines",
                    line=line_kw_n,
                    name="inhibitory (−)",
                    hoverinfo="skip",
                    showlegend=True,
                )
            )
        xn_ = [float(pos[i][0]) for i in range(k)]
        yn_ = [float(pos[i][1]) for i in range(k)]
        zn_ = [float(pos[i][2]) for i in range(k)]
        hover = [
            (
                f"adjacency index <b>{idx_order[i]}</b><br>total degree (subgraph) {int(tot_deg[i])}"
                + (
                    f"<br>nm ≈ ({xn_[i]:.0f}, {yn_[i]:.0f}, {zn_[i]:.0f})"
                    if layout_mode == "brain"
                    else ""
                )
            )
            for i in range(k)
        ]
        traces.append(
            go.Scatter3d(
                x=xn_,
                y=yn_,
                z=zn_,
                mode="markers",
                name="neurons",
                marker=dict(
                    size=6 + 5 * (node_colors / (node_colors.max() + 1e-6)),
                    color=node_colors,
                    colorscale="Viridis",
                    colorbar=dict(title="degree"),
                    line=dict(width=0.4, color="rgba(0,0,0,0.35)"),
                ),
                text=hover,
                hoverinfo="text",
            )
        )
        ax_nm = layout_mode == "brain"
        scene = dict(
            xaxis=dict(
                showbackground=False,
                showgrid=True,
                zeroline=False,
                title="x (nm)" if ax_nm else "",
            ),
            yaxis=dict(
                showbackground=False,
                showgrid=True,
                zeroline=False,
                title="y (nm)" if ax_nm else "",
            ),
            zaxis=dict(
                showbackground=False,
                showgrid=True,
                zeroline=False,
                title="z (nm)" if ax_nm else "",
            ),
            aspectmode="data",
        )
    else:
        if xp:
            traces.append(
                go.Scatter(
                    x=xp,
                    y=yp,
                    mode="lines",
                    line=line_kw_p,
                    name="excitatory (+)",
                    hoverinfo="skip",
                )
            )
        if xn:
            traces.append(
                go.Scatter(
                    x=xn,
                    y=yn,
                    mode="lines",
                    line=line_kw_n,
                    name="inhibitory (−)",
                    hoverinfo="skip",
                )
            )
        xn_ = [float(pos[i][0]) for i in range(k)]
        yn_ = [float(pos[i][1]) for i in range(k)]
        hover = [
            (
                f"adjacency index <b>{idx_order[i]}</b><br>total degree (subgraph) {int(tot_deg[i])}"
                + (
                    f"<br>x,y nm ≈ ({xn_[i]:.0f}, {yn_[i]:.0f})"
                    if layout_mode == "brain"
                    else ""
                )
            )
            for i in range(k)
        ]
        traces.append(
            go.Scatter(
                x=xn_,
                y=yn_,
                mode="markers",
                name="neurons",
                marker=dict(
                    size=8 + 6 * (node_colors / (node_colors.max() + 1e-6)),
                    color=node_colors,
                    colorscale="Viridis",
                    colorbar=dict(title="degree"),
                    line=dict(width=0.5, color="rgba(0,0,0,0.4)"),
                ),
                text=hover,
                hoverinfo="text",
            )
        )
        scene = None

    fig = go.Figure(data=traces)
    if layout_mode == "brain":
        layout_desc = "FlyWire L2 bbox centers (nm; rough centroid per neuron)"
    else:
        layout_desc = "force-directed (no position cache or disabled)"
    title = f"Local wiring (n={k} neurons, {loc_r.size} directed edges) — {layout_desc}"
    layout_kw = dict(
        title=dict(text=title, font=dict(size=14)),
        margin=dict(l=0, r=0, t=48, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_dark",
        height=560,
    )
    if scene is not None:
        layout_kw["scene"] = scene
    elif layout_mode == "brain":
        layout_kw["xaxis_title"] = "x (nm)"
        layout_kw["yaxis_title"] = "y (nm)"
    fig.update_layout(**layout_kw)
    return fig
