import os
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import scipy.sparse as sp


NT_SIGN = {
    "acetylcholine": 1.0,
    "gaba": -1.0,
    "glutamate": -1.0,
    "dopamine": 1.0,
    "serotonin": 1.0,
    "octopamine": 1.0,
}


def _sanitize_path_for_navis() -> None:
    # navis probes PATH entries for external tools; some Windows entries can be unreadable.
    raw = os.environ.get("PATH", "")
    if not raw:
        return
    safe_entries = []
    for entry in raw.split(os.pathsep):
        if not entry:
            continue
        try:
            _ = Path(entry).exists()
            safe_entries.append(entry)
        except (PermissionError, OSError):
            continue
    os.environ["PATH"] = os.pathsep.join(safe_entries)


def _signed_weight(nt_label: str, weight: float) -> float:
    sign = NT_SIGN.get(str(nt_label).lower(), 1.0)
    return sign * float(weight)


def _build_sparse_from_edges(
    neuron_ids: List[int], edges
) -> Tuple[sp.csr_matrix, Dict[str, int]]:
    id_to_idx = {rid: i for i, rid in enumerate(neuron_ids)}
    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []
    kept = 0
    for _, r in edges.iterrows():
        pre = int(r["pre"])
        post = int(r["post"])
        if pre not in id_to_idx or post not in id_to_idx:
            continue
        nt = r.get("transmitter", r.get("pred_nt", "acetylcholine"))
        w = _signed_weight(str(nt), float(r["weight"]))
        rows.append(id_to_idx[post])
        cols.append(id_to_idx[pre])
        vals.append(w)
        kept += 1
    n = len(neuron_ids)
    mat = sp.coo_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float32).tocsr()
    return mat, {"edges_kept": kept, "neurons": n}


def _query_neuron_ids(max_neurons: int, dataset: str, materialization: str) -> List[int]:
    _sanitize_path_for_navis()
    from fafbseg import flywire

    flywire.set_default_dataset(dataset)
    mat = 630 if dataset == "public" and str(materialization) == "auto" else materialization
    nc = flywire.NeuronCriteria

    pns = flywire.search_annotations(nc(cell_class="ALPN"), materialization=mat)
    kcs = flywire.search_annotations(
        nc(cell_class="Kenyon_Cell"), materialization=mat
    )
    pn_ids = pns["root_id"].astype(np.int64).tolist()
    kc_ids = kcs["root_id"].astype(np.int64).tolist()
    ids = pn_ids + kc_ids
    if len(ids) > max_neurons:
        ids = ids[:max_neurons]
    return ids


def _query_edges(ids: Iterable[int], materialization: str, dataset: str):
    _sanitize_path_for_navis()
    from fafbseg import flywire
    mat = 630 if dataset == "public" and str(materialization) == "auto" else materialization

    return flywire.get_connectivity(
        list(ids),
        upstream=False,
        downstream=True,
        transmitters=True,
        materialization=mat,
        filtered=True,
        min_score=50,
    )


def _random_fallback_graph(n: int = 800, density: float = 0.01, seed: int = 7) -> sp.csr_matrix:
    rng = np.random.default_rng(seed)
    nnz = int(n * n * density)
    rows = rng.integers(0, n, size=nnz)
    cols = rng.integers(0, n, size=nnz)
    signs = rng.choice([-1.0, 1.0], size=nnz, p=[0.2, 0.8])
    mags = rng.gamma(shape=1.5, scale=0.6, size=nnz)
    vals = (signs * mags).astype(np.float32)
    return sp.coo_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float32).tocsr()


def load_or_build_connectome(
    cache_path: Path,
    max_neurons: int,
    dataset: str = "public",
    materialization: str = "auto",
    force_rebuild: bool = False,
) -> Tuple[sp.csr_matrix, Dict[str, object]]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = cache_path.with_suffix(".meta.json")
    if cache_path.exists() and meta_path.exists() and not force_rebuild:
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        m = sp.load_npz(cache_path)
        cached_n = int(m.shape[0])
        if (
            cached_n == int(max_neurons)
            and meta.get("dataset") == dataset
            and str(meta.get("materialization")) == str(materialization)
        ):
            return m.tocsr(), {"source": "cache", "neurons": cached_n, "edges": int(m.nnz)}

    try:
        ids = _query_neuron_ids(max_neurons=max_neurons, dataset=dataset, materialization=materialization)
        edges = _query_edges(ids, materialization=materialization, dataset=dataset)
        m, stats = _build_sparse_from_edges(ids, edges)
        if m.nnz == 0:
            raise RuntimeError("FlyWire query returned zero in-subgraph edges.")
        sp.save_npz(cache_path, m)
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "dataset": dataset,
                    "materialization": materialization,
                    "max_neurons": int(max_neurons),
                    "source": "flywire",
                    "neurons": int(m.shape[0]),
                    "edges": int(m.nnz),
                },
                f,
                indent=2,
            )
        return m, {"source": "flywire", **stats}
    except Exception as exc:  # pragma: no cover - fallback for offline/auth issues
        m = _random_fallback_graph(n=max_neurons)
        sp.save_npz(cache_path, m)
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "dataset": dataset,
                    "materialization": materialization,
                    "max_neurons": int(max_neurons),
                    "source": "random_fallback",
                    "neurons": int(m.shape[0]),
                    "edges": int(m.nnz),
                    "error": str(exc),
                },
                f,
                indent=2,
            )
        return m, {
            "source": "random_fallback",
            "neurons": int(m.shape[0]),
            "edges": int(m.nnz),
            "error": str(exc),
        }

