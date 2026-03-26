import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp

from flywire_snn.connectome.auth import _strip_unreadable_path_entries

logger = logging.getLogger(__name__)

NT_SIGN = {
    "acetylcholine": 1.0,
    "gaba": -1.0,
    "glutamate": -1.0,
    "dopamine": 1.0,
    "serotonin": 1.0,
    "octopamine": 1.0,
}


def positions_npz_path(cache_path: Path) -> Path:
    """Sidecar for neuron coordinates (nm), rows aligned with connectome matrix."""
    return cache_path.with_name(f"{cache_path.stem}_positions.npz")


def load_connectome_positions(
    cache_path: Path,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load cached positions and root ids if present. Shapes (N,3) and (N,)."""
    p = positions_npz_path(cache_path)
    if not p.exists():
        return None, None
    try:
        z = np.load(p)
        pos = z["positions_nm"]
        rids = z["root_ids"]
        return pos, rids
    except Exception as exc:
        logger.warning("Could not load positions cache %s: %s", p, exc)
        return None, None


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


def _bounds_nm_to_centroid(bounds_nm) -> np.ndarray:
    """Center of L2 bounding box in nanometers; NaNs if missing."""
    out = np.full(3, np.nan, dtype=np.float64)
    if bounds_nm is None:
        return out
    try:
        b = list(bounds_nm)
    except TypeError:
        return out
    if len(b) < 6:
        return out
    out[0] = (float(b[0]) + float(b[1])) / 2.0
    out[1] = (float(b[2]) + float(b[3])) / 2.0
    out[2] = (float(b[4]) + float(b[5])) / 2.0
    return out


def fetch_l2_centroids_nm_ordered(
    root_ids: List[int],
    dataset: str,
    *,
    progress: bool = False,
) -> np.ndarray:
    """
    For each root_id (in list order), centroid of FlyWire L2 `bounds_nm` from get_l2_info.

    Units: **nanometers** (same as FlyWire / CAVE). Requires API access + token.
    """
    _strip_unreadable_path_entries()
    from fafbseg import flywire

    flywire.set_default_dataset(dataset)
    info = flywire.get_l2_info(list(root_ids), progress=progress, dataset=dataset)
    by_id: Dict[int, np.ndarray] = {}
    for _, row in info.iterrows():
        rid = int(row["root_id"])
        by_id[rid] = _bounds_nm_to_centroid(row.get("bounds_nm"))
    out = np.zeros((len(root_ids), 3), dtype=np.float32)
    nan3 = np.full(3, np.nan, dtype=np.float32)
    for i, rid in enumerate(root_ids):
        c = by_id.get(int(rid))
        if c is None:
            out[i] = nan3
        else:
            out[i] = c.astype(np.float32)
    return out


def _save_positions_cache(cache_path: Path, positions_nm: np.ndarray, root_ids: List[int]) -> None:
    p = positions_npz_path(cache_path)
    np.savez_compressed(
        p,
        positions_nm=np.asarray(positions_nm, dtype=np.float32),
        root_ids=np.asarray(root_ids, dtype=np.int64),
    )


def _try_fetch_and_save_positions(
    cache_path: Path,
    root_ids: List[int],
    dataset: str,
    *,
    progress: bool = False,
) -> Tuple[bool, Optional[str]]:
    """Returns (ok, error_message)."""
    try:
        pos = fetch_l2_centroids_nm_ordered(root_ids, dataset, progress=progress)
        _save_positions_cache(cache_path, pos, root_ids)
        return True, None
    except Exception as exc:
        return False, str(exc)


def _maybe_backfill_positions(
    cache_path: Path,
    meta: Dict,
    n_matrix: int,
    dataset: str,
    materialization: str,
    fetch_positions: bool,
) -> None:
    """If matrix cache exists without positions but meta has neuron_ids, fetch once."""
    if not fetch_positions:
        return
    pos_path = positions_npz_path(cache_path)
    if pos_path.exists():
        return
    ids = meta.get("neuron_ids")
    if not ids or len(ids) != n_matrix:
        return
    root_ids = [int(x) for x in ids]
    ok, err = _try_fetch_and_save_positions(cache_path, root_ids, dataset, progress=False)
    if ok:
        logger.info("Backfilled neuron position cache at %s", pos_path)
    else:
        logger.warning("Could not backfill positions cache: %s", err)


def _query_neuron_ids(max_neurons: int, dataset: str, materialization: str) -> List[int]:
    _strip_unreadable_path_entries()
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
    _strip_unreadable_path_entries()
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
    fetch_positions: bool = True,
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
            _maybe_backfill_positions(
                cache_path, meta, cached_n, dataset, materialization, fetch_positions
            )
            pos_path = positions_npz_path(cache_path)
            has_pos = pos_path.exists()
            with meta_path.open("r", encoding="utf-8") as f:
                meta2 = json.load(f)
            return m.tocsr(), {
                "source": "cache",
                "neurons": cached_n,
                "edges": int(m.nnz),
                "has_positions": has_pos,
                "positions_path": str(pos_path) if has_pos else None,
                "positions_units": "nm" if has_pos else None,
            }

    try:
        ids = _query_neuron_ids(max_neurons=max_neurons, dataset=dataset, materialization=materialization)
        edges = _query_edges(ids, materialization=materialization, dataset=dataset)
        mat, stats = _build_sparse_from_edges(ids, edges)
        if mat.nnz == 0:
            raise RuntimeError("FlyWire query returned zero in-subgraph edges.")
        sp.save_npz(cache_path, mat)

        has_positions = False
        positions_note = None
        if fetch_positions:
            ok, err = _try_fetch_and_save_positions(cache_path, ids, dataset, progress=True)
            has_positions = ok
            positions_note = err
            if ok:
                logger.info("Saved neuron positions to %s", positions_npz_path(cache_path))
            else:
                logger.warning("FlyWire connectome OK but L2 positions failed: %s", err)

        meta_out = {
            "dataset": dataset,
            "materialization": materialization,
            "max_neurons": int(max_neurons),
            "source": "flywire",
            "neurons": int(mat.shape[0]),
            "edges": int(mat.nnz),
            "neuron_ids": [int(x) for x in ids],
            "has_positions": has_positions,
            "positions_units": "nm" if has_positions else None,
            "positions_file": positions_npz_path(cache_path).name if has_positions else None,
        }
        if positions_note and not has_positions:
            meta_out["positions_error"] = positions_note

        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta_out, f, indent=2)
        return mat, {
            "source": "flywire",
            "has_positions": has_positions,
            "positions_path": str(positions_npz_path(cache_path)) if has_positions else None,
            "positions_units": "nm" if has_positions else None,
            **stats,
        }
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
                    "has_positions": False,
                    "neuron_ids": None,
                },
                f,
                indent=2,
            )
        return m, {
            "source": "random_fallback",
            "neurons": int(m.shape[0]),
            "edges": int(m.nnz),
            "error": str(exc),
            "has_positions": False,
            "positions_path": None,
        }
