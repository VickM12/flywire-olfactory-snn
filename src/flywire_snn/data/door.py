"""Load merged odor × receptor matrix from ropensci/DoOR.data (Or receptor CSVs)."""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

DOOR_RAW = "https://raw.githubusercontent.com/ropensci/DoOR.data/master/data/"

# Subset of Or genes with per-receptor CSVs in DoOR.data (olfactory benchmark).
DOOR_OR_FILES: List[str] = [
    "Or10a.csv",
    "Or13a.csv",
    "Or19a.csv",
    "Or1a.csv",
    "Or22a.csv",
    "Or22b.csv",
    "Or22c.csv",
    "Or23a.csv",
    "Or24a.csv",
    "Or2a.csv",
    "Or30a.csv",
    "Or33a.csv",
    "Or33b.csv",
    "Or33c.csv",
    "Or35a.csv",
    "Or42a.csv",
    "Or42b.csv",
    "Or43a.csv",
    "Or43b.csv",
    "Or45a.csv",
    "Or45b.csv",
    "Or46a.csv",
    "Or47a.csv",
    "Or47b.csv",
    "Or49a.csv",
    "Or49b.csv",
    "Or59a.csv",
    "Or59b.csv",
    "Or59c.csv",
    "Or65a.csv",
    "Or67a.csv",
    "Or67b.csv",
    "Or67c.csv",
    "Or67d.csv",
    "Or69a.csv",
    "Or71a.csv",
    "Or74a.csv",
    "Or7a.csv",
    "Or82a.csv",
    "Or83c.csv",
    "Or85a.csv",
    "Or85b.csv",
    "Or85c.csv",
    "Or85d.csv",
    "Or85e.csv",
    "Or85f.csv",
    "Or88a.csv",
    "Or92a.csv",
    "Or94a.csv",
    "Or94b.csv",
    "Or98a.csv",
    "Or9a.csv",
]


def _odor_key_row(df: pd.DataFrame) -> pd.Series:
    if "InChIKey" in df.columns:
        k = df["InChIKey"].astype(str)
        if "Name" in df.columns:
            k = k.replace("nan", np.nan).fillna(df["Name"].astype(str))
        return k
    return df.iloc[:, 2].astype(str)


def _median_response(df: pd.DataFrame) -> pd.Series:
    num = df.iloc[:, 5:].apply(pd.to_numeric, errors="coerce")
    return num.median(axis=1)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)


def build_or_merge_door_matrix(
    data_dir: Path,
    receptor_files: List[str] | None = None,
    force_refresh: bool = False,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Returns base_x (n_odors, n_receptors), list of odor keys, receptor names.
    Cached at data/processed/door_or_merged.parquet (or .csv fallback).
    """
    proc = data_dir / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    cache_csv = proc / "door_or_merged.csv"

    if not force_refresh and cache_csv.exists():
        tbl = pd.read_csv(cache_csv)
        odors = tbl["odor_key"].astype(str).tolist()
        rec_cols = [c for c in tbl.columns if c != "odor_key"]
        return tbl[rec_cols].to_numpy(dtype=np.float32), odors, rec_cols

    cache_dir = data_dir / "raw" / "door"
    files = receptor_files or DOOR_OR_FILES
    merged: Dict[str, Dict[str, float]] = {}
    rec_names: List[str] = []

    for fname in files:
        rec = fname.replace(".csv", "")
        rec_names.append(rec)
        url = DOOR_RAW + fname
        local = cache_dir / fname
        if not local.exists():
            _download(url, local)
        df = pd.read_csv(local, sep=";")
        keys = _odor_key_row(df)
        resp = _median_response(df)
        for k, v in zip(keys, resp):
            if k in ("nan", "NaN", "SFR") or (isinstance(k, float) and np.isnan(k)):
                continue
            merged.setdefault(str(k), {})[rec] = float(v) if not pd.isna(v) else np.nan

    odors_sorted = sorted(merged.keys())
    wide = np.full((len(odors_sorted), len(rec_names)), np.nan, dtype=np.float64)
    for i, odor in enumerate(odors_sorted):
        row = merged[odor]
        for j, r in enumerate(rec_names):
            wide[i, j] = row.get(r, np.nan)

    col_med = np.nanmedian(wide, axis=0)
    inds = np.where(np.isnan(wide))
    wide[inds] = np.take(col_med, inds[1])
    wide = np.nan_to_num(wide, nan=0.0).astype(np.float32)

    out_tbl = pd.DataFrame(wide, columns=rec_names)
    out_tbl.insert(0, "odor_key", odors_sorted)
    out_tbl.to_csv(cache_csv, index=False)

    return wide.astype(np.float32), odors_sorted, rec_names


def summarize_door_source(data_dir: Path) -> Dict[str, str]:
    proc = data_dir / "processed"
    if (proc / "door_or_merged.csv").exists():
        return {"source": "door_merged_cache", "path": str(proc / "door_or_merged.csv")}
    return {"source": "door_build_on_first_run", "path": str(data_dir / "raw" / "door")}
