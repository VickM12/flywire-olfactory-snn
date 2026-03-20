from __future__ import annotations

import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from flywire_snn.config import ExperimentConfig
from flywire_snn.connectome.flywire_graph import load_or_build_connectome
from flywire_snn.data.door import build_or_merge_door_matrix, summarize_door_source
from flywire_snn.data.hallem import load_hallem_base_matrix, summarize_dataset_source
from flywire_snn.data.splits import (
    build_splits_for_outer_fold_trials,
    make_outer_fold_indices,
)
from flywire_snn.models.dense_mlp import DenseMLP
from flywire_snn.models.shuffled_snn import ShuffledSNN
from flywire_snn.models.sparse_mlp import recurrent_sparsity_ratio, SparseMLP
from flywire_snn.models.snn import MaskedRecurrentLIFSNN
from flywire_snn.trainers import train_model
from flywire_snn.utils import ensure_dir, save_json, set_seed


def _parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def resolve_edge_count(conn_meta: Dict[str, Any], connectome: sp.csr_matrix) -> int:
    for key in ("edges", "edges_kept"):
        v = conn_meta.get(key)
        if v is not None:
            return int(v)
    return int(connectome.nnz)


def _init_seed_for_model(global_seed: int, model_name: str) -> None:
    salt = {"ConnectomeSNN": 0, "ShuffledSNN": 11, "SparseMLP": 22, "DenseMLP": 33}.get(model_name, 0)
    set_seed(global_seed + salt * 97)


def _build_models(
    cfg: ExperimentConfig,
    connectome: sp.csr_matrix,
    feature_dim: int,
    num_classes: int,
    run_seed: int,
    fold: int,
) -> Dict[str, nn.Module]:
    hidden_dim = int(connectome.shape[0])
    rho = recurrent_sparsity_ratio(connectome)
    shuffle_seed = cfg.base_shuffle_seed + run_seed * 10_007 + fold * 17
    sparse_seed = cfg.base_sparse_seed + run_seed * 30_011 + fold * 19

    connectome_snn = MaskedRecurrentLIFSNN(
        input_dim=feature_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        adjacency=connectome,
        steps=cfg.snn_steps,
        alpha=cfg.snn_alpha,
    )
    p_snn = _parameter_count(connectome_snn)

    return {
        "ConnectomeSNN": connectome_snn,
        "ShuffledSNN": ShuffledSNN(
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            adjacency=connectome,
            shuffle_seed=shuffle_seed,
            steps=cfg.snn_steps,
            alpha=cfg.snn_alpha,
        ),
        "SparseMLP": SparseMLP(
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            sparsity_ratio=rho,
            seed=sparse_seed,
        ),
        "DenseMLP": DenseMLP.matched_to_connectome_snn(feature_dim, num_classes, p_snn),
    }


def _aggregate(values: List[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(arr)), float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0


def _aggregate_epochs(vals: List[int]) -> Tuple[float, float]:
    usable = [float(v) for v in vals if v > 0]
    if not usable:
        return float("nan"), float("nan")
    return _aggregate(usable)


def _fmt_ms(d: Dict[str, Any]) -> str:
    m, s = d.get("mean"), d.get("std")
    if m is None or (isinstance(m, float) and math.isnan(m)):
        return "N/A"
    if s is None or (isinstance(s, float) and math.isnan(s)):
        s = 0.0
    return f"{m:.2f} ± {s:.2f}"


def _fmt_ms1(d: Dict[str, Any]) -> str:
    m, s = d.get("mean"), d.get("std")
    if m is None or (isinstance(m, float) and math.isnan(m)):
        return "N/A"
    if s is None or (isinstance(s, float) and math.isnan(s)):
        s = 0.0
    return f"{m:.1f} ± {s:.1f}"


def format_summary_table(summary_dataset: Dict[str, Any]) -> str:
    lines = [
        "| Model         | Test Acc      | Epochs to 80% | Spike Sparsity | Params |",
        "|---------------|---------------|---------------|----------------|--------|",
    ]
    order = ["ConnectomeSNN", "ShuffledSNN", "SparseMLP", "DenseMLP"]
    for name in order:
        if name not in summary_dataset:
            continue
        a = summary_dataset[name]
        ta_str = _fmt_ms(a.get("test_acc", {}))
        e80_str = _fmt_ms1(a.get("epochs_to_80", {}))
        if name in ("SparseMLP", "DenseMLP"):
            sp_str = "N/A"
        else:
            sp_str = _fmt_ms(a.get("spike_sparsity", {}))
        params = int(a.get("params", 0))
        lines.append(
            f"| {name:13s} | {ta_str:13s} | {e80_str:13s} | {sp_str:14s} | {params:6d} |"
        )
    return "\n".join(lines)


def run_experiment(cfg: ExperimentConfig) -> Dict[str, object]:
    logger = logging.getLogger(__name__)
    ensure_dir(cfg.data_dir / "processed")
    ensure_dir(cfg.result_dir)

    connectome_path = cfg.data_dir / "processed" / cfg.connectome_cache
    connectome, conn_meta = load_or_build_connectome(
        cache_path=connectome_path,
        max_neurons=cfg.max_olfactory_neurons,
        dataset=cfg.annotation_dataset,
        materialization=cfg.materialization,
        force_rebuild=cfg.rebuild_connectome,
    )
    n_edges = resolve_edge_count(conn_meta, connectome)
    conn_log = dict(conn_meta)
    conn_log["edges"] = n_edges

    logger.info(
        "Connectome source=%s neurons=%s edges=%s",
        conn_log.get("source"),
        conn_log.get("neurons"),
        n_edges,
    )
    if cfg.require_real_connectome:
        src = str(conn_meta.get("source", ""))
        err = conn_meta.get("error", None)
        # A cached connectome can still be the real FlyWire one; only fail if the
        # cache was generated from fallback or has an explicit error recorded.
        if src not in ("flywire", "cache") or err:
            msg = (
                f"Using non-FlyWire connectome source={conn_meta.get('source')}: "
                f"{conn_meta.get('error', 'no error details')}"
            )
            raise RuntimeError(msg)
    else:
        if conn_meta.get("source") not in ("flywire", "cache"):
            msg = (
                f"Using non-FlyWire connectome source={conn_meta.get('source')}: "
                f"{conn_meta.get('error', 'no error details')}"
            )
            logger.warning(msg)

    door_base, _, _ = build_or_merge_door_matrix(cfg.data_dir, force_refresh=cfg.refresh_door_cache)
    hallem_base = load_hallem_base_matrix(cfg.data_dir)

    datasets: List[Tuple[str, np.ndarray]] = [("DoOR", door_base)]
    if cfg.run_hallem_secondary:
        datasets.append(("HallemCarlson", hallem_base))

    per_run_rows: List[Dict[str, Any]] = []
    model_names = ["ConnectomeSNN", "ShuffledSNN", "SparseMLP", "DenseMLP"]

    for dataset_name, base_x in datasets:
        n_classes = int(base_x.shape[0])
        for seed_i in range(cfg.n_seeds):
            run_seed = cfg.seed + seed_i * 1_003
            for fold in range(cfg.n_cv_folds):
                train_idx_outer, test_idx = make_outer_fold_indices(
                    n_classes=n_classes,
                    fold=fold,
                    n_folds=cfg.n_cv_folds,
                    seed=run_seed,
                )
                ds = build_splits_for_outer_fold_trials(
                    base_x=base_x,
                    train_idx_outer=train_idx_outer,
                    test_idx=test_idx,
                    train_trials=cfg.train_trials_per_odor,
                    val_trials=cfg.val_trials_per_odor,
                    test_trials=cfg.test_trials_per_odor,
                    noise_std=cfg.noise_std,
                    seed=run_seed + fold,
                )

                models = _build_models(cfg, connectome, ds.feature_dim, ds.num_classes, run_seed, fold)

                for mname in model_names:
                    _init_seed_for_model(run_seed, mname)
                    model = models[mname]
                    result = train_model(
                        model=model,
                        train_x=ds.train_x,
                        train_y=ds.train_y,
                        val_x=ds.val_x,
                        val_y=ds.val_y,
                        test_x=ds.test_x,
                        test_y=ds.test_y,
                        epochs=cfg.epochs,
                        batch_size=cfg.batch_size,
                        lr=cfg.lr,
                        weight_decay=cfg.weight_decay,
                        model_name=f"{dataset_name}/{mname}",
                        early_stopping_patience=cfg.early_stopping_patience,
                    )
                    per_run_rows.append(
                        {
                            "dataset": dataset_name,
                            "fold": fold,
                            "seed": run_seed,
                            "model": mname,
                            "test_acc": result.test_acc,
                            "epochs_to_80_val": result.epochs_to_80,
                            "stopped_epoch": result.stopped_epoch,
                            "best_val_acc": result.best_val_acc,
                            "spike_sparsity": result.final_spike_sparsity,
                            "params": _parameter_count(model),
                        }
                    )

    by_ds_model: Dict[str, Dict[str, List[Any]]] = defaultdict(lambda: defaultdict(list))
    for row in per_run_rows:
        by_ds_model[row["dataset"]][row["model"]].append(row)

    summary: Dict[str, Any] = {}
    for ds_name, models_dict in by_ds_model.items():
        agg: Dict[str, Any] = {}
        for mname, rows in models_dict.items():
            agg[mname] = {
                "test_acc": _aggregate([r["test_acc"] for r in rows]),
                "epochs_to_80": _aggregate_epochs([r["epochs_to_80_val"] for r in rows]),
                "stopped_epoch": _aggregate([float(r["stopped_epoch"]) for r in rows]),
                "spike_sparsity": _aggregate([r["spike_sparsity"] for r in rows])
                if mname in ("ConnectomeSNN", "ShuffledSNN")
                else (float("nan"), float("nan")),
                "params": int(round(np.mean([r["params"] for r in rows]))),
            }
        summary[ds_name] = agg

    def _stat_pair(t: Tuple[float, float]) -> Dict[str, Any]:
        m, s = t
        return {
            "mean": None if isinstance(m, float) and math.isnan(m) else m,
            "std": None if isinstance(s, float) and math.isnan(s) else s,
        }

    summary_json: Dict[str, Any] = {}
    for ds_name, models_dict in summary.items():
        summary_json[ds_name] = {}
        for mname, met in models_dict.items():
            summary_json[ds_name][mname] = {
                "test_acc": _stat_pair(met["test_acc"]),
                "epochs_to_80": _stat_pair(met["epochs_to_80"]),
                "stopped_epoch": _stat_pair(met["stopped_epoch"]),
                "spike_sparsity": _stat_pair(met["spike_sparsity"]),
                "params": met["params"],
            }

    cfg_payload = {k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.__dict__.items()}

    payload: Dict[str, object] = {
        "config": cfg_payload,
        "connectome": conn_log,
        "datasets": {
            "DoOR": summarize_door_source(cfg.data_dir),
            "HallemCarlson": summarize_dataset_source(cfg.data_dir),
        },
        "summary": summary_json,
        "per_run": per_run_rows,
    }

    save_json(cfg.result_dir / "comparison.json", payload)
    logger.info("Saved comparison JSON to %s", cfg.result_dir / "comparison.json")
    return payload
