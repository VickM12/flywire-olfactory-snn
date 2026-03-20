from pathlib import Path
import logging
from typing import Dict

import scipy.sparse as sp

from flywire_snn.config import ExperimentConfig
from flywire_snn.connectome.flywire_graph import load_or_build_connectome
from flywire_snn.data.hallem import load_odor_dataset, summarize_dataset_source
from flywire_snn.models.mlp import BaselineMLP
from flywire_snn.models.snn import MaskedRecurrentLIFSNN
from flywire_snn.trainers import train_model
from flywire_snn.utils import ensure_dir, save_json, set_seed


def _parameter_count(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_experiment(cfg: ExperimentConfig) -> Dict[str, object]:
    logger = logging.getLogger(__name__)
    set_seed(cfg.seed)
    ensure_dir(cfg.data_dir / "processed")
    ensure_dir(cfg.result_dir)
    logger.info("Starting experiment with seed=%d epochs=%d batch_size=%d", cfg.seed, cfg.epochs, cfg.batch_size)

    connectome_path = cfg.data_dir / "processed" / cfg.connectome_cache
    connectome, conn_meta = load_or_build_connectome(
        cache_path=connectome_path,
        max_neurons=cfg.max_olfactory_neurons,
        dataset=cfg.annotation_dataset,
        materialization=cfg.materialization,
        force_rebuild=cfg.rebuild_connectome,
    )
    hidden_dim = int(connectome.shape[0])
    logger.info(
        "Connectome source=%s neurons=%s edges=%s",
        conn_meta.get("source"),
        conn_meta.get("neurons"),
        conn_meta.get("edges"),
    )

    ds = load_odor_dataset(
        data_dir=cfg.data_dir,
        train_trials_per_odor=cfg.train_trials_per_odor,
        val_trials_per_odor=cfg.val_trials_per_odor,
        test_trials_per_odor=cfg.test_trials_per_odor,
        heldout_fraction=cfg.heldout_odor_fraction,
        noise_std=cfg.noise_std,
        seed=cfg.seed,
    )
    logger.info("Dataset feature_dim=%d classes=%d", ds.feature_dim, ds.num_classes)

    snn = MaskedRecurrentLIFSNN(
        input_dim=ds.feature_dim,
        hidden_dim=hidden_dim,
        num_classes=ds.num_classes,
        adjacency=connectome,
        steps=cfg.snn_steps,
        alpha=cfg.snn_alpha,
    )
    mlp = BaselineMLP(
        input_dim=ds.feature_dim,
        hidden_dim=hidden_dim,
        num_classes=ds.num_classes,
    )

    snn_result = train_model(
        model=snn,
        train_x=ds.train_x,
        train_y=ds.train_y,
        val_x=ds.val_x,
        val_y=ds.val_y,
        test_x=ds.test_x,
        test_y=ds.test_y,
        heldout_x=ds.heldout_x,
        heldout_y=ds.heldout_y,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        model_name="snn",
    )
    mlp_result = train_model(
        model=mlp,
        train_x=ds.train_x,
        train_y=ds.train_y,
        val_x=ds.val_x,
        val_y=ds.val_y,
        test_x=ds.test_x,
        test_y=ds.test_y,
        heldout_x=ds.heldout_x,
        heldout_y=ds.heldout_y,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        model_name="mlp",
    )

    cfg_payload = {
        k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.__dict__.items()
    }

    payload = {
        "config": cfg_payload,
        "connectome": conn_meta,
        "dataset": summarize_dataset_source(cfg.data_dir),
        "models": {
            "snn": {
                "trainable_params": _parameter_count(snn),
                "test_accuracy": snn_result.test_acc,
                "heldout_accuracy": snn_result.heldout_acc,
                "epochs_to_80_val": snn_result.epochs_to_80,
                "spike_sparsity": snn_result.final_spike_sparsity,
            },
            "mlp": {
                "trainable_params": _parameter_count(mlp),
                "test_accuracy": mlp_result.test_acc,
                "heldout_accuracy": mlp_result.heldout_acc,
                "epochs_to_80_val": mlp_result.epochs_to_80,
                "spike_sparsity": mlp_result.final_spike_sparsity,
            },
        },
        "history": {
            "snn": snn_result.history,
            "mlp": mlp_result.history,
        },
    }

    save_json(cfg.result_dir / "comparison.json", payload)
    logger.info("Saved comparison JSON to %s", cfg.result_dir / "comparison.json")
    return payload

