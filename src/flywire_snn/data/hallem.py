from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split


@dataclass
class DatasetSplits:
    train_x: torch.Tensor
    train_y: torch.Tensor
    val_x: torch.Tensor
    val_y: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor
    heldout_x: torch.Tensor
    heldout_y: torch.Tensor
    feature_dim: int
    num_classes: int


def _default_dataset_path(data_dir: Path) -> Path:
    return data_dir / "raw" / "hallem_carlson_2006.csv"


def load_hallem_base_matrix(data_dir: Path) -> np.ndarray:
    """Odor × receptor matrix (float32) for CV; row index = global class id."""
    df = _load_or_build_base_matrix(data_dir)
    return df.drop(columns=["odor"]).to_numpy(dtype=np.float32)


def _load_or_build_base_matrix(data_dir: Path) -> pd.DataFrame:
    csv_path = _default_dataset_path(data_dir)
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if "odor" not in df.columns:
            df = df.rename(columns={df.columns[0]: "odor"})
        return df

    # CPU-friendly fallback: random matrix with matching paper dimensions.
    rng = np.random.default_rng(7)
    odors = [f"odor_{i:03d}" for i in range(110)]
    cols = [f"receptor_{j:02d}" for j in range(24)]
    x = rng.normal(loc=0.0, scale=1.0, size=(110, 24)).astype(np.float32)
    df = pd.DataFrame(x, columns=cols)
    df.insert(0, "odor", odors)
    return df


def _build_trials(
    base_x: np.ndarray,
    labels: np.ndarray,
    trials_per_class: int,
    noise_std: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    n_classes, n_features = base_x.shape
    xs = np.repeat(base_x, repeats=trials_per_class, axis=0)
    ys = np.repeat(labels, repeats=trials_per_class, axis=0)
    noise = rng.normal(0.0, noise_std, size=(n_classes * trials_per_class, n_features))
    return (xs + noise).astype(np.float32), ys.astype(np.int64)


def load_odor_dataset(
    data_dir: Path,
    train_trials_per_odor: int,
    val_trials_per_odor: int,
    test_trials_per_odor: int,
    heldout_fraction: float,
    noise_std: float,
    seed: int,
) -> DatasetSplits:
    rng = np.random.default_rng(seed)
    df = _load_or_build_base_matrix(data_dir)
    odor_names = df["odor"].astype(str).to_numpy()
    base_x = df.drop(columns=["odor"]).to_numpy(dtype=np.float32)
    y = np.arange(len(odor_names), dtype=np.int64)

    train_classes, heldout_classes = train_test_split(
        y, test_size=heldout_fraction, random_state=seed, shuffle=True
    )

    train_base_x = base_x[train_classes]
    heldout_base_x = base_x[heldout_classes]

    mean = train_base_x.mean(axis=0, keepdims=True)
    std = train_base_x.std(axis=0, keepdims=True) + 1e-6
    norm_base_x = (base_x - mean) / std

    train_base_x = norm_base_x[train_classes]
    heldout_base_x = norm_base_x[heldout_classes]

    train_x, train_y = _build_trials(
        train_base_x,
        train_classes,
        train_trials_per_odor,
        noise_std,
        rng,
    )
    val_x, val_y = _build_trials(
        train_base_x,
        train_classes,
        val_trials_per_odor,
        noise_std,
        rng,
    )
    test_x, test_y = _build_trials(
        train_base_x,
        train_classes,
        test_trials_per_odor,
        noise_std,
        rng,
    )
    heldout_x, heldout_y = _build_trials(
        heldout_base_x,
        heldout_classes,
        max(test_trials_per_odor, 1),
        noise_std,
        rng,
    )

    return DatasetSplits(
        train_x=torch.from_numpy(train_x),
        train_y=torch.from_numpy(train_y),
        val_x=torch.from_numpy(val_x),
        val_y=torch.from_numpy(val_y),
        test_x=torch.from_numpy(test_x),
        test_y=torch.from_numpy(test_y),
        heldout_x=torch.from_numpy(heldout_x),
        heldout_y=torch.from_numpy(heldout_y),
        feature_dim=base_x.shape[1],
        num_classes=base_x.shape[0],
    )


def summarize_dataset_source(data_dir: Path) -> Dict[str, str]:
    csv_path = _default_dataset_path(data_dir)
    return {
        "path": str(csv_path),
        "source": "local_csv" if csv_path.exists() else "synthetic_fallback",
    }

