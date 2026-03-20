"""Shared odor-level CV splits and trial tensors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from sklearn.model_selection import KFold


@dataclass
class DatasetSplits:
    train_x: torch.Tensor
    train_y: torch.Tensor
    val_x: torch.Tensor
    val_y: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor
    feature_dim: int
    num_classes: int


def _build_trials(
    base_x: np.ndarray,
    labels: np.ndarray,
    trials_per_split: int,
    noise_std: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    n_classes, n_features = base_x.shape
    xs = np.repeat(base_x, repeats=trials_per_split, axis=0)
    ys = np.repeat(labels, repeats=trials_per_split, axis=0)
    noise = rng.normal(0.0, noise_std, size=(n_classes * trials_per_split, n_features))
    return (xs + noise).astype(np.float32), ys.astype(np.int64)


def make_fold_indices(
    n_classes: int,
    fold: int,
    n_folds: int,
    seed: int,
    val_fraction: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits: List[Tuple[np.ndarray, np.ndarray]] = list(kf.split(np.arange(n_classes)))
    train_val_idx, test_idx = splits[fold]
    rng = np.random.default_rng(seed + 10_003 + fold)
    order = rng.permutation(train_val_idx)
    n_val = max(1, int(len(order) * val_fraction))
    val_idx = order[:n_val]
    train_idx = order[n_val:]
    return train_idx, val_idx, test_idx


def make_outer_fold_indices(
    n_classes: int,
    fold: int,
    n_folds: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Outer CV split over odor identities.

    We hold out some odors entirely for test evaluation, but for early stopping we
    still validate on different trials of odors seen in training (so val classes are
    not unseen).
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits: List[Tuple[np.ndarray, np.ndarray]] = list(kf.split(np.arange(n_classes)))
    train_idx_outer, test_idx = splits[fold]
    return train_idx_outer, test_idx


def build_splits_for_fold(
    base_x: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    train_trials: int,
    val_trials: int,
    test_trials: int,
    noise_std: float,
    seed: int,
) -> DatasetSplits:
    rng = np.random.default_rng(seed)
    train_base = base_x[train_idx]
    val_base = base_x[val_idx]
    test_base = base_x[test_idx]

    mean = train_base.mean(axis=0, keepdims=True)
    std = train_base.std(axis=0, keepdims=True) + 1e-6
    train_base_n = (train_base - mean) / std
    val_base_n = (val_base - mean) / std
    test_base_n = (test_base - mean) / std

    train_labels = train_idx.astype(np.int64)
    val_labels = val_idx.astype(np.int64)
    test_labels = test_idx.astype(np.int64)

    train_x, train_y = _build_trials(train_base_n, train_labels, train_trials, noise_std, rng)
    val_x, val_y = _build_trials(val_base_n, val_labels, val_trials, noise_std, rng)
    test_x, test_y = _build_trials(test_base_n, test_labels, test_trials, noise_std, rng)

    return DatasetSplits(
        train_x=torch.from_numpy(train_x),
        train_y=torch.from_numpy(train_y),
        val_x=torch.from_numpy(val_x),
        val_y=torch.from_numpy(val_y),
        test_x=torch.from_numpy(test_x),
        test_y=torch.from_numpy(test_y),
        feature_dim=int(base_x.shape[1]),
        num_classes=int(base_x.shape[0]),
    )


def build_splits_for_outer_fold_trials(
    base_x: np.ndarray,
    train_idx_outer: np.ndarray,
    test_idx: np.ndarray,
    train_trials: int,
    val_trials: int,
    test_trials: int,
    noise_std: float,
    seed: int,
) -> DatasetSplits:
    """Build train/val/test datasets for an outer CV fold.

    - Outer train odors = `train_idx_outer` (labels are seen during training)
    - Outer test odors = `test_idx` (true generalization evaluation)
    - Validation set uses different noisy trials from the same outer train odors.
    """
    rng = np.random.default_rng(seed)
    train_base = base_x[train_idx_outer]
    test_base = base_x[test_idx]

    mean = train_base.mean(axis=0, keepdims=True)
    std = train_base.std(axis=0, keepdims=True) + 1e-6
    train_base_n = (train_base - mean) / std
    test_base_n = (test_base - mean) / std

    # Val uses the same odor identities as training (different trial noise).
    train_labels = train_idx_outer.astype(np.int64)
    val_labels = train_idx_outer.astype(np.int64)
    test_labels = test_idx.astype(np.int64)

    train_x, train_y = _build_trials(
        train_base_n, train_labels, train_trials, noise_std, rng
    )
    val_x, val_y = _build_trials(train_base_n, val_labels, val_trials, noise_std, rng)
    test_x, test_y = _build_trials(test_base_n, test_labels, test_trials, noise_std, rng)

    return DatasetSplits(
        train_x=torch.from_numpy(train_x),
        train_y=torch.from_numpy(train_y),
        val_x=torch.from_numpy(val_x),
        val_y=torch.from_numpy(val_y),
        test_x=torch.from_numpy(test_x),
        test_y=torch.from_numpy(test_y),
        feature_dim=int(base_x.shape[1]),
        num_classes=int(base_x.shape[0]),
    )


def build_splits_for_outer_fold_trials_seen_and_heldout(
    base_x: np.ndarray,
    train_idx_outer: np.ndarray,
    test_idx: np.ndarray,
    train_trials: int,
    val_trials: int,
    test_trials: int,
    noise_std: float,
    seed: int,
) -> Tuple[DatasetSplits, torch.Tensor, torch.Tensor]:
    """Outer CV split returning both seen-test and heldout-test sets.

    - Train/val/test are built from *outer train odors* (seen class identities).
    - heldout is built from *outer test odors* (unseen class identities) to
      measure generalization.
    """
    train_base = base_x[train_idx_outer]
    heldout_base = base_x[test_idx]

    mean = train_base.mean(axis=0, keepdims=True)
    std = train_base.std(axis=0, keepdims=True) + 1e-6
    train_base_n = (train_base - mean) / std
    heldout_base_n = (heldout_base - mean) / std

    train_labels = train_idx_outer.astype(np.int64)
    heldout_labels = test_idx.astype(np.int64)

    rng_train = np.random.default_rng(seed)
    rng_val = np.random.default_rng(seed + 12345)
    rng_test = np.random.default_rng(seed + 54321)
    rng_heldout = np.random.default_rng(seed + 99991)

    train_x, train_y = _build_trials(train_base_n, train_labels, train_trials, noise_std, rng_train)
    val_x, val_y = _build_trials(train_base_n, train_labels, val_trials, noise_std, rng_val)
    test_x, test_y = _build_trials(train_base_n, train_labels, test_trials, noise_std, rng_test)
    heldout_x, heldout_y = _build_trials(heldout_base_n, heldout_labels, test_trials, noise_std, rng_heldout)

    ds = DatasetSplits(
        train_x=torch.from_numpy(train_x),
        train_y=torch.from_numpy(train_y),
        val_x=torch.from_numpy(val_x),
        val_y=torch.from_numpy(val_y),
        test_x=torch.from_numpy(test_x),
        test_y=torch.from_numpy(test_y),
        feature_dim=int(base_x.shape[1]),
        num_classes=int(base_x.shape[0]),
    )
    return ds, torch.from_numpy(heldout_x), torch.from_numpy(heldout_y)
