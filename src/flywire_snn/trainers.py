from dataclasses import dataclass
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrainResult:
    history: List[Dict[str, float]]
    test_acc: float
    heldout_acc: float
    epochs_to_80: int
    final_spike_sparsity: float
    stopped_epoch: int
    best_val_acc: float
    best_state_dict: Dict[str, torch.Tensor] | None = None


def _accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return float((pred == y).float().mean().item())


def _state_dict_cpu_clone(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _load_state(model: nn.Module, state: Dict[str, torch.Tensor]) -> None:
    model.load_state_dict({k: v.to(next(model.parameters()).device) for k, v in state.items()})


@torch.no_grad()
def evaluate(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    mc_samples: int = 1,
    restrict_classes: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> Tuple[float, float]:
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    if restrict_classes is not None:
        restrict_classes = restrict_classes.to(torch.long).to(device)
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)
    accs = []
    sparsities = []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        if restrict_classes is not None:
            y_rel = torch.searchsorted(restrict_classes, yb.to(torch.long))
        if mc_samples <= 1:
            logits, sparsity = model(xb)
            logits = torch.nan_to_num(logits, nan=-1e9)
            if restrict_classes is not None:
                logits = logits[:, restrict_classes]
                accs.append(_accuracy(logits, y_rel))
            else:
                accs.append(_accuracy(logits, yb))
            sparsities.append(float(sparsity.item()))
        else:
            logits_sum = None
            sp_sum = 0.0
            for _ in range(mc_samples):
                logits_i, sparsity_i = model(xb)
                logits_i = torch.nan_to_num(logits_i, nan=-1e9)
                logits_sum = logits_i if logits_sum is None else logits_sum + logits_i
                sp_sum += float(sparsity_i.item())
            logits_mean = logits_sum / float(mc_samples)
            if restrict_classes is not None:
                logits_mean = logits_mean[:, restrict_classes]
                accs.append(_accuracy(logits_mean, y_rel))
            else:
                accs.append(_accuracy(logits_mean, yb))
            sparsities.append(sp_sum / float(mc_samples))
    return float(sum(accs) / max(len(accs), 1)), float(sum(sparsities) / max(len(sparsities), 1))


def train_model(
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    model_name: str = "model",
    early_stopping_patience: int = 5,
    heldout_x: Optional[torch.Tensor] = None,
    heldout_y: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> TrainResult:
    logger = logging.getLogger(__name__)
    if device is None:
        device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)

    history: List[Dict[str, float]] = []
    epochs_to_80 = -1
    best_val = -1.0
    best_epoch = 0
    # Always have a checkpoint so final evaluation never accidentally uses
    # "last epoch" weights when val_acc never improves (e.g. NaNs).
    best_state: Dict[str, torch.Tensor] = _state_dict_cpu_clone(model)
    patience_left = early_stopping_patience
    stopped_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_sparse = 0.0
        num_batches = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits, sparsity = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            epoch_acc += _accuracy(logits.detach(), yb)
            epoch_sparse += float(sparsity.item())
            num_batches += 1

        train_acc = epoch_acc / max(num_batches, 1)
        train_sparse = epoch_sparse / max(num_batches, 1)
        mc = 5 if "SNN" in model_name else 1
        val_acc, _ = evaluate(model, val_x, val_y, batch_size=batch_size, mc_samples=mc, device=device)
        if epochs_to_80 < 0 and val_acc >= 0.8:
            epochs_to_80 = epoch

        history.append(
            {
                "epoch": float(epoch),
                "loss": epoch_loss / max(num_batches, 1),
                "train_acc": train_acc,
                "val_acc": val_acc,
                "spike_sparsity": train_sparse,
            }
        )
        logger.info(
            "[%s] epoch %d/%d loss=%.4f train_acc=%.4f val_acc=%.4f spike_sparsity=%.4f",
            model_name,
            epoch,
            epochs,
            history[-1]["loss"],
            train_acc,
            val_acc,
            train_sparse,
        )

        if val_acc > best_val + 1e-6:
            best_val = val_acc
            best_state = _state_dict_cpu_clone(model)
            best_epoch = epoch
            patience_left = early_stopping_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                stopped_epoch = epoch
                break
    else:
        stopped_epoch = epochs

    _load_state(model, best_state)

    mc = 5 if "SNN" in model_name else 1
    test_acc, test_sparse = evaluate(model, test_x, test_y, batch_size=batch_size, mc_samples=mc, device=device)
    if heldout_x is not None and heldout_y is not None:
        candidates = torch.unique(heldout_y).sort().values
        heldout_acc, _ = evaluate(
            model,
            heldout_x,
            heldout_y,
            batch_size=batch_size,
            mc_samples=mc,
            restrict_classes=candidates,
            device=device,
        )
    else:
        heldout_acc = float("nan")

    ha_str = f"{heldout_acc:.4f}" if not math.isnan(heldout_acc) else "nan"
    logger.info(
        "[%s] final test_acc=%.4f heldout_acc=%s spike_sparsity=%.4f epochs_to_80=%d stopped_epoch=%d best_val=%.4f best_epoch=%d",
        model_name,
        test_acc,
        ha_str,
        test_sparse,
        epochs_to_80,
        stopped_epoch,
        best_val,
        best_epoch,
    )

    return TrainResult(
        history=history,
        test_acc=test_acc,
        heldout_acc=heldout_acc,
        epochs_to_80=epochs_to_80,
        final_spike_sparsity=test_sparse,
        stopped_epoch=stopped_epoch,
        best_val_acc=best_val,
        best_state_dict=best_state,
    )
