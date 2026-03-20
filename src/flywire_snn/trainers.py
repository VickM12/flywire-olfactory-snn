from dataclasses import dataclass
import logging
from typing import Dict, List, Tuple

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


def _accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return float((pred == y).float().mean().item())


@torch.no_grad()
def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int) -> Tuple[float, float]:
    model.eval()
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)
    accs = []
    sparsities = []
    for xb, yb in loader:
        logits, sparsity = model(xb)
        accs.append(_accuracy(logits, yb))
        sparsities.append(float(sparsity.item()))
    return float(sum(accs) / max(len(accs), 1)), float(sum(sparsities) / max(len(sparsities), 1))


def train_model(
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    heldout_x: torch.Tensor,
    heldout_y: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    model_name: str = "model",
) -> TrainResult:
    logger = logging.getLogger(__name__)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)

    history: List[Dict[str, float]] = []
    epochs_to_80 = -1

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_sparse = 0.0
        num_batches = 0
        for xb, yb in train_loader:
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
        val_acc, _ = evaluate(model, val_x, val_y, batch_size=batch_size)
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

    test_acc, test_sparse = evaluate(model, test_x, test_y, batch_size=batch_size)
    heldout_acc, _ = evaluate(model, heldout_x, heldout_y, batch_size=batch_size)
    logger.info(
        "[%s] final test_acc=%.4f heldout_acc=%.4f spike_sparsity=%.4f epochs_to_80=%d",
        model_name,
        test_acc,
        heldout_acc,
        test_sparse,
        epochs_to_80,
    )

    return TrainResult(
        history=history,
        test_acc=test_acc,
        heldout_acc=heldout_acc,
        epochs_to_80=epochs_to_80,
        final_spike_sparsity=test_sparse,
    )

