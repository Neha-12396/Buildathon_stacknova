from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from .architectures import ZoneSequenceLSTM
    from .training_utils import SEQUENCE_LENGTH, build_prediction_dataset
except ImportError:
    from architectures import ZoneSequenceLSTM
    from training_utils import SEQUENCE_LENGTH, build_prediction_dataset

BASE_DIR = Path(__file__).resolve().parent
PREDICTION_MODEL_PATH = BASE_DIR / "prediction_model.pt"
PREDICTION_MODEL_META_PATH = BASE_DIR / "prediction_model_meta.json"

EPOCHS = int(os.getenv("PREDICTION_MODEL_EPOCHS", "12"))
BATCH_SIZE = int(os.getenv("PREDICTION_MODEL_BATCH_SIZE", "32"))
PATIENCE = int(os.getenv("PREDICTION_MODEL_PATIENCE", "4"))


def _evaluate(model: ZoneSequenceLSTM, loader: DataLoader, criterion: nn.Module) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            total_loss += float(loss.item()) * len(batch_x)
            total_correct += int((logits.argmax(dim=1) == batch_y).sum().item())
            total_examples += len(batch_x)

    return (
        total_loss / max(total_examples, 1),
        total_correct / max(total_examples, 1),
    )


def train_prediction_model() -> dict:
    torch.manual_seed(42)
    np.random.seed(42)

    sequences, labels, zone_to_index = build_prediction_dataset(sequence_length=SEQUENCE_LENGTH)
    x_train, x_val, y_train, y_val = train_test_split(
        sequences,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val)),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = ZoneSequenceLSTM(num_zones=len(zone_to_index), embedding_dim=16, hidden_dim=48)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    best_state = None
    best_accuracy = 0.0
    patience = 0

    for epoch in range(EPOCHS):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

        val_loss, val_accuracy = _evaluate(model, val_loader, criterion)
        print(
            f"[prediction] epoch {epoch + 1}/{EPOCHS} "
            f"val_loss={val_loss:.4f} val_accuracy={val_accuracy:.4f}"
        )

        if val_accuracy >= best_accuracy:
            best_accuracy = val_accuracy
            best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                break

    if best_state is None:
        best_state = model.state_dict()

    model.load_state_dict(best_state)
    PREDICTION_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, PREDICTION_MODEL_PATH)

    metadata = {
        "zone_to_index": zone_to_index,
        "sequence_length": SEQUENCE_LENGTH,
        "embedding_dim": 16,
        "hidden_dim": 48,
        "sample_count": int(len(sequences)),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "validation_accuracy": round(float(best_accuracy), 4),
    }
    with PREDICTION_MODEL_META_PATH.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return {
        "validation_accuracy": round(float(best_accuracy), 4),
        "sample_count": int(len(sequences)),
        "artifact": str(PREDICTION_MODEL_PATH),
    }


if __name__ == "__main__":
    print(json.dumps(train_prediction_model(), indent=2))
