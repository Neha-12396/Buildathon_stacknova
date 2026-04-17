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
    from .architectures import AudioCNN
    from .training_utils import (
        CANONICAL_LABELS,
        N_MELS,
        TARGET_FRAMES,
        build_audio_dataset,
    )
except ImportError:
    from architectures import AudioCNN
    from training_utils import CANONICAL_LABELS, N_MELS, TARGET_FRAMES, build_audio_dataset

BASE_DIR = Path(__file__).resolve().parent
AUDIO_MODEL_PATH = BASE_DIR / "audio_model.pt"
AUDIO_MODEL_META_PATH = BASE_DIR / "audio_model_meta.json"

EPOCHS = int(os.getenv("AUDIO_MODEL_EPOCHS", "8"))
BATCH_SIZE = int(os.getenv("AUDIO_MODEL_BATCH_SIZE", "16"))
MAX_PER_LABEL = int(os.getenv("AUDIO_MAX_PER_LABEL", "140"))
PATIENCE = int(os.getenv("AUDIO_MODEL_PATIENCE", "3"))


def _evaluate(model: AudioCNN, loader: DataLoader, criterion: nn.Module) -> tuple[float, float]:
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


def train_audio_model() -> dict:
    torch.manual_seed(42)
    np.random.seed(42)

    features, labels, label_counts = build_audio_dataset(max_per_label=MAX_PER_LABEL)
    label_to_index = {label: index for index, label in enumerate(CANONICAL_LABELS)}
    y = np.asarray([label_to_index[label] for label in labels], dtype=np.int64)

    x_train, x_val, y_train, y_val = train_test_split(
        features,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
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

    model = AudioCNN(num_classes=len(CANONICAL_LABELS))
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
            f"[audio] epoch {epoch + 1}/{EPOCHS} "
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
    AUDIO_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, AUDIO_MODEL_PATH)

    metadata = {
        "label_to_index": label_to_index,
        "class_counts": label_counts,
        "sample_count": int(len(features)),
        "spectrogram_shape": [N_MELS, TARGET_FRAMES],
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "validation_accuracy": round(float(best_accuracy), 4),
    }
    with AUDIO_MODEL_META_PATH.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return {
        "validation_accuracy": round(float(best_accuracy), 4),
        "sample_count": int(len(features)),
        "class_counts": label_counts,
        "artifact": str(AUDIO_MODEL_PATH),
    }


if __name__ == "__main__":
    print(json.dumps(train_audio_model(), indent=2))
