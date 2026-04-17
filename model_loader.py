from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import torch

try:
    from .architectures import AudioCNN, ZoneSequenceLSTM
except ImportError:
    from architectures import AudioCNN, ZoneSequenceLSTM

BASE_DIR = Path(__file__).resolve().parent
DEVICE = torch.device("cpu")

AUDIO_MODEL_PATH = BASE_DIR / "audio_model.pt"
AUDIO_MODEL_META_PATH = BASE_DIR / "audio_model_meta.json"
CONTEXT_MODEL_PATH = BASE_DIR / "context_model.pkl"
PREDICTION_MODEL_PATH = BASE_DIR / "prediction_model.pt"
PREDICTION_MODEL_META_PATH = BASE_DIR / "prediction_model_meta.json"
TRAINING_REPORT_PATH = BASE_DIR / "training_report.json"


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def load_audio_bundle() -> dict | None:
    if not AUDIO_MODEL_PATH.exists() or not AUDIO_MODEL_META_PATH.exists():
        return None

    metadata = _load_json(AUDIO_MODEL_META_PATH)
    label_to_index = {
        str(label): int(index)
        for label, index in metadata.get("label_to_index", {}).items()
    }
    index_to_label = {index: label for label, index in label_to_index.items()}

    checkpoint = torch.load(AUDIO_MODEL_PATH, map_location=DEVICE)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model = AudioCNN(num_classes=len(label_to_index))
    model.load_state_dict(state_dict)
    model.eval()

    return {
        "model": model,
        "metadata": metadata,
        "label_to_index": label_to_index,
        "index_to_label": index_to_label,
    }


@lru_cache(maxsize=1)
def load_context_model():
    if not CONTEXT_MODEL_PATH.exists():
        return None
    try:
        return joblib.load(CONTEXT_MODEL_PATH)
    except Exception as exc:
        print(f"Context model unavailable, falling back to heuristic context scoring: {exc}")
        return None


@lru_cache(maxsize=1)
def load_prediction_bundle() -> dict | None:
    if not PREDICTION_MODEL_PATH.exists() or not PREDICTION_MODEL_META_PATH.exists():
        return None

    metadata = _load_json(PREDICTION_MODEL_META_PATH)
    zone_to_index = {
        str(zone_name): int(index)
        for zone_name, index in metadata.get("zone_to_index", {}).items()
    }
    index_to_zone = {index: zone_name for zone_name, index in zone_to_index.items()}

    checkpoint = torch.load(PREDICTION_MODEL_PATH, map_location=DEVICE)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model = ZoneSequenceLSTM(
        num_zones=len(zone_to_index),
        embedding_dim=int(metadata.get("embedding_dim", 16)),
        hidden_dim=int(metadata.get("hidden_dim", 48)),
    )
    model.load_state_dict(state_dict)
    model.eval()

    return {
        "model": model,
        "metadata": metadata,
        "zone_to_index": zone_to_index,
        "index_to_zone": index_to_zone,
        "sequence_length": int(metadata.get("sequence_length", 6)),
    }


def clear_model_caches():
    load_audio_bundle.cache_clear()
    load_context_model.cache_clear()
    load_prediction_bundle.cache_clear()
