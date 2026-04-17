from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

try:
    from .model_loader import TRAINING_REPORT_PATH, clear_model_caches
    from .train_audio import train_audio_model
    from .train_context import train_context_model
    from .train_prediction import train_prediction_model
except ImportError:
    from model_loader import TRAINING_REPORT_PATH, clear_model_caches
    from train_audio import train_audio_model
    from train_context import train_context_model
    from train_prediction import train_prediction_model


def train_all_models() -> dict:
    report = {
        "trained_at": datetime.utcnow().isoformat(),
        "audio": train_audio_model(),
        "context": train_context_model(),
        "prediction": train_prediction_model(),
    }
    TRAINING_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TRAINING_REPORT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    clear_model_caches()
    return report


def main():
    print(json.dumps(train_all_models(), indent=2))


if __name__ == "__main__":
    main()
