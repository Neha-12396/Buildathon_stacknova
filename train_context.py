from __future__ import annotations

import json
from pathlib import Path

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from .training_utils import BACKEND_DATA_DIR, build_context_training_frame
except ImportError:
    from training_utils import BACKEND_DATA_DIR, build_context_training_frame

BASE_DIR = Path(__file__).resolve().parent
CONTEXT_MODEL_PATH = BASE_DIR / "context_model.pkl"
CONTEXT_DATA_PATH = BACKEND_DATA_DIR / "context_training.csv"


def train_context_model() -> dict:
    frame = build_context_training_frame()
    CONTEXT_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(CONTEXT_DATA_PATH, index=False)

    features = frame[["zone", "hour", "threat_type", "recent_count"]]
    labels = frame["risk"]

    x_train, x_val, y_train, y_val = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), ["zone", "threat_type"]),
            ("numeric", "passthrough", ["hour", "recent_count"]),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_val)
    accuracy = accuracy_score(y_val, predictions)

    joblib.dump(model, CONTEXT_MODEL_PATH)
    return {
        "validation_accuracy": round(float(accuracy), 4),
        "sample_count": int(len(frame)),
        "artifact": str(CONTEXT_MODEL_PATH),
        "dataset": str(CONTEXT_DATA_PATH),
    }


if __name__ == "__main__":
    print(json.dumps(train_context_model(), indent=2))
