from __future__ import annotations

import json
import random
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

try:
    from services.geo_agent import GeoAgent
except ImportError:
    from backend.services.geo_agent import GeoAgent

BASE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = BASE_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
BACKEND_DATA_DIR = BACKEND_DIR / "data"
SQLITE_SEED_DIR = BACKEND_DATA_DIR / "sqlite_seed"

SAMPLE_RATE = 16000
MAX_AUDIO_SECONDS = 3
MAX_AUDIO_SAMPLES = SAMPLE_RATE * MAX_AUDIO_SECONDS
N_MELS = 64
TARGET_FRAMES = 96
SEQUENCE_LENGTH = 6
CANONICAL_LABELS = ["chainsaw", "gunshot", "safe", "vehicle"]

AUDIO_DIR_CANDIDATES = [
    DATASET_DIR / "ESC-50-master" / "audio",
    DATASET_DIR / "audio",
    DATASET_DIR / "audio" / "audio",
    DATASET_DIR / "audio" / "audio" / "16000",
]
CSV_CANDIDATES = [
    DATASET_DIR / "ESC-50-master" / "meta" / "esc50.csv",
    DATASET_DIR / "esc50.csv",
]
GUNSHOT_DIR = DATASET_DIR / "gunshot_samples"

TARGET_CLASSES = {
    "chainsaw": "chainsaw",
    "hand_saw": "chainsaw",
    "engine": "vehicle",
    "helicopter": "vehicle",
    "train": "vehicle",
    "car_horn": "vehicle",
    "fireworks": "gunshot",
}
SAFE_CLASSES = {
    "rain",
    "wind",
    "crickets",
    "chirping_birds",
    "insects",
    "frog",
    "crow",
    "pouring_water",
    "water_drops",
}
THREAT_WEIGHTS = {
    "safe": 0.08,
    "uncertain": 0.36,
    "vehicle": 0.58,
    "chainsaw": 0.74,
    "gunshot": 0.86,
}
ZONE_WEIGHTS = {
    "Safari Zone": 0.03,
    "Core Forest": 0.09,
    "Elephant Corridor": 0.12,
    "Entry Gate": 0.01,
}

_GEO_AGENT = GeoAgent()


def canonical_zones() -> list[str]:
    return list(_GEO_AGENT.zones.keys())


def zone_graph() -> dict[str, list[str]]:
    return {
        zone_name: list(zone_data["neighbors"])
        for zone_name, zone_data in _GEO_AGENT.zones.items()
    }


def normalize_zone_name(zone: str | None) -> str:
    if not zone:
        return "Safari Zone"
    normalized = _GEO_AGENT._normalize_zone_key(zone)
    if normalized:
        return normalized
    return "Safari Zone"


def normalize_risk_label(risk_level: str | None) -> str:
    text = str(risk_level or "").strip().lower()
    if text in {"critical", "high"}:
        return "high"
    if text == "medium":
        return "medium"
    return "low"


def risk_score_to_label(score: float) -> str:
    if score >= 0.72:
        return "high"
    if score >= 0.42:
        return "medium"
    return "low"


def parse_timestamp(value: str | None) -> datetime:
    if not value:
        return datetime.utcnow()
    cleaned = str(value).strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(cleaned)
        if parsed.tzinfo is not None:
            return parsed.astimezone().replace(tzinfo=None)
        return parsed
    except ValueError:
        return datetime.utcnow()


def compute_context_score(
    zone: str,
    hour: int,
    threat_type: str,
    recent_count: int,
    location: str | None = None,
) -> float:
    score = THREAT_WEIGHTS.get(threat_type, THREAT_WEIGHTS["uncertain"])
    score += ZONE_WEIGHTS.get(zone, 0.02)
    score += min(max(recent_count, 0), 6) * 0.055
    if hour >= 18 or hour < 6:
        score += 0.08
    if location and any(token in location.lower() for token in ("deep", "ridge", "corridor")):
        score += 0.04
    return float(np.clip(score, 0.0, 1.0))


def _pad_audio(waveform: np.ndarray) -> np.ndarray:
    if len(waveform) >= MAX_AUDIO_SAMPLES:
        return waveform[:MAX_AUDIO_SAMPLES]
    return np.pad(waveform, (0, MAX_AUDIO_SAMPLES - len(waveform)))


def spectrogram_from_waveform(waveform: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    waveform = _pad_audio(waveform)
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_fft=1024,
        hop_length=512,
        n_mels=N_MELS,
    )
    log_mel = librosa.power_to_db(mel, ref=1.0)
    if log_mel.shape[1] < TARGET_FRAMES:
        pad_width = TARGET_FRAMES - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode="constant")
    elif log_mel.shape[1] > TARGET_FRAMES:
        log_mel = log_mel[:, :TARGET_FRAMES]

    log_mel = log_mel - np.min(log_mel)
    max_value = float(np.max(log_mel))
    if max_value > 0:
        log_mel = log_mel / max_value
    return log_mel.astype(np.float32)


def load_audio_waveform(file_path: Path) -> np.ndarray:
    waveform, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    return waveform.astype(np.float32)


def _find_existing_path(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"None of the dataset candidates exist: {', '.join(str(path) for path in candidates)}"
    )


def _find_audio_dir(candidates: list[Path], filenames: list[str]) -> Path:
    filename_set = {str(name) for name in filenames}
    for candidate in candidates:
        if not candidate.exists() or not candidate.is_dir():
            continue
        try:
            available_files = {item.name for item in candidate.iterdir() if item.is_file()}
        except OSError:
            continue
        if available_files & filename_set:
            return candidate
    raise RuntimeError("Unable to locate a usable ESC-50 audio directory.")


def synthesize_gunshot_waveform(seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    duration = float(rng.uniform(0.25, 0.8))
    sample_count = int(duration * SAMPLE_RATE)
    timeline = np.linspace(0.0, duration, sample_count, endpoint=False)
    base_freq = float(rng.uniform(450.0, 1800.0))
    decay = float(rng.uniform(6.0, 14.0))

    impulse = np.exp(-decay * timeline)
    tone = np.sin(2 * np.pi * base_freq * timeline) * impulse
    crack = rng.normal(0.0, 0.18, sample_count) * impulse
    low_boom = np.sin(2 * np.pi * rng.uniform(50.0, 150.0) * timeline) * np.exp(-3 * timeline)

    waveform = tone + crack + (0.35 * low_boom)
    if sample_count < MAX_AUDIO_SAMPLES:
        tail = rng.normal(0.0, 0.01, MAX_AUDIO_SAMPLES - sample_count)
        waveform = np.concatenate([waveform, tail])
    waveform = waveform[:MAX_AUDIO_SAMPLES]
    peak = np.max(np.abs(waveform)) or 1.0
    return (waveform / peak).astype(np.float32)


def build_audio_dataset(max_per_label: int = 140) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    csv_path = _find_existing_path(CSV_CANDIDATES)
    metadata = pd.read_csv(csv_path)
    audio_dir = _find_audio_dir(AUDIO_DIR_CANDIDATES, metadata["filename"].tolist())

    features: list[np.ndarray] = []
    labels: list[str] = []
    per_label_counts = {label: 0 for label in CANONICAL_LABELS}

    for _, row in metadata.iterrows():
        mapped_label = TARGET_CLASSES.get(row["category"])
        if not mapped_label and row["category"] in SAFE_CLASSES:
            mapped_label = "safe"
        if not mapped_label or per_label_counts[mapped_label] >= max_per_label:
            continue

        audio_path = audio_dir / row["filename"]
        if not audio_path.exists():
            continue

        try:
            waveform = load_audio_waveform(audio_path)
            features.append(spectrogram_from_waveform(waveform))
            labels.append(mapped_label)
            per_label_counts[mapped_label] += 1
        except Exception:
            continue

    if GUNSHOT_DIR.exists():
        for audio_path in sorted(GUNSHOT_DIR.glob("*.wav")):
            if per_label_counts["gunshot"] >= max_per_label:
                break
            try:
                waveform = load_audio_waveform(audio_path)
                features.append(spectrogram_from_waveform(waveform))
                labels.append("gunshot")
                per_label_counts["gunshot"] += 1
            except Exception:
                continue

    synthetic_seed = 0
    while per_label_counts["gunshot"] < max_per_label:
        waveform = synthesize_gunshot_waveform(seed=synthetic_seed)
        features.append(spectrogram_from_waveform(waveform))
        labels.append("gunshot")
        per_label_counts["gunshot"] += 1
        synthetic_seed += 1

    if len(features) < 80:
        raise RuntimeError("Not enough audio samples were collected to train the CNN.")

    stacked = np.stack(features).astype(np.float32)
    stacked = stacked[:, np.newaxis, :, :]
    return stacked, np.array(labels), per_label_counts


def _load_seed_events() -> list[dict]:
    events_path = SQLITE_SEED_DIR / "events.json"
    if not events_path.exists():
        return []
    with events_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return list(payload.get("events", []))


def load_history_events() -> list[dict]:
    deduped: dict[tuple[str | None, str | None, str | None], dict] = {}
    for record in _load_seed_events():
        event_id = record.get("event_id")
        timestamp = record.get("timestamp")
        audio_file = record.get("audio_file")
        deduped[(event_id, timestamp, audio_file)] = record

    normalized_events = []
    for record in deduped.values():
        zone = normalize_zone_name(
            record.get("zone")
            or (record.get("geo") or {}).get("zone")
            or record.get("zone_id")
        )
        threat_type = str(
            record.get("detected_class")
            or record.get("threat_type")
            or "safe"
        ).strip().lower()
        normalized_events.append(
            {
                **record,
                "normalized_zone": zone,
                "normalized_threat_type": threat_type,
                "normalized_risk_level": normalize_risk_label(record.get("risk_level")),
                "parsed_timestamp": parse_timestamp(record.get("timestamp")),
            }
        )

    normalized_events.sort(key=lambda item: item["parsed_timestamp"])
    return normalized_events


def build_context_training_frame() -> pd.DataFrame:
    records = []
    history_by_zone: defaultdict[str, list[datetime]] = defaultdict(list)

    for event in load_history_events():
        zone = event["normalized_zone"]
        event_time = event["parsed_timestamp"]
        threat_type = event["normalized_threat_type"]

        history_by_zone[zone] = [
            timestamp
            for timestamp in history_by_zone[zone]
            if event_time - timestamp <= timedelta(hours=2)
        ]
        recent_count = len(history_by_zone[zone])
        history_by_zone[zone].append(event_time)

        records.append(
            {
                "zone": zone,
                "hour": event_time.hour,
                "threat_type": threat_type,
                "recent_count": recent_count,
                "risk": normalize_risk_label(event.get("risk_level")),
            }
        )

    for zone in canonical_zones():
        for hour in range(0, 24, 3):
            for threat_type in ("safe", "vehicle", "chainsaw", "gunshot", "uncertain"):
                for recent_count in range(0, 7):
                    score = compute_context_score(zone, hour, threat_type, recent_count)
                    records.append(
                        {
                            "zone": zone,
                            "hour": hour,
                            "threat_type": threat_type,
                            "recent_count": recent_count,
                            "risk": risk_score_to_label(score),
                        }
                    )

    frame = pd.DataFrame.from_records(records)
    return frame.sample(frac=1.0, random_state=42).reset_index(drop=True)


def generate_zone_walks(
    walks_per_zone: int = 80,
    walk_length: int = 14,
    seed: int = 42,
) -> list[list[str]]:
    graph = zone_graph()
    rng = random.Random(seed)
    walks: list[list[str]] = []

    for start_zone in canonical_zones():
        for _ in range(walks_per_zone):
            current = start_zone
            walk = [current]
            for _step in range(walk_length - 1):
                options = [current] + list(graph[current])
                stay_weight = 0.28
                neighbor_weight = (1.0 - stay_weight) / max(len(options) - 1, 1)
                weights = [
                    stay_weight if option == current else neighbor_weight
                    for option in options
                ]
                current = rng.choices(options, weights=weights, k=1)[0]
                walk.append(current)
            walks.append(walk)

    return walks


def build_prediction_dataset(
    sequence_length: int = SEQUENCE_LENGTH,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    walks = generate_zone_walks(walks_per_zone=100, walk_length=sequence_length + 6)
    historic_zones = [event["normalized_zone"] for event in load_history_events()]
    if len(historic_zones) >= sequence_length + 1:
        walks.append(historic_zones)

    zone_to_index = {zone_name: index for index, zone_name in enumerate(canonical_zones())}
    sequences: list[list[int]] = []
    labels: list[int] = []

    for walk in walks:
        if len(walk) <= sequence_length:
            continue
        for index in range(len(walk) - sequence_length):
            sequence = walk[index : index + sequence_length]
            next_zone = walk[index + sequence_length]
            sequences.append([zone_to_index[zone] for zone in sequence])
            labels.append(zone_to_index[next_zone])

    return np.asarray(sequences, dtype=np.int64), np.asarray(labels, dtype=np.int64), zone_to_index
