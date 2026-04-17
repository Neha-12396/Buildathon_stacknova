from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

try:
    from .model_loader import load_audio_bundle
    from .training_utils import (
        MAX_AUDIO_SECONDS,
        MAX_AUDIO_SAMPLES,
        SAMPLE_RATE,
        spectrogram_from_waveform,
    )
except ImportError:
    from model_loader import load_audio_bundle
    from training_utils import (
        MAX_AUDIO_SECONDS,
        MAX_AUDIO_SAMPLES,
        SAMPLE_RATE,
        spectrogram_from_waveform,
    )

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CLASS_NAMES = ["chainsaw", "gunshot", "safe", "vehicle"]

_MODEL = None
_MODEL_LOAD_ERROR = None


def _bundle_class_names() -> list[str]:
    bundle = load_audio_bundle()
    if not bundle:
        return list(DEFAULT_CLASS_NAMES)
    return [
        label
        for _, label in sorted(bundle["index_to_label"].items(), key=lambda item: item[0])
    ]


CLASS_NAMES = _bundle_class_names()


def _preload_model():
    global _MODEL, _MODEL_LOAD_ERROR
    if _MODEL is not None or _MODEL_LOAD_ERROR is not None:
        return

    bundle = load_audio_bundle()
    if not bundle:
        _MODEL_LOAD_ERROR = (
            f"Audio model artifact not found in {BASE_DIR}. "
            "Run backend/model/train.py to create audio_model.pt."
        )
        return

    _MODEL = bundle


def load_model():
    _preload_model()
    if _MODEL is None:
        raise FileNotFoundError(_MODEL_LOAD_ERROR or "Audio model bundle could not be loaded.")
    return _MODEL


def _decode_audio(raw_bytes: bytes, filename: str | None = None):
    try:
        waveform, sample_rate = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=False)
        if getattr(waveform, "ndim", 1) > 1:
            waveform = waveform.mean(axis=1)
        if sample_rate != SAMPLE_RATE:
            waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
            sample_rate = SAMPLE_RATE
        return waveform[:MAX_AUDIO_SAMPLES], sample_rate, None
    except Exception:
        pass

    suffix = Path(filename).suffix if filename else ".wav"
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(raw_bytes)
            temp_path = temp_file.name
        waveform, sample_rate = librosa.load(temp_path, sr=SAMPLE_RATE, mono=True)
        return waveform[:MAX_AUDIO_SAMPLES], sample_rate, None
    except Exception as exc:
        return None, None, exc
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def _heuristic_label(waveform: np.ndarray | None, sample_rate: int | None):
    if waveform is None or sample_rate is None or not len(waveform):
        return None

    rms = float(librosa.feature.rms(y=waveform).mean())
    centroid = float(librosa.feature.spectral_centroid(y=waveform, sr=sample_rate).mean())
    flatness = float(librosa.feature.spectral_flatness(y=waveform).mean())
    onset = float(librosa.onset.onset_strength(y=waveform, sr=sample_rate).mean())
    zcr = float(librosa.feature.zero_crossing_rate(y=waveform).mean())

    if centroid > 2500 and onset > 8 and zcr > 0.08:
        return "gunshot"
    if centroid > 1800 and zcr > 0.05 and rms > 0.015:
        return "chainsaw"
    if centroid > 1200 and onset > 2:
        return "vehicle"
    if flatness > 0.15:
        return "safe"
    return "safe"


def infer_audio(audio_bytes: bytes, filename: str | None = None):
    waveform, sample_rate, decode_error = _decode_audio(audio_bytes, filename=filename)
    heuristic_label = _heuristic_label(waveform, sample_rate)

    if waveform is None or sample_rate is None:
        return {
            "label": heuristic_label or "safe",
            "confidence": 0.0,
            "probabilities": [],
            "heuristic_label": heuristic_label,
            "audio_length": 0.0,
            "decode_error": str(decode_error) if decode_error else None,
            "model_error": "Audio decode failed",
        }

    spectrogram = spectrogram_from_waveform(waveform, sample_rate=sample_rate)
    audio_length = len(waveform) / float(sample_rate) if sample_rate else 0.0

    try:
        bundle = load_model()
        model = bundle["model"]
        input_tensor = torch.from_numpy(spectrogram).unsqueeze(0).unsqueeze(0).float()
        with torch.no_grad():
            logits = model(input_tensor)
            prediction = torch.softmax(logits, dim=1)[0].cpu().numpy().astype(np.float32)

        predicted_index = int(np.argmax(prediction))
        label = bundle["index_to_label"][predicted_index]
        confidence = float(np.max(prediction))
        probabilities = prediction.tolist()
        model_error = None
    except Exception as exc:
        label = heuristic_label or "safe"
        confidence = 0.0
        probabilities = []
        model_error = str(exc)

    return {
        "label": label,
        "confidence": confidence,
        "probabilities": probabilities,
        "heuristic_label": heuristic_label,
        "audio_length": min(audio_length, MAX_AUDIO_SECONDS),
        "decode_error": str(decode_error) if decode_error else None,
        "model_error": model_error,
    }


_preload_model()
