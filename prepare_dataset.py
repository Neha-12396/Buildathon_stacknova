import os
import shutil
import subprocess
import time
import wave
import numpy as np

DATA_DIR = os.path.dirname(__file__)
GUNSHOT_DIR = os.path.join(DATA_DIR, "gunshot_samples")
ESC50_DIR = os.path.join(DATA_DIR, "ESC-50-master")
ESC50_META = os.path.join(ESC50_DIR, "meta")
ESC50_AUDIO = os.path.join(ESC50_DIR, "audio")
ESC50_KAGGLE_DATASET = os.getenv(
    "ESC50_KAGGLE_DATASET",
    "mmoreaux/environmental-sound-classification-50",
)
os.makedirs(GUNSHOT_DIR, exist_ok=True)


def synthesize_gunshot(filename: str, duration=0.6, sr=16000):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    envelope = np.exp(-5 * t)
    tone = np.sin(2 * np.pi * 800 * t) * envelope
    noise = np.random.normal(0, 0.05, tone.shape)
    signal = tone + noise
    signal = signal / np.max(np.abs(signal))
    path = os.path.join(GUNSHOT_DIR, filename)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((signal * 32767).astype(np.int16).tobytes())
    print(f"Generated {path}")

def _find_first(match_name: str):
    for root, _, files in os.walk(DATA_DIR):
        if match_name in files:
            return os.path.join(root, match_name)
    return None


def _find_audio_dir():
    candidates = ["audio", "wav_files"]
    for root, dirs, _ in os.walk(DATA_DIR):
        for candidate in candidates:
            if candidate in dirs:
                return os.path.join(root, candidate)
    return None


def _normalize_esc50_layout():
    csv_path = os.path.join(ESC50_META, "esc50.csv")
    if os.path.exists(csv_path) and os.path.isdir(ESC50_AUDIO):
        return

    found_csv = _find_first("esc50.csv")
    found_audio = _find_audio_dir()
    if not found_csv or not found_audio:
        raise FileNotFoundError("ESC-50 files not found after download.")

    os.makedirs(ESC50_META, exist_ok=True)
    os.makedirs(ESC50_AUDIO, exist_ok=True)

    if not os.path.exists(csv_path):
        shutil.copy2(found_csv, csv_path)

    for item in os.listdir(found_audio):
        src = os.path.join(found_audio, item)
        dst = os.path.join(ESC50_AUDIO, item)
        if os.path.isfile(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)


def _download_with_kaggle_cli():
    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        ESC50_KAGGLE_DATASET,
        "-p",
        DATA_DIR,
        "--unzip",
        "--force",
    ]
    subprocess.run(cmd, check=True)


def _download_with_kaggle_api():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as exc:
        raise RuntimeError(
            "Kaggle API not installed. Install with 'pip install kaggle' and "
            "set your kaggle.json credentials (KAGGLE_USERNAME/KAGGLE_KEY)."
        ) from exc

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(ESC50_KAGGLE_DATASET, path=DATA_DIR, unzip=True)


def download_esc50_from_kaggle(max_attempts=3):
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            _download_with_kaggle_cli()
            _normalize_esc50_layout()
            return
        except Exception as exc:
            last_error = exc
            print(f"Kaggle CLI download failed (attempt {attempt}/{max_attempts}): {exc}")
            time.sleep(3 * attempt)

    for attempt in range(1, max_attempts + 1):
        try:
            _download_with_kaggle_api()
            _normalize_esc50_layout()
            return
        except Exception as exc:
            last_error = exc
            print(f"Kaggle API download failed (attempt {attempt}/{max_attempts}): {exc}")
            time.sleep(3 * attempt)

    raise RuntimeError(f"Failed to download ESC-50 from Kaggle: {last_error}")


def ensure_esc50_dataset():
    csv_path = os.path.join(ESC50_META, "esc50.csv")
    audio_dir = ESC50_AUDIO
    if os.path.exists(csv_path) and os.path.isdir(audio_dir):
        return
    download_esc50_from_kaggle()


if __name__ == "__main__":
    for idx in range(1, 6):
        synthesize_gunshot(f"gunshot_{idx}.wav")
    print("Gunshot sample generation complete.")
    print("Attempting to download ESC-50 from Kaggle...")
    ensure_esc50_dataset()
