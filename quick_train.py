import os

try:
    from .train_all import main
except ImportError:
    from train_all import main


if __name__ == "__main__":
    os.environ.setdefault("AUDIO_MODEL_EPOCHS", "4")
    os.environ.setdefault("PREDICTION_MODEL_EPOCHS", "6")
    os.environ.setdefault("AUDIO_MAX_PER_LABEL", "60")
    main()
