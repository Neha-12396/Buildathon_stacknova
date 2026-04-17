from __future__ import annotations

try:
    from .train_all import main
except ImportError:
    from train_all import main


if __name__ == "__main__":
    main()
