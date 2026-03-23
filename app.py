#!/usr/bin/env python3
"""
Assignment-style entrypoint: start the Telegram bot.

Usage (from repo root, with venv activated):
  pip install -e .
  python app.py

Equivalent: health-rag-bot
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow `python app.py` without editable install (adds src/ to path).
_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from health_rag.telegram_app.bot import main  # noqa: E402

if __name__ == "__main__":
    main()
