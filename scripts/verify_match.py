#!/usr/bin/env python3
"""Wrapper: run nre.cli.verify_match with repo src on sys.path."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from nre.cli.verify_match import main  # noqa: E402

if __name__ == "__main__":
    main()
