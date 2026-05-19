#!/usr/bin/env python3
"""Run LLM DECOMPOSE verification without installing the package.

Prepends the repo's ``src/`` directory so ``import nre`` works. Invoke from repo root::

    python scripts/verify_decompose.py turn1
    python scripts/verify_decompose.py benchmark   # OpenRouter; parallel; needs OPENROUTER_API_KEY
"""

from __future__ import annotations

import sys
from pathlib import Path


def _prepend_src() -> None:
    root = Path(__file__).resolve().parent.parent
    src = root / "src"
    if not (src / "nre").is_dir():
        sys.stderr.write(f"Expected package at {src / 'nre'}, run from repo root.\n")
        sys.exit(2)
    sys.path.insert(0, str(src))


_prepend_src()

from nre.cli.verify_decompose import main  # noqa: E402

if __name__ == "__main__":
    main()
