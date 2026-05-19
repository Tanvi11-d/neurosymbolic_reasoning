#!/usr/bin/env python3
"""Run GET-ORDER CLI without installing the package (prepends src/). From repo root::

    export OPENROUTER_API_KEY=...
    python scripts/verify_order.py parallel
    python scripts/verify_order.py benchmark --jobs 8
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

from nre.cli.verify_order import main  # noqa: E402

if __name__ == "__main__":
    main()
