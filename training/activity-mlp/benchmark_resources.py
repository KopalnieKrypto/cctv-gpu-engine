#!/usr/bin/env python3
"""Compare end-to-end resource use for all activity classifier arms."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path[:0] = [str(SCRIPT_DIR), str(REPO_ROOT)]

from activity_mlp.resource_benchmark import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
