#!/usr/bin/env python3
"""Evaluate heuristic/VLM/MLP per-second agreement on Films 1+2."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path[:0] = [str(SCRIPT_DIR), str(REPO_ROOT)]

from activity_mlp.film_agreement import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
