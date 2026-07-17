"""Access to the frozen issue #33 release splits."""

from __future__ import annotations

import json
from pathlib import Path

EXPECTED_TEST_ROWS = 150


def load_test_rows(labels_path: str | Path) -> list[dict]:
    """Load the fixed held-out rows without reordering them."""
    with Path(labels_path).open(encoding="utf-8") as labels_file:
        rows = [json.loads(line) for line in labels_file if line.strip()]
    test_rows = [row for row in rows if row["split"] == "test"]
    if len(test_rows) != EXPECTED_TEST_ROWS:
        raise ValueError(
            f"expected exactly {EXPECTED_TEST_ROWS} held-out rows; found {len(test_rows)}"
        )
    return test_rows
