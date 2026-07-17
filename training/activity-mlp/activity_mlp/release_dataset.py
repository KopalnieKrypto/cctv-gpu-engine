"""Access to the frozen issue #33 release splits."""

from __future__ import annotations

import json
from pathlib import Path

EXPECTED_TEST_ROWS = 150
EXPECTED_TRAIN_ROWS = 700
EXPECTED_VALIDATION_ROWS = 150


def _load_rows(labels_path: str | Path) -> list[dict]:
    with Path(labels_path).open(encoding="utf-8") as labels_file:
        return [json.loads(line) for line in labels_file if line.strip()]


def load_test_rows(labels_path: str | Path) -> list[dict]:
    """Load the fixed held-out rows without reordering them."""
    rows = _load_rows(labels_path)
    test_rows = [row for row in rows if row["split"] == "test"]
    if len(test_rows) != EXPECTED_TEST_ROWS:
        raise ValueError(
            f"expected exactly {EXPECTED_TEST_ROWS} held-out rows; found {len(test_rows)}"
        )
    return test_rows


def load_development_rows(labels_path: str | Path) -> tuple[list[dict], list[dict]]:
    """Return only the frozen train and validation rows, preserving order."""
    rows = _load_rows(labels_path)
    frame_splits: dict[str, set[str]] = {}
    for row in rows:
        if "frame_sha256" in row:
            frame_splits.setdefault(row["frame_sha256"], set()).add(row["split"])
    for frame_sha256, splits in frame_splits.items():
        if len(splits) > 1:
            raise ValueError(f"source frame {frame_sha256} crosses {'/'.join(sorted(splits))}")
    train_rows = [row for row in rows if row["split"] == "train"]
    validation_rows = [row for row in rows if row["split"] == "validation"]
    if len(train_rows) != EXPECTED_TRAIN_ROWS or len(validation_rows) != EXPECTED_VALIDATION_ROWS:
        raise ValueError(
            "expected development split counts train=700, validation=150; "
            f"found train={len(train_rows)}, validation={len(validation_rows)}"
        )
    return train_rows, validation_rows
