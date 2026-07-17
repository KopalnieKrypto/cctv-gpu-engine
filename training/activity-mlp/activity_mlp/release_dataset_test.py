"""Behavior tests for train/validation isolation from the frozen test set."""

from __future__ import annotations

import json

import pytest

from activity_mlp.release_dataset import load_development_rows


def test_training_loader_returns_only_the_fixed_700_and_150_development_rows(tmp_path) -> None:
    labels_path = tmp_path / "labels.jsonl"
    rows = [
        *[{"sample_id": f"train-{index:03d}", "split": "train"} for index in range(700)],
        *[{"sample_id": f"validation-{index:03d}", "split": "validation"} for index in range(150)],
        *[
            {"sample_id": f"test-{index:03d}", "split": "test", "activity": "secret"}
            for index in range(150)
        ],
    ]
    labels_path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    train_rows, validation_rows = load_development_rows(labels_path)

    assert len(train_rows) == 700
    assert len(validation_rows) == 150
    assert all(row["split"] == "train" for row in train_rows)
    assert all(row["split"] == "validation" for row in validation_rows)
    assert not any("secret" in row.values() for row in (*train_rows, *validation_rows))


def test_training_loader_rejects_any_change_to_development_split_counts(tmp_path) -> None:
    labels_path = tmp_path / "labels.jsonl"
    rows = [
        *({"sample_id": f"train-{index}", "split": "train"} for index in range(699)),
        *({"sample_id": f"validation-{index}", "split": "validation"} for index in range(150)),
    ]
    labels_path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    with pytest.raises(
        ValueError,
        match="expected development split counts train=700, validation=150; found train=699",
    ):
        load_development_rows(labels_path)


def test_training_loader_rejects_source_frame_leakage_between_splits(tmp_path) -> None:
    labels_path = tmp_path / "labels.jsonl"
    train_rows = [
        {
            "sample_id": f"train-{index}",
            "split": "train",
            "frame_sha256": f"train-sha-{index}",
        }
        for index in range(700)
    ]
    validation_rows = [
        {
            "sample_id": f"validation-{index}",
            "split": "validation",
            "frame_sha256": f"validation-sha-{index}",
        }
        for index in range(150)
    ]
    validation_rows[0]["frame_sha256"] = train_rows[0]["frame_sha256"]
    labels_path.write_text(
        "".join(json.dumps(row) + "\n" for row in (*train_rows, *validation_rows))
    )

    with pytest.raises(ValueError, match="source frame train-sha-0 crosses train/validation"):
        load_development_rows(labels_path)
