"""Behavior tests for issue #33 visual-review artifacts."""

from __future__ import annotations

import json

import pytest
from PIL import Image

from pipeline.activity_dataset_review import finalize_review, render_review_sheets


def test_review_sheet_indexes_annotated_full_frame(tmp_path) -> None:  # noqa: ANN001
    """A reviewer can map every rendered tile back to its label row."""
    dataset_dir = tmp_path / "dataset"
    output_dir = tmp_path / "review"
    frame_path = dataset_dir / "geometry-1" / "frames" / "sample-1.jpg"
    frame_path.parent.mkdir(parents=True)
    Image.new("RGB", (100, 50), "white").save(frame_path)
    sample = {
        "sample_id": "sample-1",
        "frame_path": "geometry-1/frames/sample-1.jpg",
        "bbox": [10.0, 5.0, 40.0, 35.0],
        "keypoints": [{"x": 20.0, "y": 20.0, "vis": 0.9} for _ in range(17)],
        "activity": "standing",
        "split": "train",
        "camera_geometry_id": "geometry-1",
        "pose_confidence": 0.91,
        "source_timestamp_s": 1.25,
        "source_video_sha256": "a" * 64,
    }
    (dataset_dir / "labels.jsonl").write_text(json.dumps(sample) + "\n", encoding="utf-8")

    summary = render_review_sheets(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        columns=2,
        rows=2,
        thumbnail_width=160,
    )

    assert summary == {"samples": 1, "sheets": 1}
    assert (output_dir / "review-0001.jpg").is_file()
    index = json.loads((output_dir / "review-index.json").read_text(encoding="utf-8"))
    assert index["review-0001.jpg"] == ["sample-1"]


def test_finalize_review_promotes_only_a_complete_index(tmp_path) -> None:  # noqa: ANN001
    """Every label must appear once in the checked sheets before promotion."""
    dataset_dir = tmp_path / "dataset"
    review_dir = tmp_path / "review"
    dataset_dir.mkdir()
    review_dir.mkdir()
    samples = [
        {"sample_id": "sample-1", "review_status": "pending"},
        {"sample_id": "sample-2", "review_status": "pending"},
    ]
    (dataset_dir / "labels.jsonl").write_text(
        "".join(json.dumps(sample) + "\n" for sample in samples), encoding="utf-8"
    )
    decisions_path = dataset_dir / "review-decisions.json"
    decisions_path.write_text('{"schema_version": 1}\n', encoding="utf-8")
    sheet_path = review_dir / "review-0001.jpg"
    Image.new("RGB", (16, 16), "white").save(sheet_path)
    (review_dir / "review-index.json").write_text(
        json.dumps({"review-0001.jpg": ["sample-1"]}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="review index does not cover labels"):
        finalize_review(
            dataset_dir=dataset_dir,
            review_dir=review_dir,
            review_decisions_path=decisions_path,
            reviewer="test reviewer",
        )

    (review_dir / "review-index.json").write_text(
        json.dumps({"review-0001.jpg": ["sample-1", "sample-2"]}), encoding="utf-8"
    )
    record = finalize_review(
        dataset_dir=dataset_dir,
        review_dir=review_dir,
        review_decisions_path=decisions_path,
        reviewer="test reviewer",
    )

    finalized = [
        json.loads(line)
        for line in (dataset_dir / "labels.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert all(sample["review_status"] == "reviewed" for sample in finalized)
    assert record["sample_count"] == 2
    assert record["sheet_count"] == 1
    assert record["reviewer"] == "test reviewer"
    assert (dataset_dir / "review-record.json").is_file()
