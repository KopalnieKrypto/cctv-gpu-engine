"""Behavior tests for issue #33 visual-review artifacts."""

from __future__ import annotations

import json

from PIL import Image

from pipeline.activity_dataset_review import render_review_sheets


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
