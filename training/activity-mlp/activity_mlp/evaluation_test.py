"""Behavior tests for immutable activity-classifier evaluation.

Assumptions approved before the first RED for issue #34:

* Input rows follow the issue #33 label schema: an ``activity`` in the frozen
  four-class order and a ``camera_geometry_id``; production predictors receive
  the full row so heuristic and VLM adapters can use the same evaluator.
* Output is deterministic JSON-compatible data containing overall accuracy,
  per-class precision/recall/F1/support, a true-row/predicted-column confusion
  matrix, and the same metrics per held-out geometry.
* Empty input and unsupported labels are errors rather than silently producing
  misleading zero-sample metrics.
* Actual CUDA/VLM inference is intentionally outside this unit test. The
  predictor is an injected system boundary; the released model is exercised by
  the reproducible GPU benchmark gate before promotion.
"""

import json
from hashlib import sha256

import numpy as np
import pytest
from PIL import Image

from activity_mlp.evaluation import (
    CLASS_ORDER,
    VlmRowPredictor,
    build_baseline_artifact,
    evaluate_rows,
    predict_heuristic,
    write_json_once,
)
from activity_mlp.release_dataset import load_test_rows


def test_evaluate_rows_reports_overall_class_and_geometry_metrics() -> None:
    rows = [
        {"sample_id": "garden-sit", "activity": "sitting", "camera_geometry_id": "garden"},
        {
            "sample_id": "garden-stand",
            "activity": "standing",
            "camera_geometry_id": "garden",
        },
        {"sample_id": "track-walk", "activity": "walking", "camera_geometry_id": "track"},
        {"sample_id": "track-run", "activity": "running", "camera_geometry_id": "track"},
    ]
    predictions = {
        "garden-sit": "sitting",
        "garden-stand": "sitting",
        "track-walk": "walking",
        "track-run": "running",
    }

    report = evaluate_rows(rows, lambda row: predictions[row["sample_id"]])

    assert report["class_order"] == list(CLASS_ORDER)
    assert report["sample_count"] == 4
    assert report["accuracy"] == 0.75
    assert report["confusion_matrix"] == [
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
    assert report["per_class"]["sitting"] == {
        "precision": 0.5,
        "recall": 1.0,
        "f1": 2 / 3,
        "support": 1,
    }
    assert report["geometries"]["garden"]["accuracy"] == 0.5
    assert report["geometries"]["track"]["accuracy"] == 1.0


def test_load_test_rows_preserves_the_frozen_150_row_split(tmp_path) -> None:
    labels_path = tmp_path / "labels.jsonl"
    rows = [
        {"sample_id": "train-only", "split": "train"},
        *[{"sample_id": f"test-{index:03d}", "split": "test"} for index in range(150)],
        {"sample_id": "validation-only", "split": "validation"},
    ]
    labels_path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    test_rows = load_test_rows(labels_path)

    assert [row["sample_id"] for row in test_rows] == [f"test-{index:03d}" for index in range(150)]


def test_load_test_rows_rejects_a_changed_test_split(tmp_path) -> None:
    labels_path = tmp_path / "labels.jsonl"
    labels_path.write_text(
        "".join(
            json.dumps({"sample_id": f"test-{index:03d}", "split": "test"}) + "\n"
            for index in range(149)
        )
    )

    with pytest.raises(ValueError, match="expected exactly 150 held-out rows; found 149"):
        load_test_rows(labels_path)


def test_heuristic_baseline_converts_dataset_xywh_bbox_to_runtime_xyxy() -> None:
    row = {
        "bbox": [100.0, 200.0, 20.0, 80.0],
        "keypoints": [{"x": 0.0, "y": 0.0, "vis": 0.0} for _ in range(17)],
    }

    assert predict_heuristic(row) == "standing"


def test_vlm_baseline_reads_each_rows_full_source_frame(tmp_path) -> None:
    frame_path = tmp_path / "garden" / "frames" / "sample.jpg"
    frame_path.parent.mkdir(parents=True)
    Image.fromarray(np.full((3, 4, 3), [255, 0, 0], dtype=np.uint8)).save(frame_path)

    class FakeVlm:
        def classify_frame(self, frame_bgr: np.ndarray) -> str:
            assert frame_bgr.shape == (3, 4, 3)
            assert frame_bgr[0, 0, 2] > 200  # OpenCV BGR, source image was RGB red.
            return "standing"

    predictor = VlmRowPredictor(tmp_path, FakeVlm())

    assert predictor({"frame_path": "garden/frames/sample.jpg"}) == "standing"


def test_baseline_artifact_pins_dataset_and_each_sample_prediction(tmp_path) -> None:
    labels_path = tmp_path / "labels.jsonl"
    activities = list(CLASS_ORDER)
    rows = [
        {
            "sample_id": f"test-{index:03d}",
            "split": "test",
            "activity": activities[index % len(activities)],
            "camera_geometry_id": "controlled-garden" if index < 100 else "pexels-marathon",
        }
        for index in range(150)
    ]
    labels_bytes = "".join(json.dumps(row) + "\n" for row in rows).encode()
    labels_path.write_bytes(labels_bytes)

    artifact = build_baseline_artifact(
        labels_path,
        {
            "heuristic": lambda row: row["activity"],
            "vlm": lambda _row: "standing",
        },
    )

    assert artifact["schema_version"] == 1
    assert artifact["dataset"]["labels_sha256"] == sha256(labels_bytes).hexdigest()
    assert artifact["dataset"]["test_sample_count"] == 150
    assert len(artifact["baselines"]["vlm"]["predictions"]) == 150
    assert artifact["baselines"]["heuristic"]["accuracy"] == 1.0
    assert artifact["baselines"]["vlm"]["predictions"][0] == {
        "sample_id": "test-000",
        "camera_geometry_id": "controlled-garden",
        "actual": "sitting",
        "predicted": "standing",
    }


def test_immutable_baseline_writer_refuses_to_replace_existing_evidence(tmp_path) -> None:
    output_path = tmp_path / "baselines.json"
    write_json_once(output_path, {"schema_version": 1})

    with pytest.raises(FileExistsError, match="refusing to overwrite immutable evidence"):
        write_json_once(output_path, {"schema_version": 2})

    assert json.loads(output_path.read_text()) == {"schema_version": 1}
