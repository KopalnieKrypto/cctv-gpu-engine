"""Deterministic metrics shared by baseline and frozen-model evaluation."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from hashlib import sha256
from pathlib import Path

import cv2

from activity_mlp.release_dataset import load_test_rows
from pipeline.activity_classifier import classify_activity
from pipeline.postprocessing import Detection, Keypoint

CLASS_ORDER = ("sitting", "standing", "walking", "running")


def _metrics(truth: list[str], predicted: list[str]) -> dict:
    class_index = {label: index for index, label in enumerate(CLASS_ORDER)}
    matrix = [[0 for _ in CLASS_ORDER] for _ in CLASS_ORDER]
    for actual, prediction in zip(truth, predicted, strict=True):
        matrix[class_index[actual]][class_index[prediction]] += 1

    per_class: dict[str, dict[str, float | int]] = {}
    for index, label in enumerate(CLASS_ORDER):
        true_positive = matrix[index][index]
        support = sum(matrix[index])
        predicted_count = sum(row[index] for row in matrix)
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    return {
        "sample_count": len(truth),
        "accuracy": sum(
            actual == prediction for actual, prediction in zip(truth, predicted, strict=True)
        )
        / len(truth),
        "per_class": per_class,
        "confusion_matrix": matrix,
    }


def _report_from_predictions(rows: Sequence[dict], predicted: list[str]) -> dict:
    truth = [row["activity"] for row in rows]
    report = {"class_order": list(CLASS_ORDER), **_metrics(truth, predicted)}

    geometries: dict[str, dict] = {}
    for geometry_id in sorted({row["camera_geometry_id"] for row in rows}):
        indices = [
            index for index, row in enumerate(rows) if row["camera_geometry_id"] == geometry_id
        ]
        geometries[geometry_id] = _metrics(
            [truth[index] for index in indices],
            [predicted[index] for index in indices],
        )
    report["geometries"] = geometries
    return report


def evaluate_rows(rows: Sequence[dict], predictor: Callable[[dict], str]) -> dict:
    """Evaluate ``predictor`` once for each ordered dataset row."""
    return _report_from_predictions(rows, [predictor(row) for row in rows])


def evaluate_rows_with_predictions(rows: Sequence[dict], predictor: Callable[[dict], str]) -> dict:
    """Evaluate rows and retain the audit record used by promotion gates."""
    predictions = [predictor(row) for row in rows]
    return {
        **_report_from_predictions(rows, predictions),
        "predictions": [
            {
                "sample_id": row["sample_id"],
                "camera_geometry_id": row["camera_geometry_id"],
                "actual": row["activity"],
                "predicted": prediction,
            }
            for row, prediction in zip(rows, predictions, strict=True)
        ],
    }


def predict_heuristic(row: dict) -> str:
    """Run the deployed geometric classifier on one issue #33 label row."""
    x, y, width, height = row["bbox"]
    detection = Detection(
        bbox=[x, y, x + width, y + height],
        confidence=1.0,
        keypoints=[Keypoint(**keypoint) for keypoint in row["keypoints"]],
    )
    return classify_activity(detection)


class VlmRowPredictor:
    """Adapt the deployed full-frame VLM API to issue #33 label rows."""

    def __init__(self, dataset_root: str | Path, classifier: object) -> None:
        self._dataset_root = Path(dataset_root)
        self._classifier = classifier

    def __call__(self, row: dict) -> str:
        frame_path = self._dataset_root / row["frame_path"]
        frame_bgr = cv2.imread(str(frame_path))
        if frame_bgr is None:
            raise ValueError(f"could not read dataset frame: {frame_path}")
        return self._classifier.classify_frame(frame_bgr)


def build_baseline_artifact(
    labels_path: str | Path,
    predictors: dict[str, Callable[[dict], str]],
) -> dict:
    """Evaluate named pre-training baselines against the frozen held-out rows."""
    path = Path(labels_path)
    labels_bytes = path.read_bytes()
    rows = load_test_rows(path)
    sample_ids_bytes = "".join(f"{row['sample_id']}\n" for row in rows).encode()
    baselines: dict[str, dict] = {}
    for name, predictor in predictors.items():
        baselines[name] = evaluate_rows_with_predictions(rows, predictor)
    return {
        "schema_version": 1,
        "dataset": {
            "labels_sha256": sha256(labels_bytes).hexdigest(),
            "test_sample_ids_sha256": sha256(sample_ids_bytes).hexdigest(),
            "test_sample_count": len(rows),
        },
        "class_order": list(CLASS_ORDER),
        "baselines": baselines,
    }


def write_json_once(output_path: str | Path, payload: dict) -> None:
    """Write deterministic evidence without allowing accidental replacement."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as output_file:
            json.dump(payload, output_file, indent=2, sort_keys=True)
            output_file.write("\n")
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite immutable evidence: {path}") from exc


def build_quality_gate(mlp_report: dict, vlm_report: dict) -> dict:
    """Apply the frozen held-out promotion thresholds without post-hoc tuning."""
    checks = {
        "accuracy_at_least_0_85": mlp_report["accuracy"] >= 0.85,
        "accuracy_at_least_vlm": mlp_report["accuracy"] >= vlm_report["accuracy"],
    }
    for geometry_id, vlm_geometry in vlm_report["geometries"].items():
        checks[f"{geometry_id}_at_least_vlm"] = (
            mlp_report["geometries"][geometry_id]["accuracy"] >= vlm_geometry["accuracy"]
        )

    vlm_predictions = {row["sample_id"]: row for row in vlm_report.get("predictions", [])}
    regressions = [
        row["sample_id"]
        for row in mlp_report.get("predictions", [])
        if row["predicted"] != row["actual"]
        and vlm_predictions.get(row["sample_id"], {}).get("predicted") == row["actual"]
    ]
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "mlp_regressions_vs_vlm": regressions,
    }
