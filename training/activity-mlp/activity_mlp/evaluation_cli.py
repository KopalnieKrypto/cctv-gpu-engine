"""One-shot held-out evaluation of the frozen activity MLP."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from activity_mlp.baseline_cli import HeartbeatPredictor
from activity_mlp.data import row_to_detection
from activity_mlp.evaluation import (
    build_quality_gate,
    evaluate_rows_with_predictions,
    write_json_once,
)
from activity_mlp.release_dataset import load_test_rows
from pipeline.activity_dataset import validate_dataset
from pipeline.mlp_classifier import load_activity_mlp


def build_frozen_evaluation_artifact(
    *,
    labels_path: str | Path,
    baselines_path: str | Path,
    model_metadata: dict,
    mlp_report: dict,
) -> dict:
    """Bind the one-shot result to frozen weights, labels, and VLM baseline."""
    labels_bytes = Path(labels_path).read_bytes()
    baselines_bytes = Path(baselines_path).read_bytes()
    baseline_artifact = json.loads(baselines_bytes)
    labels_sha256 = hashlib.sha256(labels_bytes).hexdigest()
    if labels_sha256 != baseline_artifact["dataset"]["labels_sha256"]:
        raise RuntimeError("held-out labels checksum differs from immutable baseline dataset")
    vlm_report = baseline_artifact["baselines"]["vlm"]
    return {
        "schema_version": 1,
        "model": model_metadata["model"],
        "feature_schema": model_metadata["feature_schema"],
        "dataset": baseline_artifact["dataset"],
        "baseline_artifact_sha256": hashlib.sha256(baselines_bytes).hexdigest(),
        "baselines": baseline_artifact["baselines"],
        "mlp": mlp_report,
        "quality_gate": build_quality_gate(mlp_report, vlm_report),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--baselines", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    print("validating all held-out assets before frozen evaluation", file=sys.stderr, flush=True)
    validate_dataset(args.dataset)
    test_rows = load_test_rows(args.dataset / "labels.jsonl")
    classifier = load_activity_mlp(args.model, args.metadata)
    predictor = HeartbeatPredictor(
        "mlp",
        lambda row: classifier.classify(row_to_detection(row)),
    )
    mlp_report = evaluate_rows_with_predictions(test_rows, predictor)
    model_metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    artifact = build_frozen_evaluation_artifact(
        labels_path=args.dataset / "labels.jsonl",
        baselines_path=args.baselines,
        model_metadata=model_metadata,
        mlp_report=mlp_report,
    )
    write_json_once(args.output, artifact)
    print(
        f"frozen evaluation accuracy={mlp_report['accuracy']:.6f} "
        f"quality_gate_passed={artifact['quality_gate']['passed']}",
        file=sys.stderr,
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
