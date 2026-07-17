"""Frozen-test evaluation artifact and public command contracts."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from activity_mlp.evaluation_cli import build_frozen_evaluation_artifact


def test_frozen_evaluation_binds_model_dataset_baseline_and_quality_gate(tmp_path) -> None:
    labels_path = tmp_path / "labels.jsonl"
    labels_path.write_bytes(b"held-out labels\n")
    baselines_path = tmp_path / "baselines.json"
    baseline_artifact = {
        "dataset": {"labels_sha256": hashlib.sha256(b"held-out labels\n").hexdigest()},
        "baselines": {
            "vlm": {
                "accuracy": 0.9,
                "geometries": {"garden": {"accuracy": 0.8}},
                "predictions": [],
            }
        },
    }
    baselines_path.write_text(json.dumps(baseline_artifact))
    model_metadata = {
        "model": {"version": "activity-mlp-v1.0.0", "sha256": "model-sha"},
        "feature_schema": {"schema_version": "activity-mlp-features-v1"},
    }
    mlp_report = {
        "accuracy": 0.91,
        "geometries": {"garden": {"accuracy": 0.81}},
        "predictions": [],
    }

    artifact = build_frozen_evaluation_artifact(
        labels_path=labels_path,
        baselines_path=baselines_path,
        model_metadata=model_metadata,
        mlp_report=mlp_report,
    )

    assert artifact["model"] == model_metadata["model"]
    assert artifact["dataset"]["labels_sha256"] == baseline_artifact["dataset"]["labels_sha256"]
    assert (
        artifact["baseline_artifact_sha256"]
        == hashlib.sha256(baselines_path.read_bytes()).hexdigest()
    )
    assert artifact["quality_gate"]["passed"] is True


def test_module_exposes_the_one_shot_frozen_evaluation_command() -> None:
    repo_root = Path(__file__).parents[3]
    completed = subprocess.run(
        [sys.executable, "-m", "activity_mlp.evaluation_cli", "--help"],
        cwd=repo_root,
        env={**os.environ, "PYTHONPATH": "training/activity-mlp"},
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "--model" in completed.stdout
    assert "--baselines" in completed.stdout
    assert "--output" in completed.stdout


def test_required_eval_py_wrapper_is_directly_runnable() -> None:
    repo_root = Path(__file__).parents[3]
    completed = subprocess.run(
        [sys.executable, "training/activity-mlp/eval.py", "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "--model" in completed.stdout
    assert "--output" in completed.stdout
