"""Evidence contract emitted beside the frozen ONNX model."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

from activity_mlp.training import TrainingConfig, TrainingOutcome
from activity_mlp.training_cli import MODEL_VERSION, build_training_metadata
from pipeline.activity_features import ACTIVITY_CLASSES, feature_schema_manifest


def test_training_metadata_binds_data_schema_weights_versions_and_validation(tmp_path) -> None:
    labels_path = tmp_path / "labels.jsonl"
    labels_path.write_bytes(b"frozen labels\n")
    model_path = tmp_path / "activity-mlp.onnx"
    model_path.write_bytes(b"frozen model")
    train_rows = [{"sample_id": f"train-{index}"} for index in range(700)]
    validation_rows = [{"sample_id": f"validation-{index}"} for index in range(150)]
    outcome = TrainingOutcome(
        model=None,
        best_epoch=17,
        best_validation_accuracy=0.91,
        best_validation_loss=0.22,
        epochs_ran=67,
        history=[],
    )

    metadata = build_training_metadata(
        labels_path=labels_path,
        model_path=model_path,
        train_rows=train_rows,
        validation_rows=validation_rows,
        config=TrainingConfig(),
        outcome=outcome,
        dependency_versions={"python": "3.12.3", "torch": "2.11.0+cu128"},
    )

    assert metadata["model"] == {
        "version": MODEL_VERSION,
        "sha256": hashlib.sha256(b"frozen model").hexdigest(),
        "bytes": len(b"frozen model"),
        "class_order": list(ACTIVITY_CLASSES),
    }
    assert metadata["feature_schema"] == feature_schema_manifest()
    assert metadata["dataset"]["labels_sha256"] == hashlib.sha256(b"frozen labels\n").hexdigest()
    assert metadata["dataset"]["split_counts"] == {"train": 700, "validation": 150}
    assert metadata["training"]["config"]["seed"] == 3407
    assert metadata["training"]["best_epoch"] == 17
    assert metadata["training"]["best_validation_accuracy"] == 0.91
    assert metadata["dependencies"]["torch"] == "2.11.0+cu128"


def test_module_exposes_the_reproducible_training_command() -> None:
    repo_root = Path(__file__).parents[3]
    completed = subprocess.run(
        [sys.executable, "-m", "activity_mlp.training_cli", "--help"],
        cwd=repo_root,
        env={**os.environ, "PYTHONPATH": "training/activity-mlp"},
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "--dataset" in completed.stdout
    assert "--output-dir" in completed.stdout


def test_required_train_py_wrapper_is_directly_runnable() -> None:
    repo_root = Path(__file__).parents[3]
    completed = subprocess.run(
        [sys.executable, "training/activity-mlp/train.py", "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "--output-dir" in completed.stdout
