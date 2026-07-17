"""Public training command and immutable model metadata."""

from __future__ import annotations

import argparse
import hashlib
import platform
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np

from activity_mlp.data import build_feature_matrix
from activity_mlp.evaluation import write_json_once
from activity_mlp.release_dataset import load_development_rows
from activity_mlp.training import (
    TrainingConfig,
    TrainingOutcome,
    export_onnx,
    model_spec,
    print_epoch_heartbeat,
    train_model,
)
from pipeline.activity_dataset import validate_dataset
from pipeline.activity_features import ACTIVITY_CLASSES, feature_schema_manifest

MODEL_VERSION = "activity-mlp-v1.0.0"
MODEL_FILENAME = f"{MODEL_VERSION}.onnx"
MODEL_SIZE_LIMIT_BYTES = 10 * 1024 * 1024


def _sample_ids_sha256(rows: list[dict]) -> str:
    payload = "".join(f"{row['sample_id']}\n" for row in rows).encode()
    return hashlib.sha256(payload).hexdigest()


def build_training_metadata(
    *,
    labels_path: str | Path,
    model_path: str | Path,
    train_rows: list[dict],
    validation_rows: list[dict],
    config: TrainingConfig,
    outcome: TrainingOutcome,
    dependency_versions: dict[str, str],
) -> dict:
    """Bind weights to their exact inputs, code contract, and measured selection."""
    labels_bytes = Path(labels_path).read_bytes()
    model_bytes = Path(model_path).read_bytes()
    return {
        "schema_version": 1,
        "model": {
            "version": MODEL_VERSION,
            "sha256": hashlib.sha256(model_bytes).hexdigest(),
            "bytes": len(model_bytes),
            "class_order": list(ACTIVITY_CLASSES),
        },
        "feature_schema": feature_schema_manifest(),
        "dataset": {
            "labels_sha256": hashlib.sha256(labels_bytes).hexdigest(),
            "split_counts": {
                "train": len(train_rows),
                "validation": len(validation_rows),
            },
            "train_sample_ids_sha256": _sample_ids_sha256(train_rows),
            "validation_sample_ids_sha256": _sample_ids_sha256(validation_rows),
        },
        "architecture": model_spec(config),
        "training": {
            "config": asdict(config),
            "best_epoch": outcome.best_epoch,
            "epochs_ran": outcome.epochs_ran,
            "best_validation_accuracy": outcome.best_validation_accuracy,
            "best_validation_loss": outcome.best_validation_loss,
        },
        "dependencies": dependency_versions,
    }


def _dependency_versions() -> dict[str, str]:
    import onnx
    import onnxruntime as ort
    import torch

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": str(torch.version.cuda),
        "cudnn": str(torch.backends.cudnn.version()),
        "onnx": onnx.__version__,
        "onnxruntime": ort.__version__,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite frozen training output: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    print("validating all dataset assets before training", file=sys.stderr, flush=True)
    validate_dataset(args.dataset)
    labels_path = args.dataset / "labels.jsonl"
    train_rows, validation_rows = load_development_rows(labels_path)
    train_features, train_labels = build_feature_matrix(train_rows)
    validation_features, validation_labels = build_feature_matrix(validation_rows)
    config = TrainingConfig()
    outcome = train_model(
        train_features,
        train_labels,
        validation_features,
        validation_labels,
        config,
        progress=print_epoch_heartbeat,
    )

    model_path = args.output_dir / MODEL_FILENAME
    export_onnx(outcome.model, model_path)
    if model_path.stat().st_size > MODEL_SIZE_LIMIT_BYTES:
        raise RuntimeError(
            f"exported model is {model_path.stat().st_size} bytes; "
            f"limit is {MODEL_SIZE_LIMIT_BYTES} bytes"
        )
    metadata = build_training_metadata(
        labels_path=labels_path,
        model_path=model_path,
        train_rows=train_rows,
        validation_rows=validation_rows,
        config=config,
        outcome=outcome,
        dependency_versions=_dependency_versions(),
    )
    write_json_once(args.output_dir / "model-metadata.json", metadata)
    write_json_once(
        args.output_dir / "training-history.json",
        {"schema_version": 1, "epochs": outcome.history},
    )
    print(
        f"training complete best_epoch={outcome.best_epoch} "
        f"validation_accuracy={outcome.best_validation_accuracy:.6f} "
        f"model={model_path}",
        file=sys.stderr,
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
