"""Reproducible warm batch-1 latency gate for the frozen activity MLP."""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Protocol

from pipeline.postprocessing import Detection, Keypoint

P95_LIMIT_MS = 5.0


class ClassifierLike(Protocol):
    model_version: str
    model_sha256: str

    def classify(self, detection: Detection) -> str: ...


def benchmark_latency(
    *,
    classifier: ClassifierLike,
    detection: Detection,
    warmup_count: int,
    measured_count: int,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
    heartbeat_every: int = 100,
    heartbeat_output: IO[str] = sys.stderr,
) -> dict:
    """Measure synchronous feature extraction + ONNX batch-1 inference."""
    if warmup_count < 0 or measured_count <= 0:
        raise ValueError("benchmark counts require warmup >= 0 and measured > 0")
    if heartbeat_every <= 0:
        raise ValueError("heartbeat interval must be positive")

    for index in range(warmup_count):
        classifier.classify(detection)
        if (index + 1) % heartbeat_every == 0 or index + 1 == warmup_count:
            print(
                f"latency heartbeat warmup={index + 1}/{warmup_count}",
                file=heartbeat_output,
                flush=True,
            )

    timings_ms: list[float] = []
    for index in range(measured_count):
        started_ns = clock_ns()
        classifier.classify(detection)
        elapsed_ns = clock_ns() - started_ns
        timings_ms.append(elapsed_ns / 1_000_000)
        if (index + 1) % heartbeat_every == 0 or index + 1 == measured_count:
            print(
                f"latency heartbeat measured={index + 1}/{measured_count}",
                file=heartbeat_output,
                flush=True,
            )

    ordered = sorted(timings_ms)
    p95_higher = ordered[math.ceil(0.95 * len(ordered)) - 1]
    summary = {
        "min": min(timings_ms),
        "mean": statistics.fmean(timings_ms),
        "median": statistics.median(timings_ms),
        "p95_higher": p95_higher,
        "max": max(timings_ms),
    }
    return {
        "warmup_count": warmup_count,
        "measured_count": measured_count,
        "timings_ms": timings_ms,
        "summary_ms": summary,
        "gate": {
            "p95_limit_ms": P95_LIMIT_MS,
            "passed": p95_higher <= P95_LIMIT_MS,
        },
    }


def _representative_detection() -> Detection:
    """Deterministic valid COCO pose; values do not affect network compute."""
    return Detection(
        bbox=[100.0, 50.0, 300.0, 450.0],
        confidence=0.95,
        keypoints=[
            Keypoint(
                x=130.0 + (index % 5) * 30.0,
                y=80.0 + index * 18.0,
                vis=0.95 - (index % 3) * 0.05,
            )
            for index in range(17)
        ],
    )


def _gpu_inventory() -> list[dict[str, str]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=True, timeout=30)
    rows = []
    for line in completed.stdout.strip().splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 5:
            rows.append(
                dict(
                    zip(
                        ("index", "uuid", "name", "driver", "memory_total_mb"),
                        fields,
                        strict=True,
                    )
                )
            )
    return rows


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/activity-mlp-v1.0.0.onnx")
    parser.add_argument("--metadata", default="models/activity-mlp-v1.0.0.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--heartbeat-every", type=int, default=100)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite immutable benchmark: {args.output}")

    import numpy as np
    import onnxruntime as ort

    from pipeline.mlp_classifier import load_activity_mlp

    classifier = load_activity_mlp(args.model, args.metadata)
    measurements = benchmark_latency(
        classifier=classifier,
        detection=_representative_detection(),
        warmup_count=args.warmup,
        measured_count=args.samples,
        heartbeat_every=args.heartbeat_every,
    )
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    artifact = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "measurement": "feature extraction plus synchronous ONNX batch-1 inference per detection",
        "model": {
            "version": classifier.model_version,
            "sha256": classifier.model_sha256,
            "feature_schema_version": metadata["feature_schema"]["schema_version"],
            "class_order": metadata["model"]["class_order"],
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "onnxruntime": ort.__version__,
            "available_providers": ort.get_available_providers(),
            "active_providers": classifier.session.get_providers(),
        },
        "hardware": {
            "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
            "gpus": _gpu_inventory(),
        },
        **measurements,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"latency gate p95={measurements['summary_ms']['p95_higher']:.6f} ms "
        f"limit={P95_LIMIT_MS:.1f} ms passed={measurements['gate']['passed']}",
        flush=True,
    )
    return 0 if measurements["gate"]["passed"] else 2
