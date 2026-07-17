"""Same-image end-to-end wallclock and VRAM benchmark for issue #34."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import IO

CLASSIFIERS = ("heuristic", "vlm", "mlp")


def build_docker_command(
    *,
    docker: str,
    image: str,
    video: Path,
    output_dir: Path,
    classifier: str,
    gpu_device: str,
    hf_cache_volume: str,
) -> list[str]:
    """Build one tracked arm command with baked source and model assets."""
    if classifier not in CLASSIFIERS:
        raise ValueError(f"unknown classifier: {classifier}")
    return [
        docker,
        "run",
        "--rm",
        "--gpus",
        f"device={gpu_device}",
        "--pid=host",
        "-v",
        f"{video.resolve()}:/input/video.mp4:ro",
        "-v",
        f"{output_dir.resolve()}:/output",
        "-v",
        f"{hf_cache_volume}:/root/.cache/huggingface",
        "--entrypoint",
        "python",
        image,
        "-m",
        "pipeline.analyze",
        "/input/video.mp4",
        "--output",
        "/output/result.json",
        "--dump-detections",
        "/output/detections.jsonl",
        "--model",
        "/app/models/yolo11s-pose.onnx",
        "--classifier",
        classifier,
        "--activity-model",
        "/app/models/activity-mlp-v1.0.0.onnx",
        "--activity-model-metadata",
        "/app/models/activity-mlp-v1.0.0.json",
    ]


def resource_promotion_gate(arms: dict[str, dict]) -> dict[str, bool]:
    """The MLP must strictly beat VLM on wallclock and peak VRAM."""
    faster = arms["mlp"]["wallclock_s"] < arms["vlm"]["wallclock_s"]
    lower_vram = arms["mlp"]["peak_vram_mb"] < arms["vlm"]["peak_vram_mb"]
    return {
        "mlp_faster_than_vlm": faster,
        "mlp_lower_peak_vram_than_vlm": lower_vram,
        "passed": faster and lower_vram,
    }


def _run_text(command: list[str]) -> str:
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    ).stdout.strip()


def _gpu_snapshot(gpu_device: str) -> dict[str, int | str]:
    raw = _run_text(
        [
            "nvidia-smi",
            f"--id={gpu_device}",
            "--query-gpu=index,uuid,name,driver_version,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    fields = [field.strip() for field in raw.splitlines()[0].split(",")]
    if len(fields) != 6:
        raise RuntimeError(f"unexpected nvidia-smi row: {raw!r}")
    return {
        "index": fields[0],
        "uuid": fields[1],
        "name": fields[2],
        "driver": fields[3],
        "memory_used_mb": int(fields[4]),
        "memory_total_mb": int(fields[5]),
    }


def _video_metadata(video: Path) -> dict[str, int | float | str]:
    duration = float(
        _run_text(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video),
            ]
        )
    )
    return {
        "path": str(video.resolve()),
        "sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
        "bytes": video.stat().st_size,
        "duration_s": duration,
    }


def _run_arm(
    *,
    command: list[str],
    classifier: str,
    output_dir: Path,
    gpu_device: str,
    sample_interval_s: float,
    heartbeat_interval_s: float,
    heartbeat_output: IO[str] = sys.stderr,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    stdout_path = output_dir / "stdout.log"
    stderr_path = output_dir / "stderr.log"
    samples: list[dict[str, float | int]] = []
    baseline = _gpu_snapshot(gpu_device)
    started_at = datetime.now(UTC)
    started = time.monotonic()
    next_heartbeat = started + heartbeat_interval_s

    with (
        stdout_path.open("w", encoding="utf-8") as stdout,
        stderr_path.open("w", encoding="utf-8") as stderr,
    ):
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=os.environ.copy())
        while process.poll() is None:
            now = time.monotonic()
            snapshot = _gpu_snapshot(gpu_device)
            samples.append(
                {
                    "elapsed_s": now - started,
                    "memory_used_mb": snapshot["memory_used_mb"],
                }
            )
            if now >= next_heartbeat:
                print(
                    f"resource heartbeat classifier={classifier} "
                    f"elapsed_s={now - started:.1f} samples={len(samples)} "
                    f"memory_used_mb={snapshot['memory_used_mb']}",
                    file=heartbeat_output,
                    flush=True,
                )
                next_heartbeat = now + heartbeat_interval_s
            time.sleep(sample_interval_s)
        return_code = process.wait()

    finished = time.monotonic()
    final_snapshot = _gpu_snapshot(gpu_device)
    samples.append(
        {
            "elapsed_s": finished - started,
            "memory_used_mb": final_snapshot["memory_used_mb"],
        }
    )
    if return_code != 0:
        raise RuntimeError(f"{classifier} arm failed with exit {return_code}; see {stderr_path}")

    result_path = output_dir / "result.json"
    detections_path = output_dir / "detections.jsonl"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    peak_vram_mb = max(int(sample["memory_used_mb"]) for sample in samples)
    return {
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "wallclock_s": finished - started,
        "baseline_vram_mb": baseline["memory_used_mb"],
        "peak_vram_mb": peak_vram_mb,
        "peak_incremental_vram_mb": peak_vram_mb - int(baseline["memory_used_mb"]),
        "vram_sample_interval_s": sample_interval_s,
        "vram_samples": samples,
        "command": command,
        "result_schema_version": result["schema_version"],
        "result_diagnostics": result["diagnostics"],
        "result_total_frames": result["total_frames"],
        "detections_rows": sum(1 for _line in detections_path.open(encoding="utf-8")),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--gpu", required=True, help="Physical GPU index for Docker and nvidia-smi")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--docker", default="docker")
    parser.add_argument("--hf-cache-volume", default="cctv-gpu-engine_hf-cache")
    parser.add_argument("--sample-interval", type=float, default=0.5)
    parser.add_argument("--heartbeat-interval", type=float, default=10.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite immutable benchmark: {args.output}")
    if args.work_dir.exists():
        raise SystemExit(f"refusing to reuse benchmark work directory: {args.work_dir}")
    if not args.video.is_file():
        raise SystemExit(f"video does not exist: {args.video}")
    if args.sample_interval <= 0 or args.heartbeat_interval <= 0:
        raise SystemExit("sample and heartbeat intervals must be positive")

    image_id = _run_text([args.docker, "image", "inspect", "--format={{.Id}}", args.image])
    hardware = _gpu_snapshot(args.gpu)
    args.work_dir.mkdir(parents=True)
    arms: dict[str, dict] = {}
    for classifier in CLASSIFIERS:
        arm_dir = args.work_dir / classifier
        command = build_docker_command(
            docker=args.docker,
            image=args.image,
            video=args.video,
            output_dir=arm_dir,
            classifier=classifier,
            gpu_device=args.gpu,
            hf_cache_volume=args.hf_cache_volume,
        )
        arms[classifier] = _run_arm(
            command=command,
            classifier=classifier,
            output_dir=arm_dir,
            gpu_device=args.gpu,
            sample_interval_s=args.sample_interval,
            heartbeat_interval_s=args.heartbeat_interval,
        )

    artifact = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "methodology": {
            "fresh_container_per_arm": True,
            "tracking_enabled": True,
            "source_bind_mount": False,
            "same_image_video_gpu": True,
            "peak_vram_scope": "whole selected GPU sampled through nvidia-smi",
        },
        "image": {"reference": args.image, "id": image_id},
        "video": _video_metadata(args.video),
        "hardware": hardware,
        "arms": arms,
        "promotion_resource_gate": resource_promotion_gate(arms),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "resource gate "
        f"passed={artifact['promotion_resource_gate']['passed']} "
        f"artifact={args.output}",
        flush=True,
    )
    return 0 if artifact["promotion_resource_gate"]["passed"] else 2
