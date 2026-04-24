"""YOLO-pose surveillance CLI.

Two modes:

* **Single-frame** (issue #2 PoC) — ``--timestamp T`` runs pose inference on
  one frame at ``T`` seconds and prints a JSON document to stdout. Useful as
  a smoke-test for the GPU stack on a fresh machine.

* **Full-video** (issue #4) — ``--output report.html`` streams the entire
  video at 1 fps via ffmpeg, runs pose inference on every frame, aggregates
  the detections into a :class:`pipeline.aggregator.ReportData`, and writes
  a self-contained HTML report (vendored Chart.js, base64 keyframes).

The two modes share the same model loader and Detection objects so any GPU
issue (missing CUDA EP, silent CPU fallback) raises the same RuntimeError
in both paths.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

from pipeline.aggregator import Aggregator
from pipeline.frame_extractor import extract_frame_at
from pipeline.pose_detector import load_pose_model
from pipeline.postprocessing import Detection
from pipeline.report_renderer import render_report
from pipeline.video_frames import iter_frames

DEFAULT_FPS = 1


def run_full_video_to_html(
    chunks: list[Path],
    progress: Callable[[int], None] | None = None,
    model_path: str = "models/yolo11n-pose.onnx",
    fps: int = DEFAULT_FPS,
) -> bytes:
    """Run YOLO-pose pipeline across one or more MP4 chunks.

    Returns the standalone HTML report as bytes (no disk write). Multiple
    chunks are processed sequentially through a shared :class:`Aggregator`,
    with timestamps offset so each chunk continues from where the previous
    left off (so ``video_duration_s`` reflects the full concatenated video).

    ``progress`` — if given, called at end-of-chunk boundaries with an int
    percentage 0-100. The pipeline ``RuntimeError`` (e.g. CUDA missing) is
    *not* caught here; callers (CLI, gpu-service worker) decide how to react.
    """
    from pipeline.activity_classifier import ActivitySmoother

    detector = load_pose_model(model_path)
    smoother = ActivitySmoother()
    aggregator = Aggregator(fps=fps)
    time_offset = 0.0
    step = 1.0 / fps
    for chunk_index, chunk in enumerate(chunks):
        last_ts = time_offset
        for timestamp_s, frame in iter_frames(str(chunk), fps=fps):
            shifted = timestamp_s + time_offset
            detections = detector.detect(frame)
            detections = smoother.smooth(detections)
            aggregator.add_frame(timestamp_s=shifted, frame=frame, detections=detections)
            last_ts = shifted
        time_offset = last_ts + step
        if progress is not None:
            progress(int((chunk_index + 1) / len(chunks) * 100))
    report_data = aggregator.build_report_data()
    return render_report(report_data).encode("utf-8")


def _detection_to_dict(det: Detection) -> dict:
    return {
        "bbox": [float(v) for v in det.bbox],
        "confidence": float(det.confidence),
        "activity": det.activity,
        "keypoints": [
            {"x": float(kp.x), "y": float(kp.y), "vis": float(kp.vis)} for kp in det.keypoints
        ],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pipeline.analyze",
        description="YOLO-pose surveillance analysis (single-frame or full-video mode)",
    )
    parser.add_argument("video", help="Path to input MP4 file")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--timestamp",
        type=float,
        help="Single-frame mode: seek timestamp in seconds (e.g. 12.5)",
    )
    mode.add_argument(
        "--output",
        type=str,
        help="Full-video mode: path to write the standalone HTML report",
    )
    parser.add_argument(
        "--model",
        default="models/yolo11n-pose.onnx",
        help="Path to the YOLO-pose ONNX model file",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help="Sampling frame rate for full-video mode (default: 1)",
    )
    return parser


def _run_single_frame(args: argparse.Namespace) -> int:
    try:
        frame = extract_frame_at(Path(args.video), timestamp_s=args.timestamp)
        detector = load_pose_model(args.model)
        detections = detector.detect(frame)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    payload = {
        "video": args.video,
        "timestamp_s": args.timestamp,
        "person_count": len(detections),
        "persons": [_detection_to_dict(d) for d in detections],
    }
    json.dump(payload, sys.stdout)
    sys.stdout.write("\n")
    return 0


def _run_full_video(args: argparse.Namespace) -> int:
    try:
        html = run_full_video_to_html(
            chunks=[Path(args.video)],
            model_path=args.model,
            fps=args.fps,
        )
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(html)
    print(f"wrote report to {out_path}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.output is not None:
        return _run_full_video(args)
    return _run_single_frame(args)


if __name__ == "__main__":
    raise SystemExit(main())
