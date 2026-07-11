"""YOLO-pose surveillance CLI.

Two modes:

* **Single-frame** (issue #2 PoC) — ``--timestamp T`` runs pose inference on
  one frame at ``T`` seconds and prints a JSON document to stdout. Useful as
  a smoke-test for the GPU stack on a fresh machine.

* **Full-video** (issue #4) — ``--output result.json`` streams the entire
  video at 1 fps via ffmpeg, runs pose inference on every frame, aggregates
  the detections into a :class:`pipeline.aggregator.ReportData`, and writes
  the canonical structured ``result.json`` artifact (issue #72) that the
  platform renders natively. ``--format html`` still writes the legacy
  self-contained HTML report (vendored Chart.js, base64 keyframes) for
  debugging, but it is no longer the canonical output.

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

from pipeline.aggregator import Aggregator, ReportData
from pipeline.frame_extractor import extract_frame_at
from pipeline.pose_detector import load_pose_model
from pipeline.postprocessing import Detection
from pipeline.report_json import render_report_json
from pipeline.report_renderer import render_report
from pipeline.video_frames import iter_frames

DEFAULT_FPS = 1

# How often (in processed frames) to fire the progress callback during a chunk.
# Each call doubles as a telemetry sampling tick for gpu_service.worker — too
# rare and gpu_util_peak is noise (sampled outside the hot path); too frequent
# and we burn R2 PUTs writing status.json. 3 frames at 1 fps → ~1 sample/3s,
# ~100 samples on a 5-min job — the sweet spot user asked for.
PROGRESS_FRAME_INTERVAL = 3


def _analyze_to_report_data(
    chunks: list[Path],
    progress: Callable[[int], None] | None = None,
    model_path: str = "models/yolo11s-pose.onnx",
    fps: int = DEFAULT_FPS,
    classifier: str = "heuristic",
) -> ReportData:
    """Run YOLO-pose pipeline across one or more MP4 chunks → :class:`ReportData`.

    Shared core of :func:`run_full_video_to_json` (canonical artifact) and
    :func:`run_full_video_to_html` (debug-only standalone HTML). Multiple
    chunks are processed sequentially through a shared :class:`Aggregator`,
    with timestamps offset so each chunk continues from where the previous
    left off (so ``video_duration_s`` reflects the full concatenated video).

    ``classifier`` — ``"heuristic"`` uses geometric rules + displacement
    smoother (fast, camera-angle dependent).  ``"vlm"`` uses Qwen2.5-VL-3B
    for sitting/standing classification + YOLO displacement for walking
    (slower, camera-angle independent, much better accuracy).

    ``progress`` — if given, called every ``PROGRESS_FRAME_INTERVAL`` frames
    *and* at every end-of-chunk boundary with an int percentage 0-100. The
    intra-chunk calls report the chunk-start percentage (so the value stays
    monotonically non-decreasing) but still tick the worker's telemetry
    sampler — without them, gpu_service.metrics.MetricsAggregator only sees
    chunk boundaries and ``gpu_util_peak`` is sampled outside the hot path.
    The pipeline ``RuntimeError`` (e.g. CUDA missing) is *not* caught here;
    callers (CLI, gpu-service worker) decide how to react.
    """
    import math

    from pipeline.activity_classifier import (
        DISPLACEMENT_WALK_THRESHOLD,
        ActivitySmoother,
        bbox_center,
    )

    detector = load_pose_model(model_path)
    aggregator = Aggregator(fps=fps)

    # --- VLM hybrid path ---------------------------------------------------
    if classifier == "vlm":
        from pipeline.vlm_classifier import VLMClassifier

        vlm = VLMClassifier()
        prev_centers: list[tuple[float, float]] = []

        time_offset = 0.0
        step = 1.0 / fps
        frames_since_progress = 0
        for chunk_index, chunk in enumerate(chunks):
            chunk_start_pct = int(chunk_index / len(chunks) * 100)
            last_ts = time_offset
            for timestamp_s, frame in iter_frames(str(chunk), fps=fps):
                shifted = timestamp_s + time_offset
                detections = detector.detect(frame)

                if detections:
                    # Per-person walking decision via nearest-center matching
                    # (issue #64). NMS orders detections by confidence, which is
                    # not stable across frames, so pairing detections[0] with
                    # prev_centers[0] compared different people and flipped the
                    # branch at random in multi-person scenes. Match each
                    # detection to its *nearest* previous center instead — the
                    # same order-independent approach ActivitySmoother uses.
                    #
                    # The VLM classifies the whole frame (no per-person crops —
                    # deferred with full tracking, #32), so its single
                    # sitting/standing label is shared by every non-moving
                    # person. It is computed lazily and at most once per frame
                    # (skipped entirely when everyone is walking), so cost is
                    # unchanged from the old frame-level call.
                    vlm_label: str | None = None
                    for det in detections:
                        cx, cy = bbox_center(det)
                        bbox_h = det.bbox[3] - det.bbox[1]
                        is_moving = False
                        if prev_centers and bbox_h > 1e-6:
                            nearest = min(math.hypot(cx - px, cy - py) for px, py in prev_centers)
                            is_moving = nearest / bbox_h > DISPLACEMENT_WALK_THRESHOLD

                        if is_moving:
                            det.activity = "walking"
                        else:
                            if vlm_label is None:
                                vlm_label = vlm.classify_frame(frame)
                            det.activity = vlm_label

                    prev_centers = [bbox_center(d) for d in detections]
                else:
                    prev_centers = []

                aggregator.add_frame(timestamp_s=shifted, frame=frame, detections=detections)
                last_ts = shifted
                frames_since_progress += 1
                if progress is not None and frames_since_progress >= PROGRESS_FRAME_INTERVAL:
                    frames_since_progress = 0
                    progress(chunk_start_pct)
            time_offset = last_ts + step
            if progress is not None:
                progress(int((chunk_index + 1) / len(chunks) * 100))

    # --- Heuristic path (original) -----------------------------------------
    else:
        smoother = ActivitySmoother()
        time_offset = 0.0
        step = 1.0 / fps
        frames_since_progress = 0
        for chunk_index, chunk in enumerate(chunks):
            chunk_start_pct = int(chunk_index / len(chunks) * 100)
            last_ts = time_offset
            for timestamp_s, frame in iter_frames(str(chunk), fps=fps):
                shifted = timestamp_s + time_offset
                detections = detector.detect(frame)
                detections = smoother.smooth(detections)
                aggregator.add_frame(timestamp_s=shifted, frame=frame, detections=detections)
                last_ts = shifted
                frames_since_progress += 1
                if progress is not None and frames_since_progress >= PROGRESS_FRAME_INTERVAL:
                    frames_since_progress = 0
                    progress(chunk_start_pct)
            time_offset = last_ts + step
            if progress is not None:
                progress(int((chunk_index + 1) / len(chunks) * 100))

    return aggregator.build_report_data()


def run_full_video_to_json(
    chunks: list[Path],
    progress: Callable[[int], None] | None = None,
    model_path: str = "models/yolo11s-pose.onnx",
    fps: int = DEFAULT_FPS,
    classifier: str = "heuristic",
) -> bytes:
    """Run the pipeline and return the canonical ``result.json`` bytes (issue #72).

    This is the primary job artifact — the bytes the gpu-agent uploads. The
    platform renders it natively in React, so the JSON is pure data (base64
    JPEG keyframes, no brand/i18n/presentation strings). See
    :func:`_analyze_to_report_data` for the ``progress``/``classifier`` contract.
    """
    return render_report_json(
        _analyze_to_report_data(
            chunks,
            progress=progress,
            model_path=model_path,
            fps=fps,
            classifier=classifier,
        )
    )


def run_full_video_to_html(
    chunks: list[Path],
    progress: Callable[[int], None] | None = None,
    model_path: str = "models/yolo11s-pose.onnx",
    fps: int = DEFAULT_FPS,
    classifier: str = "heuristic",
) -> bytes:
    """Run the pipeline and return a standalone HTML report as bytes.

    Debug-only since issue #72: the canonical artifact is now
    :func:`run_full_video_to_json`. The Jinja/Chart.js report is kept behind
    the CLI ``--format html`` flag for local inspection but is no longer what
    the worker/gpu-agent upload.
    """
    return render_report(
        _analyze_to_report_data(
            chunks,
            progress=progress,
            model_path=model_path,
            fps=fps,
            classifier=classifier,
        )
    ).encode("utf-8")


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
        help="Full-video mode: path to write the canonical result.json artifact "
        "(or the standalone HTML report with --format html)",
    )
    parser.add_argument(
        "--model",
        default="models/yolo11s-pose.onnx",
        help="Path to the YOLO-pose ONNX model file",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help="Sampling frame rate for full-video mode (default: 1)",
    )
    parser.add_argument(
        "--classifier",
        choices=["heuristic", "vlm"],
        default="heuristic",
        help="Activity classifier: heuristic (fast, geometric rules) "
        "or vlm (Qwen2.5-VL, higher accuracy, default: heuristic)",
    )
    parser.add_argument(
        "--format",
        choices=["json", "html"],
        default="json",
        help="Full-video output format: json (canonical result.json artifact, "
        "default) or html (debug-only standalone report)",
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
    render = run_full_video_to_html if args.format == "html" else run_full_video_to_json
    try:
        payload = render(
            chunks=[Path(args.video)],
            model_path=args.model,
            fps=args.fps,
            classifier=args.classifier,
        )
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(payload)
    print(f"wrote {args.format} report to {out_path}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.output is not None:
        return _run_full_video(args)
    return _run_single_frame(args)


if __name__ == "__main__":
    raise SystemExit(main())
