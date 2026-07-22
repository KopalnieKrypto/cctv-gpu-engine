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
from pipeline.detections_dump import detection_to_dict
from pipeline.frame_extractor import extract_frame_at
from pipeline.pose_detector import load_pose_model
from pipeline.postprocessing import CONFIDENCE_THRESHOLD, NMS_IOU_THRESHOLD
from pipeline.preprocessing import input_wh
from pipeline.reid import DEFAULT_REID_MODEL_PATH, load_reid_model
from pipeline.report_json import render_report_json
from pipeline.report_renderer import render_report
from pipeline.track_filter import MinTrackLengthFilter
from pipeline.tracker import DEFAULT_MAX_TRACK_AGE_S, PersonTracker
from pipeline.video_frames import iter_frames
from pipeline.zones import ZoneConfig, ZoneConfigError

DEFAULT_FPS = 1
DEFAULT_ACTIVITY_MODEL_PATH = "models/activity-mlp-v1.0.0.onnx"
DEFAULT_ACTIVITY_MODEL_METADATA_PATH = "models/activity-mlp-v1.0.0.json"

# How often (in processed frames) to fire the progress callback during a chunk.
# Each call doubles as a telemetry sampling tick for gpu_service.worker — too
# rare and gpu_util_peak is noise (sampled outside the hot path); too frequent
# and we burn R2 PUTs writing status.json. 3 frames at 1 fps → ~1 sample/3s,
# ~100 samples on a 5-min job — the sweet spot user asked for.
PROGRESS_FRAME_INTERVAL = 3


def _frames_recording_source_size(chunk: Path, fps: int, diagnostics: dict):
    """Yield ``(timestamp_s, frame)`` and stamp the source resolution once.

    ``diagnostics["source_frame"]`` is ``[w, h]`` of the analysed video (issue
    #98) — a result is only interpretable against the frame it was measured on,
    and the pixel height of a mid-field person is exactly what a resolution
    experiment turns on. Recorded from the first frame that reaches the
    pipeline rather than probed separately, so it can only ever describe the
    footage actually processed.
    """
    for timestamp_s, frame in iter_frames(str(chunk), fps=fps):
        if diagnostics.get("source_frame") is None:
            height, width = frame.shape[:2]
            diagnostics["source_frame"] = [int(width), int(height)]
        yield timestamp_s, frame


def _aggregate(
    aggregator: Aggregator,
    track_filter: MinTrackLengthFilter | None,
    timestamp_s: float,
    frame,
    detections: list,
) -> None:
    """Hand one analysed frame to the aggregator, through the filter if enabled.

    With filtering on, frames reach the aggregator a couple of steps late —
    that is the cost of only counting tracks that proved they persist — so
    nothing here may assume a frame is aggregated during its own iteration.
    """
    if track_filter is None:
        aggregator.add_frame(timestamp_s=timestamp_s, frame=frame, detections=detections)
        return
    for confirmed in track_filter.push(timestamp_s, frame, detections):
        aggregator.add_frame(
            timestamp_s=confirmed.timestamp_s,
            frame=confirmed.frame,
            detections=confirmed.detections,
        )


def _analyze_to_report_data(
    chunks: list[Path],
    progress: Callable[[int], None] | None = None,
    model_path: str = "models/yolo11s-pose.onnx",
    fps: int = DEFAULT_FPS,
    classifier: str = "heuristic",
    activity_model_path: str = DEFAULT_ACTIVITY_MODEL_PATH,
    activity_model_metadata_path: str = DEFAULT_ACTIVITY_MODEL_METADATA_PATH,
    dump_detections: Path | None = None,
    track_persons: bool = True,
    reid_model_path: str = DEFAULT_REID_MODEL_PATH,
    max_track_age_s: float = DEFAULT_MAX_TRACK_AGE_S,
    zones: ZoneConfig | None = None,
) -> ReportData:
    """Run YOLO-pose pipeline across one or more MP4 chunks → :class:`ReportData`.

    Shared core of :func:`run_full_video_to_json` (canonical artifact) and
    :func:`run_full_video_to_html` (debug-only standalone HTML). Multiple
    chunks are processed sequentially through a shared :class:`Aggregator`,
    with timestamps offset so each chunk continues from where the previous
    left off (so ``video_duration_s`` reflects the full concatenated video).

    ``classifier`` — ``"heuristic"`` uses geometric rules + displacement
    smoothing; ``"vlm"`` uses Qwen2.5-VL-3B + displacement; ``"mlp"`` uses
    the checksum-pinned per-detection ONNX classifier + per-track smoothing.

    ``progress`` — if given, called every ``PROGRESS_FRAME_INTERVAL`` frames
    *and* at every end-of-chunk boundary with an int percentage 0-100. The
    intra-chunk calls report the chunk-start percentage (so the value stays
    monotonically non-decreasing) but still tick the worker's telemetry
    sampler — without them, gpu_service.metrics.MetricsAggregator only sees
    chunk boundaries and ``gpu_util_peak`` is sampled outside the hot path.
    ``dump_detections`` — if given, one JSONL line per processed frame is
    streamed to this path (issue #35). The aggregator only keeps a bounded
    keyframe buffer (#49), so this raw per-frame archive can't be
    reconstructed afterwards — it must be written inside the loop. Opt-in;
    ``None`` (default) writes nothing. See :mod:`pipeline.detections_dump`.

    The pipeline ``RuntimeError`` (e.g. CUDA missing) is *not* caught here;
    callers (CLI, gpu-service worker) decide how to react.
    """
    import math

    from pipeline.activity_classifier import (
        DISPLACEMENT_WALK_THRESHOLD,
        ActivitySmoother,
        bbox_center,
    )
    from pipeline.detections_dump import DetectionsDumpWriter

    detector = load_pose_model(model_path, zones=zones)
    aggregator = Aggregator(
        fps=fps,
        zones=zones.zones if zones is not None else None,
        shift=zones.shift_schedule if zones is not None else None,
        restrict_to_zones=zones.restrict_to_zones if zones is not None else False,
    )

    # Identity first, then the min-track-length gate: the tracker decides *who*
    # each detection is, the filter decides whether that identity has earned the
    # right to count (issue #32). Both sit downstream of pose detection and
    # upstream of aggregation, so every consumer inherits the filtering.
    person_tracker = (
        PersonTracker(
            embedder=load_reid_model(reid_model_path),
            max_track_age_s=max_track_age_s,
        )
        if track_persons
        else None
    )
    track_filter = MinTrackLengthFilter() if track_persons else None
    # Resolved detection configuration (issue #98). Every value is read back
    # from what actually ran — the detector reports the weights it opened and
    # the input size the ONNX declares, so a bind-mount over ./models or a
    # non-default MODEL_PATH shows up here instead of silently changing
    # results. ``source_frame`` is filled in by the frame iterator below.
    diagnostics: dict = {
        "classifier": classifier,
        "activity_model": None,
        "model_path": detector.model_path,
        "model_sha256": detector.model_sha256,
        "input_size": list(input_wh(detector.input_size)),
        "conf_threshold": CONFIDENCE_THRESHOLD,
        "nms_threshold": NMS_IOU_THRESHOLD,
        "source_frame": None,
    }

    with DetectionsDumpWriter(dump_detections) as dump:
        # --- Per-person MLP path -------------------------------------------
        if classifier == "mlp":
            from pipeline.activity_features import FEATURE_SCHEMA_VERSION
            from pipeline.mlp_classifier import TrackActivitySmoother, load_activity_mlp

            mlp = load_activity_mlp(activity_model_path, activity_model_metadata_path)
            smoother = TrackActivitySmoother()
            diagnostics["activity_model"] = {
                "version": mlp.model_version,
                "sha256": mlp.model_sha256,
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
            }

            time_offset = 0.0
            step = 1.0 / fps
            frames_since_progress = 0
            for chunk_index, chunk in enumerate(chunks):
                chunk_start_pct = int(chunk_index / len(chunks) * 100)
                last_ts = time_offset
                for timestamp_s, frame in _frames_recording_source_size(chunk, fps, diagnostics):
                    shifted = timestamp_s + time_offset
                    detections = detector.detect(frame)
                    if person_tracker is not None:
                        detections = person_tracker.update(frame, detections, shifted)
                    for detection in detections:
                        detection.activity = mlp.classify(detection)
                    detections = smoother.smooth(detections)
                    dump.write_frame(shifted, detections)
                    _aggregate(aggregator, track_filter, shifted, frame, detections)
                    last_ts = shifted
                    frames_since_progress += 1
                    if progress is not None and frames_since_progress >= PROGRESS_FRAME_INTERVAL:
                        frames_since_progress = 0
                        progress(chunk_start_pct)
                time_offset = last_ts + step
                if progress is not None:
                    progress(int((chunk_index + 1) / len(chunks) * 100))

        # --- VLM hybrid path -----------------------------------------------
        elif classifier == "vlm":
            from pipeline.vlm_classifier import VLMClassifier

            vlm = VLMClassifier()
            prev_centers: list[tuple[float, float]] = []

            time_offset = 0.0
            step = 1.0 / fps
            frames_since_progress = 0
            for chunk_index, chunk in enumerate(chunks):
                chunk_start_pct = int(chunk_index / len(chunks) * 100)
                last_ts = time_offset
                for timestamp_s, frame in _frames_recording_source_size(chunk, fps, diagnostics):
                    shifted = timestamp_s + time_offset
                    detections = detector.detect(frame)
                    if person_tracker is not None:
                        detections = person_tracker.update(frame, detections, shifted)

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
                                nearest = min(
                                    math.hypot(cx - px, cy - py) for px, py in prev_centers
                                )
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

                    dump.write_frame(shifted, detections)
                    _aggregate(aggregator, track_filter, shifted, frame, detections)
                    last_ts = shifted
                    frames_since_progress += 1
                    if progress is not None and frames_since_progress >= PROGRESS_FRAME_INTERVAL:
                        frames_since_progress = 0
                        progress(chunk_start_pct)
                time_offset = last_ts + step
                if progress is not None:
                    progress(int((chunk_index + 1) / len(chunks) * 100))

        # --- Heuristic path (original) -------------------------------------
        else:
            smoother = ActivitySmoother()
            time_offset = 0.0
            step = 1.0 / fps
            frames_since_progress = 0
            for chunk_index, chunk in enumerate(chunks):
                chunk_start_pct = int(chunk_index / len(chunks) * 100)
                last_ts = time_offset
                for timestamp_s, frame in _frames_recording_source_size(chunk, fps, diagnostics):
                    shifted = timestamp_s + time_offset
                    detections = detector.detect(frame)
                    if person_tracker is not None:
                        detections = person_tracker.update(frame, detections, shifted)
                    detections = smoother.smooth(detections)
                    dump.write_frame(shifted, detections)
                    _aggregate(aggregator, track_filter, shifted, frame, detections)
                    last_ts = shifted
                    frames_since_progress += 1
                    if progress is not None and frames_since_progress >= PROGRESS_FRAME_INTERVAL:
                        frames_since_progress = 0
                        progress(chunk_start_pct)
                time_offset = last_ts + step
                if progress is not None:
                    progress(int((chunk_index + 1) / len(chunks) * 100))

    if track_filter is not None:
        # End of video: whatever is still held back is judged on what we know
        # now. A track that never reached the minimum dies here.
        for confirmed in track_filter.flush():
            aggregator.add_frame(
                timestamp_s=confirmed.timestamp_s,
                frame=confirmed.frame,
                detections=confirmed.detections,
            )

    report = aggregator.build_report_data()
    report.diagnostics = diagnostics
    return report


def run_full_video_to_json(
    chunks: list[Path],
    progress: Callable[[int], None] | None = None,
    model_path: str = "models/yolo11s-pose.onnx",
    fps: int = DEFAULT_FPS,
    classifier: str = "heuristic",
    activity_model_path: str = DEFAULT_ACTIVITY_MODEL_PATH,
    activity_model_metadata_path: str = DEFAULT_ACTIVITY_MODEL_METADATA_PATH,
    dump_detections: Path | None = None,
    track_persons: bool = True,
    reid_model_path: str = DEFAULT_REID_MODEL_PATH,
    max_track_age_s: float = DEFAULT_MAX_TRACK_AGE_S,
    zones: ZoneConfig | None = None,
) -> bytes:
    """Run the pipeline and return the canonical ``result.json`` bytes (issue #72).

    This is the primary job artifact — the bytes the gpu-agent uploads. The
    platform renders it natively in React, so the JSON is pure data (base64
    JPEG keyframes, no brand/i18n/presentation strings). See
    :func:`_analyze_to_report_data` for the
    ``progress``/``classifier``/``dump_detections``/``track_persons``/``zones``
    contract.
    """
    return render_report_json(
        _analyze_to_report_data(
            chunks,
            progress=progress,
            model_path=model_path,
            fps=fps,
            classifier=classifier,
            activity_model_path=activity_model_path,
            activity_model_metadata_path=activity_model_metadata_path,
            dump_detections=dump_detections,
            track_persons=track_persons,
            reid_model_path=reid_model_path,
            max_track_age_s=max_track_age_s,
            zones=zones,
        )
    )


def run_full_video_to_html(
    chunks: list[Path],
    progress: Callable[[int], None] | None = None,
    model_path: str = "models/yolo11s-pose.onnx",
    fps: int = DEFAULT_FPS,
    classifier: str = "heuristic",
    activity_model_path: str = DEFAULT_ACTIVITY_MODEL_PATH,
    activity_model_metadata_path: str = DEFAULT_ACTIVITY_MODEL_METADATA_PATH,
    dump_detections: Path | None = None,
    track_persons: bool = True,
    reid_model_path: str = DEFAULT_REID_MODEL_PATH,
    max_track_age_s: float = DEFAULT_MAX_TRACK_AGE_S,
    zones: ZoneConfig | None = None,
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
            activity_model_path=activity_model_path,
            activity_model_metadata_path=activity_model_metadata_path,
            dump_detections=dump_detections,
            track_persons=track_persons,
            reid_model_path=reid_model_path,
            max_track_age_s=max_track_age_s,
            zones=zones,
        )
    ).encode("utf-8")


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
        choices=["heuristic", "vlm", "mlp"],
        default="heuristic",
        help="Activity classifier: heuristic (fast, geometric rules) "
        "vlm (Qwen2.5-VL, deployed quality default), or mlp "
        "(experimental per-person ONNX; default: heuristic)",
    )
    parser.add_argument(
        "--activity-model",
        default=DEFAULT_ACTIVITY_MODEL_PATH,
        help="Path to the activity MLP ONNX weights (used only with --classifier mlp)",
    )
    parser.add_argument(
        "--activity-model-metadata",
        default=DEFAULT_ACTIVITY_MODEL_METADATA_PATH,
        help="Path to the checksummed activity MLP metadata sidecar",
    )
    parser.add_argument(
        "--format",
        choices=["json", "html"],
        default="json",
        help="Full-video output format: json (canonical result.json artifact, "
        "default) or html (debug-only standalone report)",
    )
    parser.add_argument(
        "--reid-model",
        default=DEFAULT_REID_MODEL_PATH,
        help="Path to the OSNet Re-ID ONNX model used for person tracking",
    )
    parser.add_argument(
        "--no-tracker",
        dest="track_persons",
        action="store_false",
        help="Disable person tracking, counting every detection as before "
        "issue #32. Sporadic false positives (the bench-on-wheels) count as "
        "people again — use only to reproduce pre-tracking baselines.",
    )
    parser.add_argument(
        "--max-track-age",
        type=float,
        default=DEFAULT_MAX_TRACK_AGE_S,
        metavar="SECONDS",
        help="How long a person may leave the frame and still be recognised as "
        "the same person on return. Longer re-matches more returners but risks "
        f"confusing two people for one (default: {DEFAULT_MAX_TRACK_AGE_S:.0f})",
    )
    parser.add_argument(
        "--dump-detections",
        type=str,
        default=None,
        metavar="PATH.jsonl",
        help="Full-video mode: also archive raw per-frame detections as JSONL "
        "at PATH (one line per frame; opt-in, for post-hoc analysis). "
        "Default: off.",
    )
    parser.add_argument(
        "--zones",
        type=str,
        default=None,
        metavar="zones.json",
        help="Full-video mode: ROI zone config (issues #78/#79). Each detection "
        "is assigned to a zone by its foot point (per-zone posture breakdown); an "
        "optional 'shift' schedule plus 'recording_start' gates analysis to the "
        "working windows minus breaks. An optional top-level 'inference_roi' "
        "focuses the single pose call on one named zone plus margin while keeping "
        "all returned coordinates in full-frame pixels. Default: off.",
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
        "persons": [detection_to_dict(d) for d in detections],
    }
    json.dump(payload, sys.stdout)
    sys.stdout.write("\n")
    return 0


def _run_full_video(args: argparse.Namespace) -> int:
    render = run_full_video_to_html if args.format == "html" else run_full_video_to_json
    dump_path = Path(args.dump_detections) if args.dump_detections else None
    try:
        zones = ZoneConfig.load(args.zones) if args.zones else None
        payload = render(
            chunks=[Path(args.video)],
            model_path=args.model,
            fps=args.fps,
            classifier=args.classifier,
            activity_model_path=args.activity_model,
            activity_model_metadata_path=args.activity_model_metadata,
            dump_detections=dump_path,
            track_persons=args.track_persons,
            reid_model_path=args.reid_model,
            max_track_age_s=args.max_track_age,
            zones=zones,
        )
    except (RuntimeError, ZoneConfigError) as exc:
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
