"""Single-frame YOLO-pose CLI — issue #2 proof of concept.

Usage::

    python -m pipeline.analyze video.mp4 --timestamp 12.5 \\
        --model models/yolo11n-pose.onnx

Extracts one frame at the given timestamp, runs YOLO-pose inference on the
GPU, classifies activity for each detected person, and prints a JSON document
to stdout. Exits with a non-zero status (and logs to stderr) on any pipeline
failure — most importantly when CUDA is unavailable, so an investor running
this on a machine without a GPU sees a clear message instead of a silent
fallback.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pipeline.frame_extractor import extract_frame_at
from pipeline.pose_detector import load_pose_model
from pipeline.postprocessing import Detection


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
        description="Single-frame YOLO-pose surveillance analysis (issue #2 PoC)",
    )
    parser.add_argument("video", help="Path to input MP4 file")
    parser.add_argument(
        "--timestamp",
        type=float,
        required=True,
        help="Seek timestamp in seconds (e.g. 12.5)",
    )
    parser.add_argument(
        "--model",
        default="models/yolo11n-pose.onnx",
        help="Path to the YOLO-pose ONNX model file",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

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


if __name__ == "__main__":
    raise SystemExit(main())
