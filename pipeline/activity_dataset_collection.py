"""GPU-assisted frame and COCO-17 pre-label collection for issue #33."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import cv2

from pipeline.activity_classifier import ActivitySmoother
from pipeline.pose_detector import load_pose_model
from pipeline.postprocessing import Detection
from pipeline.video_frames import iter_frames

_HEARTBEAT_SECONDS = 60.0
_HEARTBEAT_FRAMES = 100


def resize_full_frame(frame: Any, *, max_width: int) -> Any:
    """Downscale a full frame to ``max_width`` without cropping or upscaling."""
    frame_height, frame_width = frame.shape[:2]
    if frame_width <= max_width:
        return frame
    scale = max_width / frame_width
    resized_height = round(frame_height * scale)
    return cv2.resize(frame, (max_width, resized_height), interpolation=cv2.INTER_AREA)


def activity_at_timestamp(intervals: list[dict[str, Any]], timestamp_s: float) -> str | None:
    """Return the activity for a start-inclusive/end-exclusive interval."""
    for interval in intervals:
        if interval["start_s"] <= timestamp_s < interval["end_s"]:
            return str(interval["activity"])
    return None


def select_primary_detection(
    detections: list[Detection],
    *,
    min_bbox_height: float,
    min_confidence: float,
) -> Detection | None:
    """Return the highest-confidence reviewable person in a frame, if any."""
    eligible = [
        detection
        for detection in detections
        if detection.confidence >= min_confidence
        and detection.bbox[3] - detection.bbox[1] >= min_bbox_height
    ]
    return max(eligible, key=lambda detection: detection.confidence, default=None)


def candidate_from_detection(
    detection: Detection,
    *,
    activity: str,
    camera_geometry_id: str,
    frame_height: int,
    frame_path: str,
    frame_sha256: str,
    frame_width: int,
    sample_id: str,
    source_id: str,
    source_timestamp_s: float,
    source_video_sha256: str,
) -> dict[str, Any]:
    """Convert one YOLO detection into one full-frame review candidate."""
    x1, y1, x2, y2 = detection.bbox
    x1 = max(0.0, min(float(frame_width), x1))
    y1 = max(0.0, min(float(frame_height), y1))
    x2 = max(0.0, min(float(frame_width), x2))
    y2 = max(0.0, min(float(frame_height), y2))

    keypoints = []
    for keypoint in detection.keypoints:
        in_frame = 0 <= keypoint.x < frame_width and 0 <= keypoint.y < frame_height
        keypoints.append(
            {
                "x": max(0.0, min(float(frame_width - 1), keypoint.x)),
                "y": max(0.0, min(float(frame_height - 1), keypoint.y)),
                "vis": keypoint.vis if in_frame else 0.0,
            }
        )
    return {
        "activity": activity,
        "bbox": [x1, y1, x2 - x1, y2 - y1],
        "camera_geometry_id": camera_geometry_id,
        "frame_height": frame_height,
        "frame_path": frame_path,
        "frame_sha256": frame_sha256,
        "frame_width": frame_width,
        "keypoints": keypoints,
        "pose_confidence": detection.confidence,
        "review_status": "pending",
        "sample_id": sample_id,
        "source_id": source_id,
        "source_timestamp_s": source_timestamp_s,
        "source_video_sha256": source_video_sha256,
        "split": "unassigned",
        "synthetic": False,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_candidate_frame(frame: Any, path: Path, *, jpeg_quality: int) -> str:
    encoded, payload = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    if not encoded:
        raise RuntimeError(f"failed to encode candidate frame {path}")
    frame_bytes = payload.tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(frame_bytes)
    return hashlib.sha256(frame_bytes).hexdigest()


def collect_candidates(
    *,
    manifest_path: Path,
    model_path: Path,
    output_dir: Path,
    source_root: Path,
    jpeg_quality: int = 82,
    only_source_ids: set[str] | None = None,
) -> dict[str, int]:
    """Run CUDA pose inference and materialize one review candidate per frame."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sources = manifest["sources"]
    if only_source_ids:
        sources = [source for source in sources if source["source_id"] in only_source_ids]
    if not sources:
        raise RuntimeError("source selection matched no manifest entries")

    output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = output_dir / "candidates.jsonl"
    if labels_path.exists():
        raise RuntimeError(
            f"refusing to append to existing candidate file {labels_path}; "
            "choose a fresh output directory"
        )

    print(f"loading CUDA pose model: {model_path}", flush=True)
    detector = load_pose_model(str(model_path))
    total_decoded = 0
    total_candidates = 0
    per_source: dict[str, int] = {}
    with labels_path.open("w", encoding="utf-8") as labels_file:
        for source in sources:
            source_id = source["source_id"]
            geometry_id = source["camera_geometry_id"]
            video_path = source_root / source["video_path"]
            actual_video_sha256 = _sha256_file(video_path)
            expected_video_sha256 = source["sha256"]
            if actual_video_sha256 != expected_video_sha256:
                raise RuntimeError(
                    f"source checksum mismatch for {video_path}: expected "
                    f"{expected_video_sha256}, found {actual_video_sha256}"
                )

            print(
                f"source-start id={source_id} video={source['video_path']} fps={source['fps']}",
                flush=True,
            )
            source_candidates = 0
            source_decoded = 0
            smoother = ActivitySmoother()
            last_heartbeat = time.monotonic()
            for timestamp_s, frame in iter_frames(video_path, fps=int(source["fps"])):
                source_decoded += 1
                total_decoded += 1
                activity_hint = activity_at_timestamp(source["intervals"], timestamp_s)
                if activity_hint is not None:
                    frame = resize_full_frame(
                        frame, max_width=int(source.get("max_frame_width", 1280))
                    )
                    detections = detector.detect(frame)
                    if activity_hint == "__model__":
                        detections = smoother.smooth(detections)
                    primary = select_primary_detection(
                        detections,
                        min_bbox_height=float(source.get("min_bbox_height", 40.0)),
                        min_confidence=float(source.get("min_confidence", 0.35)),
                    )
                    if primary is not None:
                        activity = (
                            primary.activity if activity_hint == "__model__" else activity_hint
                        )
                        timestamp_ms = round(timestamp_s * 1000)
                        video_token = actual_video_sha256[:12]
                        sample_id = f"{geometry_id}-{video_token}-{timestamp_ms:09d}"
                        relative_frame_path = Path(geometry_id) / "candidates" / f"{sample_id}.jpg"
                        frame_height, frame_width = frame.shape[:2]
                        frame_sha256 = _write_candidate_frame(
                            frame,
                            output_dir / relative_frame_path,
                            jpeg_quality=jpeg_quality,
                        )
                        candidate = candidate_from_detection(
                            primary,
                            activity=activity,
                            camera_geometry_id=geometry_id,
                            frame_height=frame_height,
                            frame_path=relative_frame_path.as_posix(),
                            frame_sha256=frame_sha256,
                            frame_width=frame_width,
                            sample_id=sample_id,
                            source_id=source_id,
                            source_timestamp_s=timestamp_s,
                            source_video_sha256=actual_video_sha256,
                        )
                        labels_file.write(json.dumps(candidate, sort_keys=True) + "\n")
                        source_candidates += 1
                        total_candidates += 1

                now = time.monotonic()
                if (
                    source_decoded % _HEARTBEAT_FRAMES == 0
                    or now - last_heartbeat >= _HEARTBEAT_SECONDS
                ):
                    labels_file.flush()
                    print(
                        f"heartbeat id={source_id} decoded={source_decoded} "
                        f"candidates={source_candidates}",
                        flush=True,
                    )
                    last_heartbeat = now

            labels_file.flush()
            per_source[source_id] = per_source.get(source_id, 0) + source_candidates
            print(
                f"source-complete id={source_id} decoded={source_decoded} "
                f"candidates={source_candidates}",
                flush=True,
            )

    summary: dict[str, int] = {
        "decoded_frames": total_decoded,
        "candidates": total_candidates,
        **{f"source:{source_id}": count for source_id, count in sorted(per_source.items())},
    }
    (output_dir / "collection-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--jpeg-quality", type=int, default=82)
    parser.add_argument("--only-source", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    collect_candidates(
        manifest_path=args.manifest,
        model_path=args.model,
        output_dir=args.output,
        source_root=args.source_root,
        jpeg_quality=args.jpeg_quality,
        only_source_ids=set(args.only_source),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised on the GPU VPS
    raise SystemExit(main())
