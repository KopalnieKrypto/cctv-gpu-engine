"""Video pose inference smoke test — standalone runner for the GPU VPS.

This script is the integration counterpart to the unit-tested ``pipeline``
package: it downloads a sample video, extracts frames at 1 fps, runs the same
preprocess/postprocess/classify code that ``pipeline.pose_detector`` uses, and
saves a few annotated keyframes plus a per-frame summary to stdout.

It exists so we can sanity-check the full GPU stack on the test VPS in one
shot. Production logic lives in ``pipeline/`` and is covered by ``pytest``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# Make the repository's `pipeline` package importable when this script is
# invoked directly via `python test_video_pose.py` from the test/ directory.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from pipeline.pose_detector import PoseDetector, load_pose_model  # noqa: E402
from pipeline.postprocessing import Detection  # noqa: E402

# ── Constants ──────────────────────────────────────────────────────────────────

ACTIVITY_COLORS = {
    "sitting": (0, 200, 0),
    "standing": (0, 120, 255),
    "walking": (255, 165, 0),
    "running": (255, 0, 0),
    "unknown": (128, 128, 128),
}

# Skeleton edges for COCO 17.
SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),               # head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),      # torso + arms
    (5, 11), (6, 12), (11, 12),                    # torso
    (11, 13), (13, 15), (12, 14), (14, 16),        # legs
]

# Sample videos for different activity testing (Mixkit, free, no watermark)
SAMPLE_VIDEOS = {
    "walking": "https://assets.mixkit.co/videos/34563/34563-720.mp4",
    "park": "https://assets.mixkit.co/videos/4401/4401-720.mp4",
    "street": "https://assets.mixkit.co/videos/4048/4048-720.mp4",
}


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class FrameResult:
    timestamp_s: float
    person_count: int
    activities: dict[str, int] = field(default_factory=dict)
    detections: list[Detection] = field(default_factory=list)


# ── Drawing / annotation ──────────────────────────────────────────────────────

def draw_annotations(img_bgr: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Draw bboxes, skeleton, and activity labels on a BGR image."""
    img = img_bgr.copy()
    for det in detections:
        color = ACTIVITY_COLORS.get(det.activity, ACTIVITY_COLORS["unknown"])
        x1, y1, x2, y2 = (int(v) for v in det.bbox)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        label = f"{det.activity} {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        kps = det.keypoints
        for (a, b) in SKELETON_EDGES:
            if kps[a].vis > 0.5 and kps[b].vis > 0.5:
                cv2.line(img,
                         (int(kps[a].x), int(kps[a].y)),
                         (int(kps[b].x), int(kps[b].y)),
                         color, 2)
        for kp in kps:
            if kp.vis > 0.5:
                cv2.circle(img, (int(kp.x), int(kp.y)), 4, (0, 255, 255), -1)

    return img


# ── Main pipeline ──────────────────────────────────────────────────────────────

def download_video(url: str, dest: Path) -> Path:
    if dest.exists():
        print(f"  Video already at {dest}")
        return dest
    print(f"  Downloading sample video from {url}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
        f.write(resp.read())
    print(f"  Saved to {dest} ({dest.stat().st_size / 1024 / 1024:.1f} MB)")
    return dest


def extract_frames(video_path: Path, frames_dir: Path, fps: int = 1) -> list[Path]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(frames_dir.glob("frame_*.jpg"))
    if existing:
        print(f"  {len(existing)} frames already extracted")
        return existing

    print(f"  Extracting frames at {fps}fps...")
    subprocess.run(
        ["ffmpeg", "-i", str(video_path), "-vf", f"fps={fps}", "-q:v", "2",
         str(frames_dir / "frame_%06d.jpg")],
        check=True, capture_output=True,
    )
    frames = sorted(frames_dir.glob("frame_*.jpg"))
    print(f"  Extracted {len(frames)} frames")
    return frames


def run_inference(
    detector: PoseDetector,
    frames: list[Path],
) -> list[FrameResult]:
    results: list[FrameResult] = []

    for idx, frame_path in enumerate(frames):
        img_bgr = cv2.imread(str(frame_path))
        if img_bgr is None:
            print(f"  WARN: could not read {frame_path}")
            continue

        t0 = time.perf_counter()
        detections = detector.detect(img_bgr)
        latency = (time.perf_counter() - t0) * 1000

        activity_counts = Counter(d.activity for d in detections)
        timestamp_s = float(idx)  # 1fps → frame index = seconds

        results.append(FrameResult(
            timestamp_s=timestamp_s,
            person_count=len(detections),
            activities=dict(activity_counts),
            detections=detections,
        ))

        acts_str = ", ".join(f"{k}:{v}" for k, v in sorted(activity_counts.items()))
        print(f"  Frame {idx:3d} | t={timestamp_s:6.1f}s | persons={len(detections):2d} "
              f"| {acts_str} | {latency:.0f}ms")

    return results


def save_keyframes(
    results: list[FrameResult],
    frames: list[Path],
    output_dir: Path,
    num_keyframes: int = 3,
) -> list[Path]:
    """Select and save annotated keyframes (top N by person count)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    ranked = sorted(enumerate(results), key=lambda x: x[1].person_count, reverse=True)
    selected_indices: list[int] = []
    min_spacing = max(1, len(results) // (num_keyframes + 1))

    for idx, _ in ranked:
        if len(selected_indices) >= num_keyframes:
            break
        if all(abs(idx - s) >= min_spacing for s in selected_indices):
            selected_indices.append(idx)

    if len(selected_indices) < num_keyframes:
        for idx, _ in ranked:
            if idx not in selected_indices and len(selected_indices) < num_keyframes:
                selected_indices.append(idx)

    saved: list[Path] = []
    for rank, idx in enumerate(sorted(selected_indices)):
        img_bgr = cv2.imread(str(frames[idx]))
        if img_bgr is None:
            continue
        annotated = draw_annotations(img_bgr, results[idx].detections)
        out_path = output_dir / f"keyframe_{rank:02d}_t{results[idx].timestamp_s:.0f}s.png"
        cv2.imwrite(str(out_path), annotated)
        saved.append(out_path)
        print(f"  Saved keyframe: {out_path.name} (t={results[idx].timestamp_s:.0f}s, "
              f"persons={results[idx].person_count})")

    return saved


def print_summary(results: list[FrameResult]) -> None:
    total_person_frames = sum(r.person_count for r in results)
    activity_person_frames: Counter[str] = Counter()
    for r in results:
        for act, cnt in r.activities.items():
            activity_person_frames[act] += cnt

    duration_s = len(results)  # 1fps
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Video duration (sampled):  {duration_s}s ({duration_s / 60:.1f} min)")
    print(f"  Frames analyzed:           {len(results)}")
    print(f"  Total person-frames:       {total_person_frames}")
    print()

    print("  Per-frame breakdown:")
    for r in results:
        acts = ", ".join(f"{k}:{v}" for k, v in sorted(r.activities.items())) or "none"
        print(f"    t={r.timestamp_s:6.1f}s  persons={r.person_count:2d}  [{acts}]")

    print()
    print("  Person-minutes per activity:")
    for act in ["sitting", "standing", "walking", "running", "unknown"]:
        pf = activity_person_frames.get(act, 0)
        pm = pf / 60.0  # person-frames at 1fps → person-minutes
        if pf > 0:
            pct = pf / total_person_frames * 100 if total_person_frames > 0 else 0
            print(f"    {act:<10s}  {pm:6.2f} person-min  ({pct:5.1f}%)")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="YOLO-pose video inference smoke test")
    parser.add_argument("--model", default="./yolo11n-pose.onnx", help="Path to pose ONNX model")
    parser.add_argument("--video-url", default=None, help="URL of sample video")
    parser.add_argument("--scene", default="park", choices=SAMPLE_VIDEOS.keys(),
                        help="Preset scene: walking, park, street")
    parser.add_argument("--fps", type=int, default=1, help="Frame extraction rate")
    args = parser.parse_args()

    base_dir = Path(__file__).parent
    video_path = base_dir / "sample_video.mp4"
    frames_dir = base_dir / "frames"
    output_dir = base_dir / "output"

    video_url = args.video_url or SAMPLE_VIDEOS[args.scene]
    print(f"[1/5] Download sample video (scene: {args.scene})")
    download_video(video_url, video_path)

    print("[2/5] Extract frames")
    frames = extract_frames(video_path, frames_dir, fps=args.fps)
    if not frames:
        print("ERROR: no frames extracted")
        sys.exit(1)

    print("[3/5] Load ONNX model (CUDA only)")
    detector = load_pose_model(args.model)

    print("[4/5] Run pose inference + activity classification")
    results = run_inference(detector, frames)

    print("[5/5] Save annotated keyframes")
    saved = save_keyframes(results, frames, output_dir, num_keyframes=3)

    print_summary(results)

    if saved:
        print(f"\nAnnotated keyframes saved to: {output_dir}/")


if __name__ == "__main__":
    main()
