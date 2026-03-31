"""
Video pose inference test — standalone script for ai-test VPS.
Downloads sample video, extracts frames at 1fps, runs YOLO-pose ONNX inference
(CUDAExecutionProvider only), classifies activities via keypoint heuristics,
outputs per-frame + total summary, saves 3 annotated keyframes.
"""

import argparse
import math
import os
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont

# ── Constants ──────────────────────────────────────────────────────────────────

IMG_SIZE = 640
CONF_THRESHOLD = 0.25
NMS_IOU_THRESHOLD = 0.45

# Activity thresholds (tune as needed)
KNEE_ANGLE_SIT = 120.0       # degrees — below this = bent knees
HIP_HEIGHT_RATIO_SIT = 0.40  # hip close to bottom of bbox
STRIDE_RATIO_WALK = 1.3
STRIDE_RATIO_RUN = 2.0
TORSO_LEAN_RUN = 15.0        # degrees from vertical

# COCO 17 keypoint indices
KP_L_SHOULDER, KP_R_SHOULDER = 5, 6
KP_L_HIP, KP_R_HIP = 11, 12
KP_L_KNEE, KP_R_KNEE = 13, 14
KP_L_ANKLE, KP_R_ANKLE = 15, 16

# Skeleton connections for drawing
SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),        # head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), # torso+arms
    (5, 11), (6, 12), (11, 12),              # torso
    (11, 13), (13, 15), (12, 14), (14, 16),  # legs
]

ACTIVITY_COLORS = {
    "SITTING": (0, 200, 0),
    "STANDING": (0, 120, 255),
    "WALKING": (255, 165, 0),
    "RUNNING": (255, 0, 0),
    "UNKNOWN": (128, 128, 128),
}

# Sample videos for different activity testing (Mixkit, free, no watermark)
SAMPLE_VIDEOS = {
    "walking": "https://assets.mixkit.co/videos/34563/34563-720.mp4",     # people walking on sidewalk
    "park": "https://assets.mixkit.co/videos/4401/4401-720.mp4",          # park scene, sitting+walking
    "street": "https://assets.mixkit.co/videos/4048/4048-720.mp4",        # busy street, walking+standing
}
SAMPLE_VIDEO_URL = SAMPLE_VIDEOS["park"]  # default: mixed activities


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class Keypoint:
    x: float
    y: float
    vis: float  # visibility/confidence


@dataclass
class Detection:
    bbox: list[float]          # [x1, y1, x2, y2] in original coords
    confidence: float
    keypoints: list[Keypoint]  # 17 COCO keypoints
    activity: str = "UNKNOWN"


@dataclass
class FrameResult:
    timestamp_s: float
    person_count: int
    activities: dict[str, int] = field(default_factory=dict)
    detections: list[Detection] = field(default_factory=list)


# ── Preprocessing (matches yolo-serve/app.py) ─────────────────────────────────

def preprocess(img_bgr: np.ndarray) -> tuple[np.ndarray, int, int]:
    orig_h, orig_w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    arr = img_resized.astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)   # HWC → CHW
    arr = np.expand_dims(arr, 0)   # [1, 3, 640, 640]
    return arr, orig_w, orig_h


# ── Postprocessing ─────────────────────────────────────────────────────────────

def iou(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms(detections: list[Detection]) -> list[Detection]:
    dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
    kept: list[Detection] = []
    suppressed: set[int] = set()
    for i, di in enumerate(dets):
        if i in suppressed:
            continue
        kept.append(di)
        for j in range(i + 1, len(dets)):
            if j not in suppressed and iou(di.bbox, dets[j].bbox) > NMS_IOU_THRESHOLD:
                suppressed.add(j)
    return kept


def postprocess(output: np.ndarray, orig_w: int, orig_h: int) -> list[Detection]:
    """
    YOLO-pose output: [1, 56, num_boxes]
    Rows 0-3: cx, cy, w, h (in 640x640 space)
    Row 4: confidence
    Rows 5-55: 17 keypoints × 3 (x, y, visibility)
    """
    data = output[0]  # [56, N]
    num_boxes = data.shape[1]
    sx = orig_w / IMG_SIZE
    sy = orig_h / IMG_SIZE

    detections: list[Detection] = []
    for i in range(num_boxes):
        conf = float(data[4, i])
        if conf < CONF_THRESHOLD:
            continue

        cx, cy, w, h = data[0, i], data[1, i], data[2, i], data[3, i]
        x1 = (cx - w / 2) * sx
        y1 = (cy - h / 2) * sy
        x2 = (cx + w / 2) * sx
        y2 = (cy + h / 2) * sy

        kps: list[Keypoint] = []
        for k in range(17):
            base = 5 + k * 3
            kx = float(data[base, i]) * sx
            ky = float(data[base + 1, i]) * sy
            kv = float(data[base + 2, i])
            kps.append(Keypoint(kx, ky, kv))

        detections.append(Detection(
            bbox=[x1, y1, x2, y2],
            confidence=conf,
            keypoints=kps,
        ))

    return nms(detections)


# ── Activity classification ───────────────────────────────────────────────────

def _angle_deg(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    """Angle at point b formed by segments ba and bc, in degrees."""
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2)
    mag_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2)
    if mag_ba < 1e-6 or mag_bc < 1e-6:
        return 180.0
    cos_val = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cos_val))


def _kp_visible(kps: list[Keypoint], *indices: int, threshold: float = 0.5) -> bool:
    return all(kps[i].vis > threshold for i in indices)


def classify_activity(det: Detection) -> str:
    kps = det.keypoints
    x1, y1, x2, y2 = det.bbox
    bbox_h = y2 - y1

    # need lower body keypoints
    lower_body_ids = (KP_L_HIP, KP_R_HIP, KP_L_KNEE, KP_R_KNEE, KP_L_ANKLE, KP_R_ANKLE)
    if not _kp_visible(kps, *lower_body_ids):
        # fallback: bbox aspect ratio
        bbox_w = x2 - x1
        if bbox_w < 1e-6:
            return "UNKNOWN"
        return "SITTING" if bbox_h / bbox_w < 1.5 else "STANDING"

    # knee angle (average both legs)
    angle_l = _angle_deg(
        (kps[KP_L_HIP].x, kps[KP_L_HIP].y),
        (kps[KP_L_KNEE].x, kps[KP_L_KNEE].y),
        (kps[KP_L_ANKLE].x, kps[KP_L_ANKLE].y),
    )
    angle_r = _angle_deg(
        (kps[KP_R_HIP].x, kps[KP_R_HIP].y),
        (kps[KP_R_KNEE].x, kps[KP_R_KNEE].y),
        (kps[KP_R_ANKLE].x, kps[KP_R_ANKLE].y),
    )
    knee_angle = (angle_l + angle_r) / 2.0

    # hip height ratio
    avg_hip_y = (kps[KP_L_HIP].y + kps[KP_R_HIP].y) / 2.0
    hip_height_ratio = (y2 - avg_hip_y) / bbox_h if bbox_h > 1e-6 else 0.5

    if knee_angle < KNEE_ANGLE_SIT and hip_height_ratio < HIP_HEIGHT_RATIO_SIT:
        return "SITTING"

    # stride ratio
    hip_dx = abs(kps[KP_L_HIP].x - kps[KP_R_HIP].x)
    ankle_dx = abs(kps[KP_L_ANKLE].x - kps[KP_R_ANKLE].x)
    stride_ratio = ankle_dx / hip_dx if hip_dx > 1e-6 else 0.0

    # torso lean (angle from vertical)
    mid_shoulder = ((kps[KP_L_SHOULDER].x + kps[KP_R_SHOULDER].x) / 2,
                    (kps[KP_L_SHOULDER].y + kps[KP_R_SHOULDER].y) / 2)
    mid_hip = ((kps[KP_L_HIP].x + kps[KP_R_HIP].x) / 2,
               (kps[KP_L_HIP].y + kps[KP_R_HIP].y) / 2)
    torso_dx = mid_shoulder[0] - mid_hip[0]
    torso_dy = mid_shoulder[1] - mid_hip[1]
    torso_len = math.sqrt(torso_dx ** 2 + torso_dy ** 2)
    if torso_len > 1e-6:
        # angle from vertical (vertical = (0, -1) since y increases downward)
        cos_lean = abs(torso_dy) / torso_len
        torso_lean = math.degrees(math.acos(max(-1.0, min(1.0, cos_lean))))
    else:
        torso_lean = 0.0

    if stride_ratio > STRIDE_RATIO_RUN and torso_lean > TORSO_LEAN_RUN:
        return "RUNNING"
    if stride_ratio > STRIDE_RATIO_WALK:
        return "WALKING"
    return "STANDING"


# ── Drawing / annotation ──────────────────────────────────────────────────────

def draw_annotations(img_bgr: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Draw bboxes, skeleton, and activity labels on image."""
    img = img_bgr.copy()
    for det in detections:
        color = ACTIVITY_COLORS.get(det.activity, (128, 128, 128))
        x1, y1, x2, y2 = [int(v) for v in det.bbox]

        # bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # label
        label = f"{det.activity} {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # skeleton edges
        kps = det.keypoints
        for (a, b) in SKELETON_EDGES:
            if kps[a].vis > 0.5 and kps[b].vis > 0.5:
                pt1 = (int(kps[a].x), int(kps[a].y))
                pt2 = (int(kps[b].x), int(kps[b].y))
                cv2.line(img, pt1, pt2, color, 2)

        # keypoint dots
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


def load_onnx_session(model_path: str) -> ort.InferenceSession:
    providers = ort.get_available_providers()
    print(f"  Available ONNX providers: {providers}")
    if "CUDAExecutionProvider" not in providers:
        raise RuntimeError("CUDAExecutionProvider not available — GPU required, no CPU fallback")
    session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider"])
    active = session.get_providers()
    print(f"  Active providers: {active}")
    if "CUDAExecutionProvider" not in active:
        raise RuntimeError("CUDAExecutionProvider failed to initialize")
    return session


def run_inference(
    session: ort.InferenceSession,
    frames: list[Path],
) -> list[FrameResult]:
    input_name = session.get_inputs()[0].name
    results: list[FrameResult] = []

    for idx, frame_path in enumerate(frames):
        img_bgr = cv2.imread(str(frame_path))
        if img_bgr is None:
            print(f"  WARN: could not read {frame_path}")
            continue

        tensor, orig_w, orig_h = preprocess(img_bgr)
        t0 = time.perf_counter()
        output = session.run(None, {input_name: tensor})
        latency = (time.perf_counter() - t0) * 1000

        detections = postprocess(output[0], orig_w, orig_h)

        # classify each person
        for det in detections:
            det.activity = classify_activity(det)

        activity_counts = Counter(d.activity for d in detections)
        timestamp_s = float(idx)  # 1fps → frame index = seconds

        fr = FrameResult(
            timestamp_s=timestamp_s,
            person_count=len(detections),
            activities=dict(activity_counts),
            detections=detections,
        )
        results.append(fr)

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

    # sort by person count desc, pick top N with some spacing
    ranked = sorted(enumerate(results), key=lambda x: x[1].person_count, reverse=True)
    selected_indices: list[int] = []
    min_spacing = max(1, len(results) // (num_keyframes + 1))

    for idx, _ in ranked:
        if len(selected_indices) >= num_keyframes:
            break
        if all(abs(idx - s) >= min_spacing for s in selected_indices):
            selected_indices.append(idx)

    # if we didn't get enough, fill without spacing constraint
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
    for act in ["SITTING", "STANDING", "WALKING", "RUNNING", "UNKNOWN"]:
        pf = activity_person_frames.get(act, 0)
        pm = pf / 60.0  # person-frames at 1fps → person-minutes
        if pf > 0:
            pct = pf / total_person_frames * 100 if total_person_frames > 0 else 0
            print(f"    {act:<10s}  {pm:6.2f} person-min  ({pct:5.1f}%)")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="YOLO-pose video inference test")
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
    session = load_onnx_session(args.model)

    print("[4/5] Run pose inference + activity classification")
    results = run_inference(session, frames)

    print("[5/5] Save annotated keyframes")
    saved = save_keyframes(results, frames, output_dir, num_keyframes=3)

    print_summary(results)

    if saved:
        print(f"\nAnnotated keyframes saved to: {output_dir}/")


if __name__ == "__main__":
    main()
