# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "opencv-python-headless"]
# ///
"""Annotation aid for the magazyn-hall-v1 fixture.

The camera is static, so a per-window temporal MEDIAN is a clean background
plate: anything that moved during the 60-second window differs from it. This
localizes candidate people so the annotator magnifies only where something
changed, instead of scanning 8.3 megapixels per frame by eye.

DELIBERATELY NOT A PERSON DETECTOR. It reports "this changed", nothing more,
and it is not any candidate arm - so using it cannot make an arm its own ground
truth (the circularity `bending-pilot-v1`'s METHODOLOGY.md guards against).

Its blind spot must be stated wherever its output is used: a person who holds
still for the whole window becomes part of the median and produces NO signal.
So it is a "did I miss anyone?" cross-check on a second pass, never the primary
source, and every frame still needs eye review at full resolution.

MEASURED, not theorized. Scored against the six hand-annotated people in
window-1-frame-001:

    red shirt (mid)     54% covered
    blue shirt (mid)    96%
    dark blue (near)    99%
    far standing         0%   <-- all three far-field workers
    far green top        0%
    far bent over        0%

It covers the mid and near field usefully and misses the ENTIRE far field. Two
structural causes: those workers stand still at the racks (so they are the
median), and CLOCK_REGION below overlaps them, because the burned-in timestamp
is drawn on top of that part of the hall.

=> USE FOR MID/NEAR FIELD ONLY. Never conclude "the aid found nothing there, so
nobody is there" - in the far field that inference is exactly backwards. The far
field requires the magnified-crop pass regardless of what this reports.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

# A person at the far wall is ~31x84 px (measured on window-1-frame-001), so the
# floor sits well under that to avoid deleting the very population this fixture
# exists to capture. Smaller blobs are overwhelmingly JPEG noise and rebar glint.
MIN_BLOB_AREA_PX = 800
# Absolute per-pixel difference that counts as "changed". The first pass used 18
# and returned 42-134 regions per frame - the hall is wall-to-wall rebar, mesh
# and wire spools, so sub-pixel camera drift lights up every high-frequency edge
# in the scene. Raised well above that noise floor.
DIFF_THRESHOLD = 40
# Person-plausible geometry, from boxes measured on window-1-frame-001: the
# far-field cluster runs ~31x84 px, the nearest worker ~154x292. Edge-drift
# artifacts are long thin slivers along rebar, so an aspect gate removes most of
# them without touching anything person-shaped.
MIN_W, MAX_W = 20, 320
MIN_H, MAX_H = 55, 460
MIN_ASPECT, MAX_ASPECT = 1.0, 5.0
# The burned-in clock changes every second, so it differs from the median in
# every single frame. Excluded explicitly rather than left to pollute the output.
CLOCK_REGION = (0, 0, 720, 150)  # x, y, w, h


def window_frames(frames_dir: Path, window: str) -> list[Path]:
    return sorted(frames_dir.glob(f"{window}-frame-*.jpg"))


def build_background(paths: list[Path]) -> np.ndarray:
    """Per-pixel temporal median across the window - the empty-hall plate."""
    stack = np.stack([cv2.imread(str(p)) for p in paths], axis=0)
    return np.median(stack, axis=0).astype(np.uint8)


def candidate_boxes(frame: np.ndarray, background: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Axis-aligned boxes around regions that differ from the background plate."""
    diff = cv2.absdiff(frame, background)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (11, 11), 0)
    cx, cy, cw, ch = CLOCK_REGION
    gray[cy : cy + ch, cx : cx + cw] = 0
    _, mask = cv2.threshold(gray, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
    # Close gaps so a torso and legs split by a rebar rack read as one region.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        if cv2.contourArea(contour) < MIN_BLOB_AREA_PX:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if not (MIN_W <= w <= MAX_W and MIN_H <= h <= MAX_H):
            continue
        if not (MIN_ASPECT <= h / w <= MAX_ASPECT):
            continue
        boxes.append((int(x), int(y), int(w), int(h)))
    return sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)


def main() -> None:
    fixture = Path(sys.argv[1])
    frames_dir = fixture / "frames"
    out_dir = fixture / "review" / "motion"
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, list[dict[str, object]]] = {}
    for window in ("window-1", "window-2", "window-3"):
        paths = window_frames(frames_dir, window)
        if not paths:
            raise SystemExit(f"no frames for {window} in {frames_dir}")
        background = build_background(paths)
        cv2.imwrite(str(out_dir / f"{window}-background.jpg"), background)

        rows = []
        thumbs = []
        for path in paths:
            frame = cv2.imread(str(path))
            boxes = candidate_boxes(frame, background)
            overlay = frame.copy()
            for x, y, w, h in boxes:
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 200, 255), 6)
            cv2.imwrite(str(out_dir / f"{path.stem}-motion.jpg"), overlay)
            thumbs.append(cv2.resize(overlay, (640, 360)))
            rows.append({"frame": path.stem, "candidates": len(boxes), "boxes_xywh": boxes})

        # 4x5 contact sheet: the whole window's movement pattern in one image.
        sheet = np.vstack([np.hstack(thumbs[i : i + 4]) for i in range(0, 20, 4)])
        cv2.imwrite(str(out_dir / f"{window}-contact-sheet.jpg"), sheet)
        report[window] = rows
        counts = [r["candidates"] for r in rows]
        print(
            f"{window}: {sum(counts)} candidate regions across 20 frames "
            f"(min {min(counts)}, max {max(counts)})"
        )

    (out_dir / "motion-candidates.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
