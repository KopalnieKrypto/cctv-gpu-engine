#!/usr/bin/env python3
"""#124: how tall is a person, in pixels, inside the station rectangle?

## Why a ruler and not a model

`run_resolution_sweep.py` finds the detail scale at which the station head stops
working. That scale is a fraction, and a fraction is useless to an annotator
drawing a zone on a new camera. What they can act on is "a person is about N
pixels tall in this rectangle, and below M you are wasting your session".

This converts one into the other by measuring the only thing that can be measured
directly: the height of the person the zone is about, in native crop pixels.

**The pose detector is used here purely as a ruler.** It is not in the inference
path, it does not run at annotation time, and nothing downstream consumes its
output. `#124` permits exactly this and nothing more. The alternative was
hand-measuring a handful of frames, which is allowed too and is what the
`--report-only` output lets you sanity-check against.

## What it measures

One number per sampled crop: the height of the **tallest** detection above the
confidence floor. Tallest rather than all, because a station zone is about the
person working at the bench, and a partially visible passer-by at the rectangle's
edge is not the subject. Frames with no detection are counted and reported rather
than dropped silently, because an empty bench is a real state of this fixture
(`brak_na_stanowisku`) and a high no-detection rate with a low sample count would
mean the median rests on very little.

## Usage

    # inside the GPU container, on a fleet GPU
    python benchmarks/activity/tools/measure_person_height.py \
      --manifest benchmarks/activity/hala-prawe-v1/manifest.source.json \
      --crops-root benchmarks/activity/hala-prawe-v1/crops \
      --pose-model models/yolo11s-pose.onnx \
      --out runs/resolution/person-height.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

CONFIDENCE_FLOOR = 0.25  # the pipeline's own threshold, not a new one


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--crops-root", required=True, type=Path)
    ap.add_argument("--pose-model", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--every",
        type=int,
        default=10,
        help="sample every Nth crop; 10 gives about 300 frames over five windows",
    )
    args = ap.parse_args()

    import cv2

    from pipeline.pose_detector import load_pose_model

    manifest = json.loads(args.manifest.read_text())
    roi = manifest["station_roi"]["crop"]
    rect_h, rect_w = int(roi["h"]), int(roi["w"])

    detector = load_pose_model(str(args.pose_model))

    per_window: dict[str, dict] = {}
    all_heights: list[float] = []
    for clip in manifest["clips"]:
        if not clip.get("annotated"):
            continue
        slot = clip["slot"]
        files = sorted((args.crops_root / f"{slot}-native").glob("t*.jpg"))
        if not files:
            sys.exit(
                f"no native crops for {slot} in {args.crops_root}. Run "
                "extract_station_crops.py first - the 640px directories are the old "
                "rectangle and would measure the wrong thing."
            )
        sampled = files[:: args.every]
        heights: list[float] = []
        empty = 0
        for f in sampled:
            img = cv2.imread(str(f))
            if img is None:
                sys.exit(f"could not read {f}")
            if img.shape[0] != rect_h or img.shape[1] != rect_w:
                sys.exit(
                    f"{f} is {img.shape[1]}x{img.shape[0]} but the manifest rectangle "
                    f"is {rect_w}x{rect_h}. These crops predate the rectangle change; "
                    "re-cut them rather than measuring the wrong pixels."
                )
            dets = [d for d in detector.detect(img) if d.confidence >= CONFIDENCE_FLOOR]
            if not dets:
                empty += 1
                continue
            heights.append(max(d.bbox[3] - d.bbox[1] for d in dets))
        per_window[slot] = {
            "sampled": len(sampled),
            "with_person": len(heights),
            "no_detection": empty,
            "median_px": round(float(np.median(heights)), 1) if heights else None,
            "p10_px": round(float(np.percentile(heights, 10)), 1) if heights else None,
            "p90_px": round(float(np.percentile(heights, 90)), 1) if heights else None,
        }
        all_heights.extend(heights)
        print(f"{slot}: {per_window[slot]}", file=sys.stderr)

    if not all_heights:
        sys.exit("no person detected in any sampled crop - the ruler has nothing to read")

    median = float(np.median(all_heights))
    out = {
        "issue": 124,
        "purpose": "convert a detail scale into the unit an annotator can act on",
        "method": (
            "tallest detection above the pipeline's 0.25 confidence floor, per "
            f"sampled native crop, every {args.every}th crop of each annotated window"
        ),
        "pose_is_a_ruler_only": (
            "the detector is not in the inference path and does not run at "
            "annotation time; it is used here to measure, nothing else"
        ),
        "rectangle_native_px": {"w": rect_w, "h": rect_h},
        "frames_sampled": sum(w["sampled"] for w in per_window.values()),
        "frames_with_person": len(all_heights),
        "median_person_height_px": round(median, 1),
        "p10_person_height_px": round(float(np.percentile(all_heights, 10)), 1),
        "p90_person_height_px": round(float(np.percentile(all_heights, 90)), 1),
        "median_as_fraction_of_rect_height": round(median / rect_h, 3),
        "per_window": per_window,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nwrote {args.out}")
    print(f"\npass --person-height-px {median:.1f} to run_resolution_sweep.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
