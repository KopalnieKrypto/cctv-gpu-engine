#!/usr/bin/env python3
"""Prove what the station head is actually fed, on both paths, in pixels.

## Why this exists as a tool and not a one-off check

The station head was fitted on the middle 43.2% of the rectangle every document
said it measured, for the whole life of v1.0.0, and nothing caught it. Not the
tests, not the card, not `coverage`. It was invisible because the two
preprocessing paths - `pipeline.station_classifier.preprocess_crop` for a
production run, and `run_pixel_probe_arm.embed_windows` for training - are
separate implementations of one convention, and both had the same unintended
centre-crop. Reading either one in isolation confirms nothing about the other.

So this does not read code. It runs both paths on the same frame, writes out the
image each one hands the backbone, and compares them numerically. A crop, a
resize, an aspect squeeze or a channel swap anywhere in either path shows up as a
picture you can look at and a number that is not zero.

## Usage

    python benchmarks/activity/tools/verify_model_input.py \
      --video test-data/clip.mp4 --at 30 --out-dir /tmp/verify

Writes `native.png` (the rectangle as cropped), `analysis.png` and
`training.png` (what each path feeds the model, de-normalised back to viewable
pixels), and prints the agreement between them.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from pipeline.station_classifier import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    model_input_for_rect,
    preprocess_crop,
    station_crop,
)

DEFAULT_MANIFEST = Path("benchmarks/activity/hala-prawe-v1/manifest.source.json")


def _native_frame(video: Path, at: float) -> np.ndarray:
    """One frame at `at` seconds, full resolution, BGR — as the pipeline decodes."""
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0:s=x",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    width, height = (int(v) for v in probe.split("x"))
    raw = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            str(at),
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-pix_fmt",
            "bgr24",
            "-f",
            "rawvideo",
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)


def _denormalise(tensor: np.ndarray) -> np.ndarray:
    """A `(1, 3, h, w)` normalised tensor back to viewable RGB pixels."""
    pixels = tensor[0].transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(pixels * 255.0, 0, 255).astype(np.uint8)


def _training_tensor(crop_bgr: np.ndarray, model_input: tuple[int, int]) -> np.ndarray:
    """What the TRAINING path feeds the backbone, through the real processor.

    Reproduced end to end rather than approximated: the crop is JPEG-encoded by
    ffmpeg at `-q:v 2` first, because `extract_native_crops` writes JPEGs and the
    embedder reads them back, and then handed to the actual
    `AutoImageProcessor`. Skipping either step would verify a pipeline nobody
    runs.
    """
    from PIL import Image
    from run_pixel_probe_arm import BACKBONE
    from transformers import AutoImageProcessor

    with tempfile.TemporaryDirectory() as tmp:
        # Lossless on the way in, so the only compression in this path is the one
        # `extract_native_crops` actually applies: ffmpeg's own JPEG at -q:v 2.
        lossless = Path(tmp) / "crop.png"
        jpeg = Path(tmp) / "crop.jpg"
        Image.fromarray(crop_bgr[:, :, ::-1]).save(lossless)
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(lossless), "-q:v", "2", str(jpeg)],
            check=True,
        )
        image = Image.open(jpeg).convert("RGB")
        processor = AutoImageProcessor.from_pretrained(BACKBONE)
        out = processor(
            images=[image],
            return_tensors="np",
            size={"height": model_input[0], "width": model_input[1]},
            do_center_crop=False,
        )
    return np.asarray(out["pixel_values"], dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--at", type=float, default=0.0, help="seconds into the clip")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--image-size", type=int, default=420, help="target tensor height")
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    from PIL import Image

    roi = json.loads(args.manifest.read_text())["station_roi"]["crop"]
    rect = (int(roi["x"]), int(roi["y"]), int(roi["w"]), int(roi["h"]))
    model_input = model_input_for_rect(rect, args.image_size)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frame = _native_frame(args.video, args.at)
    crop = station_crop(frame, rect)
    Image.fromarray(crop[:, :, ::-1]).save(args.out_dir / "native.png")

    analysis = preprocess_crop(crop, model_input)
    training = _training_tensor(crop, model_input)
    Image.fromarray(_denormalise(analysis)).save(args.out_dir / "analysis.png")
    Image.fromarray(_denormalise(training)).save(args.out_dir / "training.png")

    w = sys.stdout.write
    w(f"native frame       {frame.shape[1]}x{frame.shape[0]}\n")
    w(f"station rectangle  {rect[2]}x{rect[3]} at ({rect[0]}, {rect[1]})\n")
    w(f"crop delivered     {crop.shape[1]}x{crop.shape[0]}\n")
    w(f"tensor expected    {model_input[1]}x{model_input[0]} (width x height)\n")
    w(f"analysis path      {analysis.shape}\n")
    w(f"training path      {training.shape}\n")

    ok = True
    if crop.shape[:2] != (rect[3], rect[2]):
        w("FAIL: the crop is not the rectangle\n")
        ok = False
    if analysis.shape[-2:] != model_input or training.shape[-2:] != model_input:
        w("FAIL: a path does not produce the expected tensor size\n")
        ok = False
    if analysis.shape != training.shape:
        w("FAIL: the two paths disagree on tensor shape\n")
        ok = False
    else:
        # JPEG re-encoding and PIL-vs-processor resampling differ slightly; a
        # centre-crop or an aspect change would not be slight.
        diff = float(np.abs(analysis - training).max())
        mean = float(np.abs(analysis - training).mean())
        w(f"agreement          max |diff| {diff:.4f}, mean {mean:.4f} (normalised units)\n")
        if mean > 0.15:
            w("FAIL: the paths disagree by more than encoding noise\n")
            ok = False

    w(f"\nwrote {args.out_dir}/native.png, analysis.png, training.png\n")
    w("OK - both paths feed the whole rectangle\n" if ok else "\nNOT VERIFIED\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
