#!/usr/bin/env python3
"""Test Qwen2.5-VL-3B activity classification on a surveillance video.

Extracts frames at 1fps, sends each to Qwen2.5-VL-3B-Instruct with a
structured classification prompt, and prints per-frame results + totals.

Usage:
    python scripts/test_vlm_classifier.py /path/to/video.mp4
"""

from __future__ import annotations

import sys
import time

import cv2
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

ACTIVITIES = ("sitting", "standing", "walking", "running")

PROMPT = (
    "Look at this surveillance camera frame. A person is visible. "
    "Classify their current activity as exactly one of: sitting, standing, walking, running. "
    "Reply with only one word — the activity label, nothing else."
)

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


def extract_frames(video_path: str, fps: int = 1):
    """Yield (timestamp_s, PIL.Image) from video at given fps."""
    from PIL import Image

    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        raise RuntimeError(f"Cannot read video FPS from {video_path}")
    frame_interval = int(round(video_fps / fps))
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            ts = frame_idx / video_fps
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            yield ts, img
        frame_idx += 1
    cap.release()


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <video.mp4>", file=sys.stderr)
        sys.exit(1)

    video_path = sys.argv[1]
    print(f"Loading model {MODEL_ID}...", file=sys.stderr)
    t0 = time.time()

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        min_pixels=256 * 28 * 28,
        max_pixels=512 * 28 * 28,
    )
    print(f"Model loaded in {time.time() - t0:.1f}s", file=sys.stderr)

    counts: dict[str, int] = dict.fromkeys(ACTIVITIES, 0)
    unknown_count = 0
    total_frames = 0
    total_inference = 0.0

    for ts, img in extract_frames(video_path, fps=1):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text],
            images=[img],
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        t1 = time.time()
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        elapsed = time.time() - t1
        total_inference += elapsed

        generated = output_ids[0][inputs.input_ids.shape[1] :]
        answer = processor.decode(generated, skip_special_tokens=True).strip().lower()

        # Parse activity from answer
        activity = "unknown"
        for act in ACTIVITIES:
            if act in answer:
                activity = act
                break

        if activity in counts:
            counts[activity] += 1
        else:
            unknown_count += 1

        total_frames += 1
        print(f"t={ts:5.0f}s  answer={answer:20s}  activity={activity:10s}  ({elapsed:.2f}s)")

    print(file=sys.stderr)
    print("=== TOTALS ===", file=sys.stderr)
    for act, cnt in counts.items():
        print(f"  {act:10s}: {cnt:3d}s", file=sys.stderr)
    print(f"  unknown   : {unknown_count:3d}", file=sys.stderr)
    print(f"  total det : {total_frames:3d} frames", file=sys.stderr)
    avg = total_inference / max(total_frames, 1)
    print(f"  inference : {total_inference:.1f}s total, {avg:.2f}s/frame", file=sys.stderr)


if __name__ == "__main__":
    main()
