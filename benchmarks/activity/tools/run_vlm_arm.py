"""Run the VLM spike arm over a `hala-prawe-v1` window and emit a prediction file.

The first of the three C.0 arms in #117: Qwen2.5-VL-3B prompted per station over
the seven-activity process vocabulary. Output drops straight into
`evaluate_arms.py`, so this script never scores anything itself - it produces
predictions and an honest cost measurement, and the harness decides.

## It samples the same grid the annotator saw

Crops come out of the NATIVE 3840x2160 frame at the manifest's `station_roi`,
at the annotation's own stride, indexed by `pts_time`. That is not a detail: a
prediction on a different grid, or on the 1280x736 downscale, is not comparable
to the ground truth and would quietly measure something else. W1's container
reports `r_frame_rate=120/1` against a true ~20 fps, so frame-indexed sampling
mis-maps every W1 timestamp - ffmpeg's `fps` filter is PTS-derived and correct.

## The prompt is written from the vocabulary, not tuned on a window

Zero-shot, and deliberately so. The prompt is derived from the activity
definitions in `manifest.source.json` and the two labelling rules in
`METHODOLOGY.md`, without looking at either window's results. Both windows are
therefore clean held-out material and the union figure is honest.

If you DO tune the prompt against a window, you must record it: pass
`--tuned-on W1`, which stamps the prediction file so the harness and the report
can mark that window's figures in-sample. An untracked prompt edit is
indistinguishable from training on the test set.

## Cost is measured, not estimated

Peak VRAM is sampled from `nvidia-smi` for this PID (the same method as #86,
which is what the 7 866 MiB figure in CLAUDE.md came from), because
`torch.cuda.max_memory_allocated` ignores the CUDA context and the allocator's
reserve and so under-reports against the 12 GB single-card budget.

## Usage

    # on cctv-vps GPU 1, or cctv-vps-2
    CUDA_VISIBLE_DEVICES=1 uv run benchmarks/activity/tools/run_vlm_arm.py \
      --manifest benchmarks/activity/hala-prawe-v1/manifest.source.json \
      --slot W2 --box cctv-vps --gpu-index 1 \
      --out runs/vlm/W2.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

# Written from manifest.source.json -> activities and the two labelling rules in
# METHODOLOGY.md ("postoj requires all three conditions", "inna_czynnosc is real
# work"). Never adjusted against a window's results - see the module docstring.
PROMPT = """You are watching a fixed camera view of a single welding bench in a \
rebar fabrication hall. One frame is shown. Decide what is happening AT THIS BENCH \
right now, and answer with exactly one label from this list:

spawanie - a worker is welding: arc visible, sparks, or holding the torch to the \
workpiece with the helmet down.
ukladanie_pretow - a worker is placing, positioning or arranging steel rods or bars \
in the jig.
sciaganie_elementu - a worker is removing or lifting a finished element off the bench.
inna_czynnosc - a worker is doing other real work: carrying wire, grinding, measuring, \
adjusting equipment, cleaning.
postoj - a worker is present but idle: standing still, nothing in the hands, not \
working. If they are holding a rod or a tool, this is NOT postoj.
brak_na_stanowisku - no worker is at the bench.
nierozpoznane - the frame genuinely does not show enough to tell.

Answer with the label only, no explanation."""

LABELS = [
    "spawanie",
    "ukladanie_pretow",
    "sciaganie_elementu",
    "inna_czynnosc",
    "postoj",
    "brak_na_stanowisku",
    "nierozpoznane",
]

# Long edge the ROI crop is scaled to before the model sees it. Matches the
# annotation review size, so the model is given neither more nor less than the
# human had.
VIEW_LONG_EDGE = 640


def extract_crops(clip: Path, roi: dict, stride: int, out_dir: Path) -> list[Path]:
    """ROI crops from the native frame at `stride` seconds of PTS."""
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not on PATH")
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("t*.jpg"):
        stale.unlink()
    crop = f"crop={roi['w']}:{roi['h']}:{roi['x']}:{roi['y']}"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(clip),
            "-vf",
            f"fps=1/{stride},{crop},scale={VIEW_LONG_EDGE}:-1",
            "-q:v",
            "3",
            str(out_dir / "t%05d.jpg"),
        ],
        check=True,
    )
    return sorted(out_dir.glob("t*.jpg"))


class VramSampler:
    """Peak VRAM for this PID, sampled from nvidia-smi (the #86 method)."""

    def __init__(self, interval: float = 0.5) -> None:
        self.peak_mib = 0
        self._stop = threading.Event()
        self._interval = interval
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        pid = str(os.getpid())
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=pid,used_memory",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                for line in out.stdout.splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) == 2 and parts[0] == pid:
                        self.peak_mib = max(self.peak_mib, int(parts[1]))
            except Exception:  # nvidia-smi absent or transient - cost stays UNMEASURED
                pass
            self._stop.wait(self._interval)

    def __enter__(self) -> VramSampler:
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)


def classify(crops: list[Path], stride: int) -> tuple[list[dict], float]:
    """Label every crop. Returns samples and the seconds spent loading the model."""
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    load_start = time.monotonic()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype=torch.bfloat16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        min_pixels=256 * 28 * 28,
        max_pixels=512 * 28 * 28,
    )
    load_seconds = time.monotonic() - load_start

    samples: list[dict] = []
    for i, path in enumerate(crops):
        img = Image.open(path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": img}, {"type": "text", "text": PROMPT}],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[img], padding=True, return_tensors="pt").to(
            model.device
        )
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=12, do_sample=False)
        answer = (
            processor.decode(out[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
            .strip()
            .lower()
        )

        # Longest label first, so `inna_czynnosc` is not shadowed by a substring.
        label = next((v for v in sorted(LABELS, key=len, reverse=True) if v in answer), None)
        if label is None:
            # An unparseable answer is `nierozpoznane` - the model declining to
            # commit, which is a real outcome. It is NOT silently dropped: the
            # harness counts an unanswered sample as an error either way.
            label = "nierozpoznane"
        samples.append({"t_s": i * stride, "activity_id": label})

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(crops)} crops", file=sys.stderr)

    return samples, load_seconds


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--slot", required=True, help="clip slot, e.g. W2")
    ap.add_argument("--clip", type=Path, help="override the clip path")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--box", required=True, help="fleet box this ran on, e.g. cctv-vps")
    ap.add_argument("--gpu-index", type=int, help="GPU index used on that box")
    ap.add_argument(
        "--tuned-on",
        help="window the prompt was tuned against, if any - stamps the run as in-sample",
    )
    ap.add_argument("--arm", default="vlm-qwen2.5-vl-3b")
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text())
    fixture_dir = args.manifest.parent
    clip_meta = next((c for c in manifest["clips"] if c["slot"] == args.slot), None)
    if clip_meta is None:
        sys.exit(f"no clip {args.slot} in the manifest")
    roi = manifest["station_roi"]["crop"]

    truth_path = fixture_dir / clip_meta["annotation_file"]
    truth = json.loads(truth_path.read_text())
    stride = int(truth["stride_s"])

    clip = args.clip or next(fixture_dir.glob(f"{args.slot}-*.mp4"), None)
    if clip is None or not clip.exists():
        sys.exit(f"clip for {args.slot} not found in {fixture_dir} - see r2_key_preserved")

    print(f"extracting ROI crops from {clip.name} at {stride}s stride ...", file=sys.stderr)
    crops = extract_crops(clip, roi, stride, fixture_dir / "crops" / f"{args.slot}-arm")
    print(f"{len(crops)} crops", file=sys.stderr)

    run_start = time.monotonic()
    with VramSampler() as vram:
        samples, load_seconds = classify(crops, stride)
    wall_seconds = time.monotonic() - run_start

    doc = {
        "arm": args.arm,
        "window": args.slot,
        "model": "Qwen/Qwen2.5-VL-3B-Instruct",
        "prompt_tuning": args.tuned_on or "none - written from the vocabulary definition",
        "in_sample": bool(args.tuned_on and args.tuned_on == args.slot),
        "samples": samples,
        "gpu": {
            "box": args.box,
            "gpu_index": args.gpu_index,
            "gpus_used": 1,
            "gpu_seconds": round(wall_seconds, 1),
            "load_seconds": round(load_seconds, 1),
            "video_seconds": truth["duration_s"],
            "peak_vram_mib": vram.peak_mib or None,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2, ensure_ascii=False))

    per_hour = wall_seconds / truth["duration_s"] * 3600
    print(f"\nwrote:      {args.out}")
    print(f"cost:       {per_hour:.0f} GPU-seconds per video-hour on {args.box}")
    print(f"peak VRAM:  {vram.peak_mib or 'UNMEASURED'} MiB")
    print(f"model load: {load_seconds:.1f}s of that")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
