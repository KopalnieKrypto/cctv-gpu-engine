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

## Prompts are versioned, and their provenance is recorded

`--prompt-version` selects one; the arm name carries it, so two versions never
land in the same bucket in a report.

- **v1** was written from the activity definitions in `manifest.source.json` and
  METHODOLOGY's two labelling rules, without looking at either window. Both
  windows were therefore clean, and it failed on both.
- **v2** was written after reading v1's confusion matrix on **W1**, so it is
  tuned on W1. Run it as `--prompt-version v2 --tuned-on W1`: W1's figures are
  then in-sample and only **W2 is a clean measurement of v2**.

An untracked prompt edit is indistinguishable from training on the test set, so
the stamp is not optional book-keeping. If v2 gets iterated again against W2,
there is no clean window left in this fixture and the arm's figure stops meaning
anything - stop and say so rather than quietly reporting the better number.

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

# v1: written from manifest.source.json -> activities and METHODOLOGY's two
# labelling rules, never adjusted against a window. Measured 2026-09-01 and it
# fails: the model answers `spawanie` to almost everything and never once emits
# `brak_na_stanowisku`, including on frames where the bench is verifiably empty.
# Kept so that run is reproducible, and because v2 only means something next to it.
PROMPT_V1 = """You are watching a fixed camera view of a single welding bench in a \
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

# v2: TUNED ON W1 ONLY (--tuned-on W1), so W1's figures are in-sample and W2 stays
# clean held-out. Three changes, each aimed at a specific v1 failure:
#
#   1. An ordered decision procedure instead of a flat list. v1 let the model
#      reach for a plausible activity without first asking whether anyone is
#      there; the empty bench at t=540s came back `inna_czynnosc`.
#   2. "Is anyone at the bench?" is forced FIRST and given its own hard rule,
#      because `brak_na_stanowisku` was never emitted once in 1199 samples.
#   3. `spawanie` now requires positive visual evidence (arc glare, sparks, or
#      helmet down with the torch on the work). v1 called 333 of 359
#      `ukladanie_pretow` samples `spawanie`.
PROMPT_V2 = """A fixed overhead camera watches ONE welding bench in a rebar \
fabrication hall. Look at this frame and answer the questions in order.

STEP 1. Is there a person at the bench?
If you see no person at all - only the bench, the jig, rods, cables or the floor - \
answer `brak_na_stanowisku` and stop. An empty bench is common and normal; do not \
invent a worker.

STEP 2. If a person is there, is there direct visual evidence of welding RIGHT NOW: \
a bright arc glare, visible sparks, or the welding helmet lowered over the face with \
the torch held against the workpiece?
If yes, answer `spawanie`. If the person is merely near the bench, holding something, \
or wearing a helmet raised, this is NOT `spawanie`.

STEP 3. Otherwise, what are their hands doing?
- placing, positioning or arranging steel rods or bars in the jig -> `ukladanie_pretow`
- lifting or pulling a finished element off the bench -> `sciaganie_elementu`
- other real work: carrying wire, grinding, measuring, adjusting equipment, cleaning \
-> `inna_czynnosc`
- nothing at all: standing still, empty hands, not working -> `postoj`. If anything \
is in their hands, it is not `postoj`.

STEP 4. If the frame is too blurred, too dark, or too occluded to tell, answer \
`nierozpoznane`. This is a real answer, not a failure - use it rather than guessing.

Answer with one label only, no explanation."""

PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2}

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


def classify(crops: list[Path], stride: int, prompt: str) -> tuple[list[dict], float]:
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
                "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}],
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
    ap.add_argument("--prompt-version", default="v2", choices=sorted(PROMPTS))
    ap.add_argument("--arm", default=None, help="default: vlm-qwen2.5-vl-3b-<prompt version>")
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
        samples, load_seconds = classify(crops, stride, PROMPTS[args.prompt_version])
    wall_seconds = time.monotonic() - run_start

    doc = {
        "arm": args.arm or f"vlm-qwen2.5-vl-3b-{args.prompt_version}",
        "window": args.slot,
        "model": "Qwen/Qwen2.5-VL-3B-Instruct",
        "prompt_version": args.prompt_version,
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
