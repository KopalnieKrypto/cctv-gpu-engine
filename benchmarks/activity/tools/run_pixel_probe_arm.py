#!/usr/bin/env python3
"""Rung 1 of #120: do the station-crop PIXELS carry what pose keypoints threw away?

## The question

The C.0 winner (`tcn-pose`) consumes 59 numbers of pure geometry per sample -
bbox, confidence, 17 keypoints, crop size, a found flag. Zero pixels. It sees a
stick figure. And the classes that failed, failed on exactly that: from overhead
at this distance, "standing with empty hands" and "standing holding a rod"
produce nearly the same skeleton. The rod is in the pixels, and the arm never
got them.

No arm in #117 both saw pixels *and* was trained on the task: the two TCN arms
are blind to pixels by construction, and the VLM saw pixels but was prompted
zero-shot. This tool closes that gap in the cheapest way that can answer it.

## What it does, deliberately

A **frozen** vision backbone embeds each station crop, and a linear classifier is
fitted per fold on the training windows only. No fine-tuning, no augmentation, no
temporal context, no hyperparameter search. One frame in, one label out.

That austerity is the point. This is a diagnostic, not a candidate model:

- **A positive result is decisive.** If a general pretrained representation
  already separates the hand-work classes with nothing but a linear layer on top,
  the information is in the pixels and is cheap to reach. Rung 2 (temporal model
  over these embeddings) then has something to work with.
- **A negative result is NOT decisive.** It says the information is not
  *linearly* available in *this* frozen representation. Fine-tuning, or motion
  over pixels, could still find it. Do not report a null here as "pixels do not
  carry it".

`--image-size` matters more than usual and is swept rather than assumed. The
native crop is 900x800; a backbone's default 224 may destroy exactly the detail
the question turns on. If 224 fails and 518 succeeds, the finding is that the
signal is resolution-bound, which is worth knowing before anyone fine-tunes
anything.

## Comparability

Folds come from the manifest, embeddings are cached by (backbone, size, window
set), and predictions are written in the same shape every other arm uses, so
`evaluate_arms.py` scores this beside them with no special handling. An arm
scored by a bespoke script is an arm that cannot be compared.

    # inside the GPU container, on cctv-vps GPU 1
    python benchmarks/activity/tools/run_pixel_probe_arm.py \
      --manifest benchmarks/activity/hala-prawe-v1/manifest.source.json \
      --crops-root benchmarks/activity/hala-prawe-v1/crops \
      --out-dir runs/pixel --box cctv-vps --gpu-index 1 --image-size 224
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

# Fixed before the run and recorded in the output, same discipline as the TCN
# arm. A probe that gets a hyperparameter search is no longer a cheap
# diagnostic, and tuning it against the held-out fold would defeat the split.
EPOCHS = 400
LR = 1e-3
WEIGHT_DECAY = 1e-2
SEED = 117
BACKBONE = "facebook/dinov2-base"

# Embedding batch, expressed as a patch budget rather than an image count.
# DINOv2's activations scale with patches, not images, and the station tensor is
# no longer a fixed 224 square: at 882x420 one image is 1890 patches against the
# 256 a square used to be, so a batch of 32 asks for 7x the memory it used to and
# simply runs the card out. 8192 is the budget the old 32x256 actually used.
BATCH_PATCH_BUDGET = 8192
PATCH = 14


def embedding_batch(model_input: tuple[int, int]) -> int:
    """How many crops fit one forward pass at this tensor size."""
    patches = (model_input[0] // PATCH) * (model_input[1] // PATCH)
    return max(1, BATCH_PATCH_BUDGET // max(1, patches))


class VramSampler:
    """Peak VRAM for this PID, sampled from nvidia-smi (the #86 method).

    torch.cuda.max_memory_allocated sees only the allocator and misses the
    backbone's own footprint, which is the arm's real cost.
    """

    def __init__(self, interval: float = 0.5) -> None:
        self.peak_mib = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._interval = interval

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
            except Exception:  # nvidia-smi absent or transient - stays UNMEASURED
                pass
            self._stop.wait(self._interval)

    def __enter__(self) -> VramSampler:
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)


def embed_windows(
    manifest_path: Path, crops_root: Path, cache: Path, size: int, device: str
) -> dict:
    """Frozen-backbone embeddings per annotated window, cached by identity.

    ``size`` is the target **height**; the width comes from the manifest's
    station rectangle so the tensor keeps the bench's aspect ratio. The whole
    crop is resized into it and **never centre-cropped** - see
    `pipeline.station_classifier.preprocess_crop` for why that step was removed,
    and note that this is the second of the two implementations that had it.
    """
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    from pipeline.station_classifier import model_input_for_rect

    m = json.loads(manifest_path.read_text())
    fixture_dir = manifest_path.parent
    classes = [a["id"] for a in m["activities"]]
    slots = sorted(c["slot"] for c in m["clips"] if c.get("annotated"))

    roi = m["station_roi"]["crop"]
    rect = (int(roi["x"]), int(roi["y"]), int(roi["w"]), int(roi["h"]))
    model_input = model_input_for_rect(rect, size)

    # The rectangle is part of the cache identity, not just the tensor size. A
    # widened ROI produces different crops at the same `size`, and a stale hit
    # would train the head on embeddings of pixels nobody is looking at any more.
    key = hashlib.sha256(f"{BACKBONE}:{model_input}:{rect}:{','.join(slots)}".encode()).hexdigest()[
        :12
    ]
    npz = cache / f"emb-{key}.npz"
    meta_file = cache / f"emb-{key}.json"
    if npz.exists() and meta_file.exists():
        z = np.load(npz)
        meta = json.loads(meta_file.read_text())
        print(f"embedding cache hit: {npz.name}", file=sys.stderr)
        return {
            "classes": classes,
            "windows": {
                s: {"x": z[f"x_{s}"], "y": z[f"y_{s}"], "stride": info["stride"]}
                for s, info in meta.items()
            },
        }

    processor = AutoImageProcessor.from_pretrained(BACKBONE)
    model = AutoModel.from_pretrained(BACKBONE).to(device).eval()

    windows: dict[str, dict] = {}
    for clip in m["clips"]:
        if not clip.get("annotated"):
            continue
        slot = clip["slot"]
        truth = json.loads((fixture_dir / clip["annotation_file"]).read_text())
        labels = [s["activity_id"] for s in truth["samples"]]
        files = sorted((crops_root / f"{slot}-native").glob("t*.jpg"))
        if not files:
            sys.exit(f"no crops for {slot} in {crops_root} - run run_tcn_arm.py first")
        n = min(len(files), len(labels))

        vecs = []
        batch = embedding_batch(model_input)
        for start in range(0, n, batch):
            imgs = [Image.open(f).convert("RGB") for f in files[start : start + batch]]
            inputs = processor(
                images=imgs,
                return_tensors="pt",
                size={"height": model_input[0], "width": model_input[1]},
                # Not a default worth trusting: `crop_size` is independent of
                # `size` and stays at 224 unless told otherwise, which is how the
                # head came to read the middle 43% of the station.
                do_center_crop=False,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model(**inputs)
            # CLS token: the pooled representation DINOv2 is trained to make
            # linearly useful, which is exactly what a probe should read.
            vecs.append(out.last_hidden_state[:, 0].float().cpu().numpy())
            if (start + batch) % 200 < batch:
                print(f"  [{slot}] {min(start + batch, n)}/{n}", file=sys.stderr)
        windows[slot] = {
            "x": np.concatenate(vecs)[:n],
            "y": np.asarray([classes.index(v) for v in labels[:n]], dtype=np.int64),
            "stride": int(truth["stride_s"]),
        }

    cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz,
        **{f"x_{s}": w["x"] for s, w in windows.items()},
        **{f"y_{s}": w["y"] for s, w in windows.items()},
    )
    meta_file.write_text(json.dumps({s: {"stride": w["stride"]} for s, w in windows.items()}))
    return {"classes": classes, "windows": windows}


def fit_probe(train, xte, n_class: int, device: str):
    """Multinomial logistic regression, class-weighted, on standardised features."""
    import torch

    torch.manual_seed(SEED)
    xtr = np.concatenate([x for x, _ in train])
    ytr = np.concatenate([y for _, y in train])
    mu, sd = xtr.mean(0, keepdims=True), xtr.std(0, keepdims=True) + 1e-6

    xb = torch.tensor((xtr - mu) / sd, dtype=torch.float32, device=device)
    yb = torch.tensor(ytr, device=device)
    xt = torch.tensor((xte - mu) / sd, dtype=torch.float32, device=device)

    counts = np.bincount(ytr, minlength=n_class).astype(np.float32)
    weight = torch.tensor(
        np.where(counts > 0, counts.sum() / np.maximum(counts, 1), 0.0),
        dtype=torch.float32,
        device=device,
    )
    head = torch.nn.Linear(xb.shape[1], n_class).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    lossf = torch.nn.CrossEntropyLoss(weight=weight)
    head.train()
    for ep in range(EPOCHS):
        opt.zero_grad()
        loss = lossf(head(xb), yb)
        loss.backward()
        opt.step()
        if (ep + 1) % 100 == 0:
            print(f"  epoch {ep + 1}/{EPOCHS} loss {loss.item():.3f}", file=sys.stderr)
    head.eval()
    with torch.no_grad():
        return head(xt).argmax(1).cpu().numpy()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--crops-root", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--box", required=True)
    ap.add_argument("--gpu-index", type=int)
    ap.add_argument("--image-size", type=int, default=224)
    args = ap.parse_args()

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        sys.exit("no CUDA device - #117's envelope requires a fleet GPU")

    manifest = json.loads(args.manifest.read_text())
    folds = manifest["split"]["folds"]

    vram = VramSampler()
    vram.__enter__()
    t0 = time.monotonic()
    data = embed_windows(
        args.manifest, args.crops_root, args.out_dir / "cache", args.image_size, device
    )
    embed_seconds = time.monotonic() - t0
    embed_cached = embed_seconds < 5
    classes, windows = data["classes"], data["windows"]
    total_video = sum(len(w["y"]) * w["stride"] for w in windows.values())
    print(f"embeddings: {next(iter(windows.values()))['x'].shape[1]}-dim @ {args.image_size}px")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for fold in folds:
        tr = [s for s in fold["train_dev"] if s in windows]
        te = fold["held_out"][0]
        if len(tr) != len(fold["train_dev"]) or te not in windows:
            sys.exit(f"fold {fold['id']} names a window the fixture does not have")
        print(f"\n=== fold {fold['id']}: train {'+'.join(tr)} -> predict {te} ===", file=sys.stderr)
        t1 = time.monotonic()
        pred = fit_probe(
            [(windows[s]["x"], windows[s]["y"]) for s in tr],
            windows[te]["x"],
            len(classes),
            device,
        )
        fit_seconds = time.monotonic() - t1
        peak = None if embed_cached else (vram.peak_mib or None)
        stride = windows[te]["stride"]
        doc = {
            "arm": f"pixel-probe-dinov2-{args.image_size}",
            "window": te,
            "fold": fold["id"],
            "trained_on": tr,
            "model": f"frozen {BACKBONE} CLS embedding + linear probe, single frame",
            "rung": "1 of #120 - diagnostic, not a candidate model",
            "image_size": args.image_size,
            "hyperparameters": {
                "epochs": EPOCHS,
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "seed": SEED,
                "backbone_frozen": True,
                "temporal_context": None,
                "fixed_before_run": True,
            },
            "samples": [
                {"t_s": i * stride, "activity_id": classes[int(c)]} for i, c in enumerate(pred)
            ],
            "gpu": {
                "box": args.box,
                "gpu_index": args.gpu_index,
                "gpus_used": 1,
                "gpu_seconds": round(embed_seconds / max(len(folds), 1) + fit_seconds, 1),
                "embed_seconds_total": round(embed_seconds, 1),
                "fit_seconds": round(fit_seconds, 1),
                "video_seconds": len(windows[te]["y"]) * stride,
                "peak_vram_mib": int(peak) if peak else None,
                "peak_vram_note": (
                    "backbone not loaded (cache hit) - end-to-end peak unmeasured"
                    if embed_cached
                    else "whole process, nvidia-smi per PID: backbone + probe"
                ),
            },
        }
        out = args.out_dir / f"pixel{args.image_size}-{te}.json"
        out.write_text(json.dumps(doc, indent=2, ensure_ascii=False))
        print(f"wrote {out}  (fit {fit_seconds:.0f}s, peak {peak} MiB)")

    vram.__exit__()
    print(f"\nembedding pass: {embed_seconds:.0f}s for {total_video:.0f}s of video")
    print(f"peak VRAM (process): {vram.peak_mib or 'UNMEASURED'} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
