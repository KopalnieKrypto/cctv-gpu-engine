#!/usr/bin/env python3
"""Rung 2 of #120: the C.0 winner's model, fed pixels instead of geometry.

## What rung 1 left open

Rung 1 embedded each station crop with a frozen DINOv2 and fitted a linear probe
per fold. It answered one class outright - `brak_na_stanowisku` went from 4.4%
(geometry) to 97.1% (pixels at 518 px) - and left the two hard hand-work classes
unresolved: `inna_czynnosc` 19.6% -> 28.0%, `postoj` 1.7% -> 8.6%. A few points,
moving in opposite directions across resolutions, on 177 and 62 samples.

That null is not decisive, and the reason is a handicap the probe carries by
construction: **it sees one frame.** `tcn-pose` sees 64 samples, a 128 s window.
Comparing a single-frame probe against a temporal model and concluding "pixels do
not carry it" would blame the representation for the absence of time.

This arm removes the handicap and changes nothing else. Same dilated temporal
CNN, same window, same dilations, same kernel, same epochs, same learning rate,
same seed, same folds - literally the same `train_fold` function, imported rather
than copied, so the claim "one variable changed" is checkable and not asserted.

## The configurations, all four fixed before the run and all four reported

Reporting only the best of several configurations scored on the same held-out
folds is test-set fitting with extra steps. So the list is declared here, in the
committed script, before any of them ran:

| `--features` | width | what it isolates |
|---|---|---|
| `pixel` | 768 | the literal swap: geometry out, DINOv2 CLS in |
| `pixel-pca64` | 64 | capacity control: geometry is 56-dim, so 768 vs 56 is a confound |
| `fused` | 120 | does adding pixels help the arm that already works, rather than replace it |

`pixel` is rung 2 as #120's ladder defines it. `fused` is the one whose answer a
product decision would actually turn on, which is why it is here too.

**Image size is 518, chosen a priori, not from rung 1's held-out scores.** The
question is whether a rod in the hands is visible; the native crop is 900x800 and
224 px leaves a rod a couple of pixels wide. 518 is also DINOv2's own
high-resolution evaluation size. `pixel` additionally runs at 224 because rung 1
hinted the hard classes preferred it, and whether that hint survives temporal
modelling is a real question - but the hint came from held-out numbers, so a 224
win here is reported as suggestive rather than as a finding.

**With four configurations scored on the same three folds, the best figure of the
four is optimistically biased.** Read the table as a set, not as a champion.

## PCA is fitted per fold, on training windows only

Components come from the two training windows of each fold and are then applied
to the held-out one. Fitting them on all three would leak the held-out window's
covariance into the representation - a quiet form of the retrofit this fixture's
split exists to prevent.

## Usage

    # inside the GPU container, on cctv-vps GPU 1
    python benchmarks/activity/tools/run_tcn_pixel_arm.py \
      --manifest benchmarks/activity/hala-prawe-v1/manifest.source.json \
      --crops-root benchmarks/activity/hala-prawe-v1/crops \
      --out-dir runs/tcn-pixel --box cctv-vps --gpu-index 1 \
      --features pixel --image-size 518

`--features fused` additionally needs `--pose-model`, because it concatenates the
same 56 geometry numbers `tcn-pose` consumes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).parent))

# Imported, never re-implemented: the whole point of this arm is that the model
# and the training loop are byte-identical to the ones that produced tcn-pose.
from run_pixel_probe_arm import BACKBONE, VramSampler, embed_windows  # noqa: E402
from run_tcn_arm import (  # noqa: E402
    CHANNELS,
    DILATIONS,
    EPOCHS,
    KERNEL,
    LR,
    SEED,
    WINDOW,
    build_feature_matrix,
    build_sequences,
    train_fold,
)

PCA_COMPONENTS = 64  # matched to geometry's 56 dims, fixed before the run


def fit_pca(train_x: list[np.ndarray], k: int) -> tuple[np.ndarray, np.ndarray]:
    """Principal components of the TRAINING windows only.

    Returns (mean, components) so the caller applies the identical transform to
    the held-out window without ever having fitted on it.
    """
    xtr = np.concatenate(train_x, axis=0)
    mean = xtr.mean(0, keepdims=True)
    # Economy SVD on a (n_samples, 768) matrix - cheap, and avoids forming the
    # covariance explicitly.
    _, _, vt = np.linalg.svd(xtr - mean, full_matrices=False)
    return mean.astype(np.float32), vt[:k].astype(np.float32)


def project(x: np.ndarray, mean: np.ndarray, comps: np.ndarray) -> np.ndarray:
    return ((x - mean) @ comps.T).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--crops-root", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--box", required=True)
    ap.add_argument("--gpu-index", type=int)
    ap.add_argument("--image-size", type=int, default=518)
    ap.add_argument(
        "--features",
        default="pixel",
        choices=("pixel", "pixel-pca64", "fused"),
        help="see the configuration table in the module docstring",
    )
    ap.add_argument(
        "--pose-model",
        type=Path,
        help="required for --features fused: the same detector tcn-pose used",
    )
    args = ap.parse_args()

    if args.features == "fused" and args.pose_model is None:
        sys.exit("--features fused concatenates geometry and needs --pose-model")

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

    geom: dict[str, np.ndarray] = {}
    if args.features == "fused":
        pose = build_sequences(args.manifest, args.pose_model, args.out_dir / "cache")
        for slot, w in windows.items():
            if slot not in pose["windows"]:
                sys.exit(f"pose cache has no window {slot}")
            g = build_feature_matrix(pose["windows"][slot]["x"], "pose", None)
            if len(g) != len(w["x"]):
                sys.exit(
                    f"{slot}: {len(g)} geometry rows vs {len(w['x'])} embeddings - "
                    "the two passes disagree on sample count, do not fuse them"
                )
            # Same crops, same stride, same annotation file: if the label vectors
            # differ, one of the two caches is stale and fusing them would align
            # pixels to the wrong labels.
            if not np.array_equal(pose["windows"][slot]["y"], w["y"]):
                sys.exit(f"{slot}: pose and embedding label vectors differ - stale cache")
            geom[slot] = g

    args.out_dir.mkdir(parents=True, exist_ok=True)
    total_video = sum(len(w["y"]) * w["stride"] for w in windows.values())
    emb_dim = next(iter(windows.values()))["x"].shape[1]
    print(f"embeddings: {emb_dim}-dim @ {args.image_size}px, features={args.features}")

    for fold in folds:
        tr = [s for s in fold["train_dev"] if s in windows]
        te = fold["held_out"][0]
        if len(tr) != len(fold["train_dev"]) or te not in windows:
            sys.exit(f"fold {fold['id']} names a window the fixture does not have")
        print(f"\n=== fold {fold['id']}: train {'+'.join(tr)} -> predict {te} ===", file=sys.stderr)

        if args.features == "pixel":
            feats = {s: windows[s]["x"] for s in (*tr, te)}
        else:
            mean, comps = fit_pca([windows[s]["x"] for s in tr], PCA_COMPONENTS)
            feats = {s: project(windows[s]["x"], mean, comps) for s in (*tr, te)}
            if args.features == "fused":
                feats = {s: np.concatenate([geom[s], feats[s]], axis=1) for s in (*tr, te)}

        t1 = time.monotonic()
        pred = train_fold(
            [(feats[s], windows[s]["y"]) for s in tr], feats[te], len(classes), device
        )
        fit_seconds = time.monotonic() - t1
        torch_peak = torch.cuda.max_memory_allocated() // (1024 * 1024)
        peak = None if embed_cached else (vram.peak_mib or None)
        stride = windows[te]["stride"]

        doc = {
            "arm": f"tcn-{args.features}-{args.image_size}",
            "window": te,
            "fold": fold["id"],
            "trained_on": tr,
            "model": (
                f"dilated temporal CNN over frozen {BACKBONE} CLS embeddings"
                + (" fused with tcn-pose geometry" if args.features == "fused" else "")
            ),
            "rung": "2 of #120 - the C.0 model with pixels in place of geometry",
            "feature_set": args.features,
            "feature_width": int(feats[te].shape[1]),
            "image_size": args.image_size,
            "hyperparameters": {
                "window": WINDOW,
                "channels": CHANNELS,
                "dilations": list(DILATIONS),
                "kernel": KERNEL,
                "epochs": EPOCHS,
                "lr": LR,
                "seed": SEED,
                "pca_components": (None if args.features == "pixel" else PCA_COMPONENTS),
                "pca_fitted_on": ("train_dev windows only" if args.features != "pixel" else None),
                "backbone_frozen": True,
                "fixed_before_run": True,
                "configurations_declared_before_run": 4,
            },
            "receptive_field_frames": 1 + (KERNEL - 1) * sum(DILATIONS),
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
                    else "whole process, nvidia-smi per PID: backbone + TCN"
                ),
                "torch_allocator_peak_mib": int(torch_peak),
            },
        }
        out = args.out_dir / f"{args.features}-{args.image_size}-{te}.json"
        out.write_text(json.dumps(doc, indent=2, ensure_ascii=False))
        print(f"wrote {out}  (fit {fit_seconds:.0f}s, peak {peak} MiB)")

    vram.__exit__()
    print(f"\nembedding pass: {embed_seconds:.0f}s for {total_video:.0f}s of video")
    print(f"peak VRAM (process): {vram.peak_mib or 'UNMEASURED'} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
