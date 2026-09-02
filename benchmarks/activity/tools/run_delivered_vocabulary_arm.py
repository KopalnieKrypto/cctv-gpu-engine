#!/usr/bin/env python3
"""Train `tcn-pixel-518` directly on the delivered vocabulary (#122).

## The one question this answers

#121 scored the three delivered categories by **collapsing a seven-class model
afterwards**: the model learned to separate `postoj` from `inna_czynnosc`, and
the scorer then merged the two. Training on three categories from the start is a
different model - it never spends capacity on a distinction nobody asked for, and
its loss is weighted by the three classes that ship rather than by seven.

Which of the two is better is not obvious in either direction. Merging labels
removes a distinction the model was being penalised for getting wrong, which
helps; it also removes supervision that may have been teaching a useful boundary,
which hurts. #122 requires the answer to be measured rather than assumed, with
#121's collapsed figures as the floor.

## One variable changed, and it is checkable

Same folds, same crops, same embeddings (the cache key does not include the
vocabulary, so this arm reads the identical `.npz` rung 2 wrote), same
`train_fold` imported rather than copied, same window, dilations, kernel, epochs,
learning rate and seed. The only difference is `y`: label indices are remapped
through the manifest's `delivery_vocabulary` before training.

Class weighting therefore differs as a *consequence* rather than as a second
change - `train_fold` weights by inverse class frequency, and the collective
bucket is one class of 441 samples where it used to be four of 110/177/62/92.
That is part of what "training on three" means, not a confound smuggled in.

## Its predictions are scored in their own report

A three-class arm predicts none of the four merged activities, so in a
seven-category table it reads as four catastrophic zeroes rather than as a model
that does not have those classes. That is the confusion #121 exists to prevent,
so these predictions go to their own directory and their own report, and only the
**delivery-vocabulary section** of that report is quoted.

## Usage

    # inside the GPU container, on cctv-vps GPU 1
    python benchmarks/activity/tools/run_delivered_vocabulary_arm.py \
      --manifest benchmarks/activity/hala-prawe-v1/manifest.source.json \
      --crops-root benchmarks/activity/hala-prawe-v1/crops \
      --out-dir benchmarks/activity/hala-prawe-v1/predictions-delivered \
      --cache-dir runs/tcn-pixel/cache \
      --box cctv-vps --gpu-index 1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Imported, never re-implemented: "one variable changed" is a claim about this
# script, and it is only checkable if the model and the loop are the same objects
# rung 2 used.
from evaluate_arms import resolve_collapse  # noqa: E402
from run_pixel_probe_arm import BACKBONE, VramSampler, embed_windows  # noqa: E402
from run_tcn_arm import (  # noqa: E402
    CHANNELS,
    DILATIONS,
    EPOCHS,
    KERNEL,
    LR,
    SEED,
    WINDOW,
    train_fold,
)
from station_head import remap_labels  # noqa: E402

IMAGE_SIZE = 518  # rung 2's winner, not re-tuned here


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--crops-root", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument(
        "--cache-dir",
        type=Path,
        help="reuse rung 2's embedding cache; the key ignores the vocabulary, so "
        "the identical .npz is read and no backbone pass runs",
    )
    ap.add_argument("--box", required=True)
    ap.add_argument("--gpu-index", type=int)
    ap.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    args = ap.parse_args()

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        sys.exit("no CUDA device - #117's envelope requires a fleet GPU")

    manifest = json.loads(args.manifest.read_text())
    collapse = resolve_collapse(manifest, None)
    if collapse is None:
        sys.exit(
            f"{args.manifest} declares no `delivery_vocabulary`. This arm exists to "
            "train that vocabulary directly; without it there is nothing to train."
        )
    folds = manifest["split"]["folds"]

    vram = VramSampler()
    vram.__enter__()
    t0 = time.monotonic()
    cache = args.cache_dir or (args.out_dir / "cache")
    data = embed_windows(args.manifest, args.crops_root, cache, args.image_size, device)
    embed_seconds = time.monotonic() - t0
    embed_cached = embed_seconds < 5
    classes, windows = data["classes"], data["windows"]

    # The remap is the whole arm. Everything downstream is rung 2 unchanged.
    delivered: list[str] = []
    for w in windows.values():
        w["y"], delivered = remap_labels(w["y"], classes, collapse)
    print(f"vocabulary: {len(classes)} -> {len(delivered)} classes {delivered}", file=sys.stderr)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for fold in folds:
        tr = [s for s in fold["train_dev"] if s in windows]
        te = fold["held_out"][0]
        if len(tr) != len(fold["train_dev"]) or te not in windows:
            sys.exit(f"fold {fold['id']} names a window the fixture does not have")
        print(f"\n=== fold {fold['id']}: train {'+'.join(tr)} -> predict {te} ===", file=sys.stderr)

        t1 = time.monotonic()
        pred = train_fold(
            [(windows[s]["x"], windows[s]["y"]) for s in tr],
            windows[te]["x"],
            len(delivered),
            device,
        )
        fit_seconds = time.monotonic() - t1
        torch_peak = torch.cuda.max_memory_allocated() // (1024 * 1024)
        peak = None if embed_cached else (vram.peak_mib or None)
        stride = windows[te]["stride"]

        doc = {
            "arm": f"tcn-pixel-{args.image_size}-delivered",
            "window": te,
            "fold": fold["id"],
            "trained_on": tr,
            "model": (
                f"dilated temporal CNN over frozen {BACKBONE} CLS embeddings, "
                "trained directly on the delivered vocabulary"
            ),
            "rung": (
                "#122 - the delivered vocabulary trained directly, against #121's "
                "collapsed figures as the floor"
            ),
            "vocabulary": {
                "classes": delivered,
                "bucket": collapse["bucket"],
                "bucket_contains": collapse["members"],
                "source": collapse["source"],
            },
            "one_variable_changed": (
                "labels only. Same folds, crops, embedding cache, train_fold, window, "
                "dilations, kernel, epochs, lr and seed as tcn-pixel-518."
            ),
            "image_size": args.image_size,
            "hyperparameters": {
                "window": WINDOW,
                "channels": CHANNELS,
                "dilations": list(DILATIONS),
                "kernel": KERNEL,
                "epochs": EPOCHS,
                "lr": LR,
                "seed": SEED,
                "backbone_frozen": True,
            },
            "samples": [
                {"t_s": i * stride, "activity_id": delivered[int(c)]} for i, c in enumerate(pred)
            ],
            "gpu": {
                "box": args.box,
                "gpu_index": args.gpu_index,
                "gpus_used": 1,
                "gpu_seconds": round(embed_seconds / max(len(folds), 1) + fit_seconds, 1),
                "embed_seconds_total": round(embed_seconds, 1),
                "embed_cached": embed_cached,
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
        out = args.out_dir / f"delivered-{args.image_size}-{te}.json"
        out.write_text(json.dumps(doc, indent=2, ensure_ascii=False))
        print(f"wrote {out}  (fit {fit_seconds:.0f}s)")

    vram.__exit__()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
