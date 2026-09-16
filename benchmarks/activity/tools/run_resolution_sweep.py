#!/usr/bin/env python3
"""#124: at what resolution does the station head stop being able to see?

## The question, and why the old answer is gone

`zone-annotator` has to tell an annotator, before a session starts, whether a
zone is resolvable enough to be worth labelling. `#119` originally took that
floor from the pose detector's recall risk (`diagnostics.detection_scale`,
`recall_risk`, `#113`). The winning arm runs no pose detector at all, so that
basis no longer exists and shipping the old threshold would be quoting a number
whose measurement has been deleted.

This measures the replacement, on data already in hand.

## What it does

The station crop is degraded to a fraction of its native resolution and put
straight back at full size, so the tensor the backbone sees never changes and the
only variable is how much detail the camera resolved. The head is then **refitted
at every scale** over the same folds, and the resulting predictions are scored by
`evaluate_arms.py` like any other arm.

**Refit, not evaluate.** Running the shipped head on shrunken inputs would measure
train/test mismatch, which is a different and much easier question. Refitting asks
the only thing the annotator cares about: at this resolution, is the information
there at all?

## Declared before the run

Reporting the best of several sweeps chosen afterwards is test-set fitting. So
the scales and the knee rule are in the committed script, before anything ran:

- Scales: 1.0, 0.75, 0.5, 0.35, 0.25, 0.18, 0.125
- Knee rule: the SMALLEST scale at which delivered `spawanie` recall stays within
  5.0 percentage points of the scale-1.0 figure AND its time ratio stays inside
  [0.90, 1.10]. Recall alone is not enough: an arc-flash threshold in this project
  hit 99.4% recall while reporting 2.18x the real welding time, so a recall-only
  bar passes a model that cannot count.

`spawanie` carries the rule because it is the activity the offer is sold on. Every
other class is reported beside it and none of them moves the verdict.

## What this does NOT measure

The human half. "Can an annotator still tell these apart" needs inter-annotator
agreement, which does not exist for this fixture, so the floor this produces is
the model-side floor and must be labelled as such. `zone-annotator` phase 9 says
so in the interface until the other half exists.

## Usage

    # inside the GPU container, on a fleet GPU
    python benchmarks/activity/tools/run_resolution_sweep.py \
      --manifest benchmarks/activity/hala-prawe-v1/manifest.source.json \
      --crops-root benchmarks/activity/hala-prawe-v1/crops \
      --out-dir runs/resolution --box cctv-vps --gpu-index 1 \
      --person-height-px 0   # 0 until measure_person_height.py has run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).parent))

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

# Fixed before the run. See the docstring.
SCALES = (1.0, 0.75, 0.5, 0.35, 0.25, 0.18, 0.125)
KNEE_CLASS = "spawanie"
KNEE_RECALL_TOLERANCE_PP = 5.0
KNEE_TIME_RATIO_BAND = (0.90, 1.10)


def arm_name(size: int, scale: float) -> str:
    return f"tcn-pixel-{size}-d{scale:g}"


def cv_folds(manifest: dict) -> list[dict]:
    """The cross-validation folds, without the ablation.

    The manifest's own reporting rule is that every labelled sample is predicted
    exactly once on the union. `E0-ablation` re-uses E's held-out window with a
    different training set, so including it would predict W5 twice and quietly
    weight it double in every figure below.
    """
    folds = [f for f in manifest["split"]["folds"] if "ablation" not in f["id"]]
    held = [f["held_out"][0] for f in folds]
    if len(set(held)) != len(held):
        sys.exit(f"folds hold out the same window twice: {held}")
    return folds


def gpu_from_predictions(pred_dir: Path, name: str) -> dict:
    """Cost for one arm, read back from the fold documents it wrote."""
    docs = sorted(pred_dir.glob(f"{name}-*.json"))
    if not docs:
        return {}
    gpus = [json.loads(d.read_text()).get("gpu", {}) for d in docs]
    first = gpus[0]
    return {
        "embed_seconds": first.get("embed_seconds_total"),
        "fit_seconds_total": round(sum(g.get("fit_seconds") or 0 for g in gpus), 1),
        "peak_vram_mib": first.get("peak_vram_mib"),
    }


def scores_for(report: dict, name: str) -> tuple[dict, str]:
    """Delivered-vocabulary scores for one arm, or the raw ones with a note."""
    entry = next((a for a in report["arms"] if a["name"] == name), None)
    if entry is None:
        sys.exit(f"evaluate_arms produced no arm named {name}")
    if entry.get("collapsed"):
        return entry["collapsed"]["scores"], "delivered"
    return entry["scores"], "raw (manifest declares no delivery_vocabulary)"


def knee(rows: list[dict]) -> dict:
    """Apply the declared rule. Returns the verdict, never a bare number."""
    base = next((r for r in rows if r["detail_scale"] == 1.0), None)
    if base is None or KNEE_CLASS not in base["scores"]:
        return {"scale": None, "why": f"no scale-1.0 baseline for {KNEE_CLASS}"}
    base_recall = base["scores"][KNEE_CLASS]["recall"]

    passing = []
    for r in sorted(rows, key=lambda x: x["detail_scale"]):
        s = r["scores"].get(KNEE_CLASS)
        if s is None:
            continue
        drop_pp = (base_recall - s["recall"]) * 100
        tr = s["time_ratio"]
        ok = drop_pp <= KNEE_RECALL_TOLERANCE_PP and (
            tr is not None and KNEE_TIME_RATIO_BAND[0] <= tr <= KNEE_TIME_RATIO_BAND[1]
        )
        if ok:
            passing.append(r["detail_scale"])
    return {
        "scale": min(passing) if passing else None,
        "rule": (
            f"smallest detail scale where {KNEE_CLASS} recall is within "
            f"{KNEE_RECALL_TOLERANCE_PP} pp of the scale-1.0 figure "
            f"({base_recall * 100:.1f}%) and time ratio is inside "
            f"{KNEE_TIME_RATIO_BAND}"
        ),
        "scales_passing": passing,
        "baseline_recall": base_recall,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--crops-root", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--box", required=True)
    ap.add_argument("--gpu-index", type=int)
    ap.add_argument("--image-size", type=int, default=518)
    ap.add_argument(
        "--person-height-px",
        type=float,
        default=0.0,
        help=(
            "median person height in NATIVE crop pixels, from "
            "measure_person_height.py. Converts each scale into the unit the "
            "annotator interface needs. 0 means unknown and the column is omitted."
        ),
    )
    ap.add_argument(
        "--scales",
        help="override the declared sweep, comma separated. Use only to resume.",
    )
    ap.add_argument(
        "--score-only",
        action="store_true",
        help=(
            "skip embedding and refitting, score the predictions already on disk. "
            "For when the sweep survived but the scoring step did not; refitting "
            "is not reproducible, so re-running it would silently replace the "
            "predictions the log describes."
        ),
    )
    args = ap.parse_args()

    if not args.score_only:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device != "cuda":
            sys.exit("no CUDA device - this sweep refits a head seven times")
    else:
        device = "cpu"

    manifest = json.loads(args.manifest.read_text())
    folds = cv_folds(manifest)
    scales = tuple(float(s) for s in args.scales.split(",")) if args.scales else SCALES
    if args.scales:
        print(f"WARNING: sweeping {scales}, not the declared {SCALES}", file=sys.stderr)

    pred_dir = args.out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    timings: dict[str, dict] = {}
    for scale in scales if not args.score_only else ():
        name = arm_name(args.image_size, scale)
        print(f"\n########## detail scale {scale:g}  ({name}) ##########", file=sys.stderr)

        vram = VramSampler()
        vram.__enter__()
        t0 = time.monotonic()
        data = embed_windows(
            args.manifest,
            args.crops_root,
            args.out_dir / "cache",
            args.image_size,
            device,
            detail_scale=scale,
        )
        embed_seconds = time.monotonic() - t0
        classes, windows = data["classes"], data["windows"]

        fit_total = 0.0
        for fold in folds:
            tr = [s for s in fold["train_dev"] if s in windows]
            te = fold["held_out"][0]
            if len(tr) != len(fold["train_dev"]) or te not in windows:
                sys.exit(f"fold {fold['id']} names a window the fixture does not have")
            feats = {s: windows[s]["x"] for s in (*tr, te)}
            t1 = time.monotonic()
            pred = train_fold(
                [(feats[s], windows[s]["y"]) for s in tr], feats[te], len(classes), device
            )
            fit_seconds = time.monotonic() - t1
            fit_total += fit_seconds
            stride = windows[te]["stride"]
            doc = {
                "arm": name,
                "window": te,
                "fold": fold["id"],
                "trained_on": tr,
                "model": f"dilated temporal CNN over frozen {BACKBONE} CLS embeddings",
                "rung": "#124 - resolution sweep for the zone-annotator floor",
                "feature_set": "pixel",
                "feature_width": int(feats[te].shape[1]),
                "image_size": args.image_size,
                "detail_scale": scale,
                "detail_scale_note": (
                    "crop degraded to this fraction of native and restored to full "
                    "size before embedding; the tensor the backbone sees is identical "
                    "at every scale, only the detail in it differs"
                ),
                "refit_at_this_scale": True,
                "hyperparameters": {
                    "window": WINDOW,
                    "channels": CHANNELS,
                    "dilations": list(DILATIONS),
                    "kernel": KERNEL,
                    "epochs": EPOCHS,
                    "lr": LR,
                    "seed": SEED,
                    "backbone_frozen": True,
                    "fixed_before_run": True,
                },
                "receptive_field_frames": 1 + (KERNEL - 1) * sum(DILATIONS),
                "samples": [
                    {"t_s": i * stride, "activity_id": classes[int(c)]} for i, c in enumerate(pred)
                ],
                "gpu": {
                    "box": args.box,
                    "gpu_index": args.gpu_index,
                    "gpus_used": 1,
                    # evaluate_arms reads this key directly, so its absence is a
                    # KeyError after the expensive part has already run. Same
                    # amortisation as run_tcn_pixel_arm: the embedding pass is
                    # shared by every fold at this scale.
                    "gpu_seconds": round(embed_seconds / max(len(folds), 1) + fit_seconds, 1),
                    "embed_seconds_total": round(embed_seconds, 1),
                    "fit_seconds": round(fit_seconds, 1),
                    "video_seconds": len(windows[te]["y"]) * stride,
                    "peak_vram_mib": int(vram.peak_mib) if vram.peak_mib else None,
                },
            }
            out = pred_dir / f"{name}-{fold['id']}-{te}.json"
            out.write_text(json.dumps(doc, indent=2, ensure_ascii=False))
            print(f"  fold {fold['id']} -> {te}: fit {fit_seconds:.0f}s", file=sys.stderr)
        vram.__exit__()
        timings[name] = {
            "embed_seconds": round(embed_seconds, 1),
            "fit_seconds_total": round(fit_total, 1),
            "peak_vram_mib": int(vram.peak_mib) if vram.peak_mib else None,
        }

    # Scored by the same scorer as every other arm in this fixture. An arm scored
    # by a bespoke script is an arm that cannot be compared.
    report_json = args.out_dir / "sweep-report.json"
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "evaluate_arms.py"),
        "--manifest",
        str(args.manifest),
        "--predictions",
        *[str(p) for p in sorted(pred_dir.glob("*.json"))],
        "--json-out",
        str(report_json),
        "--out",
        str(args.out_dir / "sweep-report.md"),
    ]
    print("\n" + " ".join(cmd), file=sys.stderr)
    subprocess.run(cmd, check=True)

    report = json.loads(report_json.read_text())
    rows = []
    vocab_note = ""
    for scale in scales:
        name = arm_name(args.image_size, scale)
        scores, vocab_note = scores_for(report, name)
        row = {
            "detail_scale": scale,
            "arm": name,
            "scores": scores,
            # Under --score-only the in-memory timings are gone, so they come
            # back from the prediction docs, which recorded them at the time.
            "gpu": timings.get(name) or gpu_from_predictions(pred_dir, name),
        }
        if args.person_height_px > 0:
            row["person_height_px"] = round(args.person_height_px * scale, 1)
        rows.append(row)

    verdict = knee(rows)
    summary = {
        "issue": 124,
        "question": "the resolution floor for a station zone, with no pose detector in the path",
        "method": (
            "crop degraded to a fraction of native and restored to full size, head "
            "REFIT at every scale over the same cross-validation folds, scored by "
            "evaluate_arms.py"
        ),
        "vocabulary": vocab_note,
        "scales_declared_before_run": list(SCALES),
        "scales_run": list(scales),
        "person_height_px_native": args.person_height_px or None,
        "person_height_source": (
            "measure_person_height.py, median over sampled native crops"
            if args.person_height_px
            else "NOT MEASURED - the floor cannot be expressed in annotator units yet"
        ),
        "knee": verdict,
        "proposed_floor_person_height_px": (
            round(args.person_height_px * verdict["scale"], 1)
            if args.person_height_px and verdict.get("scale")
            else None
        ),
        "measures_only_the_model_half": (
            "What a human annotator can still tell apart is NOT measured here and "
            "needs inter-annotator agreement, which this fixture does not have. "
            "zone-annotator must call the threshold provisional until it does."
        ),
        "rows": rows,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"\nvocabulary: {vocab_note}")
    hdr = f"{'scale':>7}  {'person px':>9}  " + "  ".join(
        f"{c:>22}" for c in sorted(rows[0]["scores"])
    )
    print(hdr)
    for r in rows:
        ph = f"{r.get('person_height_px', 0):.0f}" if args.person_height_px else "-"
        cells = []
        for c in sorted(r["scores"]):
            s = r["scores"][c]
            rec = "-" if s["recall"] is None else f"{s['recall'] * 100:5.1f}%"
            tr = "-" if s["time_ratio"] is None else f"{s['time_ratio']:.2f}x"
            cells.append(f"{rec} {tr:>7} n={s['support']:<5}")
        print(f"{r['detail_scale']:>7g}  {ph:>9}  " + "  ".join(cells))
    print(f"\nknee: {verdict}")
    print(f"wrote {args.out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
