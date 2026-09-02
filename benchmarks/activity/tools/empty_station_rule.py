#!/usr/bin/env python3
"""Measure the parameter-free `brak_na_stanowisku` rule: nobody detected, station empty.

## Why this exists as its own tool

The C.0 client report carried **98.5%** for `brak_na_stanowisku` in a column of
held-out classifier accuracies. No arm ever scored that. 98.5% is 65/66, the
POSE DETECTOR's hit rate on W1's empty-bench frames - one window, in-sample, and
not a classifier output at all. It reached a published client report because it
was typed into a table by hand from a docstring.

So the rule gets measured properly, by a script, over every annotated window,
and the number it produces is the only one allowed to be quoted.

## The rule

No person detected in the station ROI at this sample, therefore the station is
empty. There is nothing to fit: no threshold, no training, no fold. That is what
makes it quotable across all three windows at once rather than per fold.

## What it finds, and why the good number is the wrong number

Recall is high - the detector really does miss nobody when the bench is empty.
The reported *time* is roughly double the true empty time, because the operator
is also undetected while working: crouched behind the jig, occluded by the cage,
or at the edge of the crop. Under this fixture's own `INFLATION_LIMIT` of 1.25x
that disqualifies the rule for time reporting, by exactly the same test that
disqualifies the arc-flash welding baseline. A recall-only bar would have passed
both.

Reads the pose cache written by `run_tcn_arm.py`, so it costs no GPU and cannot
disagree with the arm about what the detector saw.

    uv run benchmarks/activity/tools/empty_station_rule.py \
      --manifest benchmarks/activity/hala-prawe-v1/manifest.source.json \
      --cache runs/tcn/cache \
      --out benchmarks/activity/hala-prawe-v1/empty-station-rule.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

EMPTY = "brak_na_stanowisku"
# Column of the `found` flag in the cached raw detection vector. Kept in step
# with run_tcn_arm.FOUND; the loader below refuses a cache of another width
# rather than reading whatever sits at that index.
FOUND = 58
RAW_WIDTH = 59
INFLATION_LIMIT = 1.25


def newest_cache(cache_dir: Path) -> tuple[Path, dict]:
    files = sorted(cache_dir.glob("pose-seq-*.npz"), key=lambda p: p.stat().st_mtime)
    if not files:
        sys.exit(f"no pose cache in {cache_dir} - run run_tcn_arm.py first")
    npz = files[-1]
    meta = json.loads(npz.with_suffix(".json").read_text())
    return npz, meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text())
    fixture = args.manifest.parent
    npz_path, meta = newest_cache(args.cache)
    z = np.load(npz_path)

    per_window = {}
    truth_total = fires_total = hit_total = 0
    for clip in manifest["clips"]:
        if not clip.get("annotated"):
            continue
        slot = clip["slot"]
        if slot not in meta:
            sys.exit(f"cache {npz_path.name} has no window {slot} - it predates this fixture")
        x = z[f"x_{slot}"]
        if x.shape[1] != RAW_WIDTH:
            sys.exit(f"cache {npz_path.name} is {x.shape[1]} wide, expected {RAW_WIDTH}")
        gt = json.loads((fixture / clip["annotation_file"]).read_text())
        labels = [s["activity_id"] for s in gt["samples"]][: len(x)]
        empty = x[:, FOUND] == 0

        truth = sum(1 for v in labels if v == EMPTY)
        hit = sum(1 for i, v in enumerate(labels) if v == EMPTY and empty[i])
        fires = int(empty[: len(labels)].sum())
        per_window[slot] = {
            "truth_samples": truth,
            "rule_fires": fires,
            "hit": hit,
            "recall": hit / truth if truth else None,
            "time_ratio": fires / truth if truth else None,
        }
        truth_total += truth
        fires_total += fires
        hit_total += hit

    if not truth_total:
        sys.exit("no annotated sample carries the empty-station label")
    recall = hit_total / truth_total
    ratio = fires_total / truth_total
    doc = {
        "rule": "no person detected in the station ROI -> brak_na_stanowisku",
        "parameters": "none - nothing is fitted, so this is quotable over every window at once",
        "source_cache": npz_path.name,
        "per_window": per_window,
        "union": {
            "truth_samples": truth_total,
            "rule_fires": fires_total,
            "hit": hit_total,
            "recall": recall,
            "time_ratio": ratio,
        },
        "verdict": "usable"
        if ratio <= INFLATION_LIMIT
        else "recall passes, reported time inflated",
        "note": (
            "Replaces the 98.5% that appeared in the 2026-09-01 client report. That "
            "figure was the detector's W1-only hit rate, in-sample, quoted beside "
            "held-out classifier accuracies."
        ),
    }
    text = json.dumps(doc, indent=2, ensure_ascii=False)
    if args.out:
        args.out.write_text(text + "\n")
        print(f"wrote {args.out}")
    print(f"recall {recall:.1%}  time {ratio:.2f}x  n={truth_total}  -> {doc['verdict']}")
    for slot, w in per_window.items():
        r = "n/a" if w["recall"] is None else f"{w['recall']:.1%}"
        t = "n/a" if w["time_ratio"] is None else f"{w['time_ratio']:.2f}x"
        print(f"  {slot}: recall {r}  time {t}  n={w['truth_samples']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
