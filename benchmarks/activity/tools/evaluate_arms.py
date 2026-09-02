# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Score C.0 spike arms against the `hala-prawe-v1` ground truth.

Issue #117 asks each arm for five things, and this tool produces all five from
one command so that no arm gets scored by a slightly different rule than
another:

1. per-activity accuracy on held-out material, as a confusion matrix,
2. boundary timing error against the human annotation,
3. GPU-seconds per video-hour, measured on a named fleet box,
4. a hardware verdict - anything over 12 GB on one card, or needing two, is
   disqualified whatever its accuracy,
5. a comparison against the arc-flash baseline on `spawanie`.

## The split is read, never chosen

Folds come from `manifest.source.json` -> `split`. The tool refuses to run if
that block is missing, because a split invented at scoring time is exactly what
the acceptance criterion forbids. Per-activity figures are computed on the
UNION of the folds' held-out predictions, so every labelled sample is scored
once, by a model that did not train on it.

## Recall is the bar, and the bar alone is not enough

The pass bar - 85% correct per activity - is scored as **recall**: of the
seconds the human called `spawanie`, how many did the arm also call `spawanie`.
That is what "correct classifications per activity" means for a work-study.

A recall-only bar is gameable, and not hypothetically: the free arc-flash
baseline, tuned for best F1, reaches **99.4% recall on `spawanie`** by calling
2.18x as much time welding as actually happened, at 45.6% precision. It clears
the client's bar while being useless.

So every class also reports `time_ratio` - predicted seconds over true seconds -
and a class whose recall passes above `INFLATION_LIMIT` is marked **gamed**
rather than passed. That column is the one a chronometraz client feels: being
told 900 s of welding when 447 s happened. Over-reporting productive time is the
commercially dangerous direction of error, and it is worse than under-reporting.

## `nierozpoznane` is scored, but not part of the bar

It is the honest "cannot tell" and never counts as work or downtime, so it gets
its own confusion row while the go/no-go verdict is taken over the six real
activities. An arm that abstains its way to a good score is caught by the
abstention rate, reported per arm.

## The delivered vocabulary is declared, never inferred

The client accepted a three-category scope while every arm was scored on seven,
so `manifest.source.json` -> `delivery_vocabulary` names one bucket and the
activities it merges, and the tool scores that vocabulary in a section of its
own. `--collapse bucket=a,b,c` overrides the block for exploration; the report
header prints which source was used, because an ad-hoc collapse and a declared
one do not carry the same weight. A collapse invented at scoring time is the
same defect as a split invented at scoring time.

`nierozpoznane` is refused as a member and keeps its own row - it is neither
work nor downtime, so bucketing it would convert unknown time into measured
time. The two vocabularies never share a table: a three-category figure must
never be mistakable for a seven-category one, and a reader cannot undo a merge
by reading harder.

## The baseline is reported at two operating points

The arc metric is clip-relative - the raw values differ ~3x between W1 and W2 on
the identical crop - so any single threshold tells a partial story, and which
partial story you get depends on where you put it:

- **conservative** (the cut-off the annotation hints used): 40.4% recall at
  93.1% precision. Reports 0.43x the real welding time.
- **oracle F1** (best threshold in hindsight, in-sample, unavailable in
  production): 99.4% recall at 45.6% precision. Reports 2.18x the real time.

Both are the same signal. Quoting either alone is misleading, which is why the
report prints both. The honest summary is that arc-flash trades recall against
precision along one axis and never gets both, because it detects an **arc**
while `spawanie` as an activity includes positioning, tacking, chipping slag and
the pauses between beads.

It stays the cost floor rather than a candidate for a reason that no threshold
fixes: it cannot separate `ukladanie_pretow` from `postoj` at all - five of the
seven activities, and all of the hard part.

## Prediction file format

One JSON per (arm, window). Either shape is accepted:

    {
      "arm": "vlm-qwen2.5-vl-3b",
      "window": "W2",
      "intervals": [{"activity_id": "spawanie", "start_s": 0, "end_s": 12}],
      "gpu": {
        "box": "cctv-vps", "gpu_index": 1,
        "gpu_seconds": 412.0, "video_seconds": 1199,
        "peak_vram_mib": 7866, "gpus_used": 1
      }
    }

`samples: [{"t_s": 0, "activity_id": "..."}]` works in place of `intervals`.
The `gpu` block may be omitted while an arm is still being developed; the tool
then reports its cost as UNMEASURED and refuses to call it a pass.

## Usage

    uv run benchmarks/activity/tools/evaluate_arms.py \
      --manifest benchmarks/activity/hala-prawe-v1/manifest.source.json \
      --predictions runs/vlm/*.json \
      --out benchmarks/activity/hala-prawe-v1/C0-report.md
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

# The single-card budget from #117. `cctv-vps` GPU 0 is held by SGLang, so the
# real ceiling is one 12 GB card - never the sum of two.
VRAM_BUDGET_MIB = 12 * 1024

# The client's bar (assumption A11), applied per activity and never to an average.
PASS_BAR = 0.85

# Ground-truth boundaries are only accurate to +/- stride/2, so a predicted
# boundary inside that window is exact as far as this fixture can tell.
# Anything beyond `SPURIOUS_TOLERANCE_S` from any real boundary is an invented one.
SPURIOUS_TOLERANCE_S = 4

# `nierozpoznane` is not an activity (manifest: "never counted as work or
# downtime"), so it is scored but held out of the go/no-go verdict.
NON_ACTIVITY = "nierozpoznane"

# A recall-only bar is gameable: an arm that calls everything `spawanie` scores
# ~100% recall on it. The arc baseline does exactly this when tuned for F1 -
# 99.4% recall at 45.6% precision - so the guard is not hypothetical.
# `time_ratio` is predicted seconds over true seconds for an activity, which is
# the number a chronometraz client actually feels: told 900 s of welding when
# 447 s happened. Beyond this ratio a passing recall is reported as gamed.
INFLATION_LIMIT = 1.25


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text())
    if "split" not in manifest:
        sys.exit(
            f"{path} has no `split` block. Declare the split in the manifest before "
            "scoring - a split chosen at scoring time is what the C.0 criterion forbids."
        )
    return manifest


def _grid_from_intervals(intervals: list[dict], stride: int, count: int) -> dict[int, str]:
    """Sample an interval list onto the annotation's own `t_s` grid."""
    grid: dict[int, str] = {}
    for x in intervals:
        start, end = float(x["start_s"]), float(x["end_s"])
        for i in range(count):
            t = i * stride
            if start <= t < end:
                grid[t] = x["activity_id"]
    return grid


def load_annotation(path: Path) -> dict:
    doc = json.loads(path.read_text())
    stride = int(doc["stride_s"])
    samples = {int(s["t_s"]): s["activity_id"] for s in doc["samples"]}
    return {
        "window": doc["window"],
        "stride_s": stride,
        "duration_s": float(doc["duration_s"]),
        "grid": samples,
        "intervals": doc["intervals"],
        "count": len(doc["samples"]),
    }


def load_prediction(path: Path, truth: dict) -> dict:
    doc = json.loads(path.read_text())
    stride, count = truth["stride_s"], truth["count"]
    if "samples" in doc:
        grid = {int(s["t_s"]): s["activity_id"] for s in doc["samples"]}
        intervals = doc.get("intervals") or _fold_to_intervals(grid, stride, truth["duration_s"])
    elif "intervals" in doc:
        intervals = doc["intervals"]
        grid = _grid_from_intervals(intervals, stride, count)
    else:
        sys.exit(f"{path}: prediction needs `intervals` or `samples`")
    return {
        "arm": doc.get("arm") or path.stem,
        "window": doc["window"],
        "grid": grid,
        "intervals": intervals,
        "gpu": doc.get("gpu"),
        # What the arm says it is, rendered under its heading. A diagnostic
        # scored in the same table as the candidates has to say so, or the next
        # reader counts it as a candidate.
        "what": doc.get("rung") or doc.get("model"),
        "path": path,
    }


def _fold_to_intervals(grid: dict[int, str], stride: int, duration: float) -> list[dict]:
    """Fold a per-sample grid into intervals, boundaries at sample midpoints."""
    ts = sorted(grid)
    out: list[dict] = []
    for t in ts:
        label = grid[t]
        if out and out[-1]["activity_id"] == label:
            out[-1]["end_s"] = min(duration, t + stride / 2)
            continue
        start = 0.0 if not out else max(0.0, t - stride / 2)
        out.append({"activity_id": label, "start_s": start, "end_s": min(duration, t + stride / 2)})
    if out:
        out[-1]["end_s"] = duration
    return out


# --------------------------------------------------------------------------
# arc-flash baseline
# --------------------------------------------------------------------------


def arc_metric_series(arc_csv: Path, window: str) -> dict[int, float]:
    per_second: dict[int, float] = {}
    with arc_csv.open() as fh:
        for row in csv.DictReader(fh):
            if row.get("window") == window:
                per_second[int(row["t_s"])] = float(row["arc_metric"])
    return per_second


def arc_baseline_grid(
    per_second: dict[int, float], truth: dict, threshold: float
) -> dict[int, str]:
    """Binary spawanie / not-spawanie on the annotation grid."""
    stride, count = truth["stride_s"], truth["count"]
    grid: dict[int, str] = {}
    for i in range(count):
        t = i * stride
        # An arc is intermittent, so any arcing second inside the sample counts.
        vals = [per_second.get(s, 0.0) for s in range(t, t + stride)]
        grid[t] = "spawanie" if any(v >= threshold for v in vals) else "__nie_spawanie__"
    return grid


def conservative_arc_threshold(per_second: dict[int, float], fraction: float = 0.25) -> float:
    """The cut-off `build_interval_annotation.py` used for annotation hints.

    Clip-relative by construction — a fraction of the way from this clip's median
    to its 99th percentile — because the raw metric differs ~3x between W1 and W2
    on the identical crop.
    """
    if not per_second:
        return 0.0
    values = sorted(per_second.values())
    median = statistics.median(values)
    p99 = values[min(len(values) - 1, int(len(values) * 0.99))]
    return median + (p99 - median) * fraction


def tune_arc_threshold(per_second: dict[int, float], truth: dict) -> tuple[float, float]:
    """Best-F1 threshold on `spawanie` for THIS clip. Oracle, in-sample, on purpose."""
    if not per_second:
        return 0.0, 0.0
    candidates = sorted(set(per_second.values()))
    best_f1, best_t = 0.0, candidates[0]
    for t in candidates:
        grid = arc_baseline_grid(per_second, truth, t)
        tp = sum(
            1 for k, v in truth["grid"].items() if v == "spawanie" and grid.get(k) == "spawanie"
        )
        fp = sum(
            1 for k, v in truth["grid"].items() if v != "spawanie" and grid.get(k) == "spawanie"
        )
        fn = sum(
            1 for k, v in truth["grid"].items() if v == "spawanie" and grid.get(k) != "spawanie"
        )
        if tp == 0:
            continue
        f1 = 2 * tp / (2 * tp + fp + fn)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


# --------------------------------------------------------------------------
# delivery vocabulary
# --------------------------------------------------------------------------


def parse_collapse_flag(spec: str) -> dict:
    """`bucket=a,b,c` — the same mapping shape the manifest declares."""
    bucket, _, members = spec.partition("=")
    parts = [m.strip() for m in members.split(",") if m.strip()]
    if not bucket.strip() or not parts:
        sys.exit(
            f"--collapse {spec!r}: expected `bucket=activity,activity,...`, e.g. "
            "`--collapse pozostale=sciaganie_elementu,inna_czynnosc,postoj`"
        )
    return {"bucket": bucket.strip(), "members": parts, "source": "--collapse"}


def resolve_collapse(manifest: dict, flag: str | None) -> dict | None:
    """Resolve the delivery vocabulary and refuse an unusable one.

    The flag wins over the declared block so a vocabulary can be explored
    without editing the fixture, and `source` records which one was used —
    the report header prints it, because an ad-hoc collapse and a declared
    one do not carry the same weight.
    """
    declared = manifest.get("delivery_vocabulary")
    if flag is not None:
        collapse = parse_collapse_flag(flag)
    elif declared is not None:
        collapse = {**declared, "source": "manifest.source.json"}
    else:
        return None
    known = {a["id"] for a in manifest["activities"]}
    unknown = [m for m in collapse["members"] if m not in known]
    if unknown:
        sys.exit(
            f"delivery vocabulary: {', '.join(repr(u) for u in unknown)} "
            "is not an activity in the manifest. A typo would build a smaller "
            "bucket and still print a confident three-category number."
        )
    if collapse["bucket"] in known:
        sys.exit(
            f"delivery vocabulary: `{collapse['bucket']}` is already an activity "
            "id, so the name would mean both the activity and the bucket and no "
            "reader could tell which figure a row reports. Pick another name."
        )
    if NON_ACTIVITY in collapse["members"]:
        sys.exit(
            f"delivery vocabulary: `{NON_ACTIVITY}` cannot be a member of "
            f"`{collapse['bucket']}`. The manifest defines it as never work and "
            "never downtime, so folding it into a bucket converts unknown time "
            "into measured time. It keeps its own row."
        )
    return collapse


def collapse_classes(classes: list[str], collapse: dict) -> list[str]:
    """The class list under the collapsed vocabulary, in the manifest's order.

    The bucket takes the position of its first member so the delivered
    categories read in the order the fixture declares them.
    """
    members, bucket = set(collapse["members"]), collapse["bucket"]
    out: list[str] = []
    for c in classes:
        if c not in members:
            out.append(c)
        elif bucket not in out:
            out.append(bucket)
    return out


def collapse_pairs(pairs: list[tuple[str, str]], collapse: dict) -> list[tuple[str, str]]:
    """Re-label (truth, predicted) pairs into the collapsed vocabulary."""
    members, bucket = set(collapse["members"]), collapse["bucket"]
    return [(bucket if t in members else t, bucket if p in members else p) for t, p in pairs]


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------


def confusion(pairs: list[tuple[str, str]]) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = {}
    for truth_label, pred_label in pairs:
        matrix.setdefault(truth_label, {})
        matrix[truth_label][pred_label] = matrix[truth_label].get(pred_label, 0) + 1
    return matrix


def per_activity_scores(pairs: list[tuple[str, str]], classes: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for c in classes:
        support = sum(1 for t, _ in pairs if t == c)
        tp = sum(1 for t, p in pairs if t == c and p == c)
        predicted = sum(1 for _, p in pairs if p == c)
        recall = tp / support if support else None
        precision = tp / predicted if predicted else None
        time_ratio = (predicted / support) if support else None
        inflated = time_ratio is not None and time_ratio > INFLATION_LIMIT
        passes = recall is not None and recall >= PASS_BAR
        out[c] = {
            "support": support,
            "tp": tp,
            "predicted": predicted,
            "recall": recall,
            "precision": precision,
            "time_ratio": time_ratio,
            "inflated": inflated,
            # A class with few held-out samples cannot resolve an 85% bar; say so
            # rather than printing a confident percentage over a handful of samples.
            "granularity_pp": (100.0 / support) if support else None,
            "passes": passes,
            # The client's bar is recall, so `passes` reports it unchanged. But a
            # pass bought by over-calling the class is not a usable measurement,
            # and a work-study that over-reports productive time is worse than one
            # that under-reports it.
            "usable": passes and not inflated,
        }
    return out


def boundary_errors(truth_intervals: list[dict], pred_intervals: list[dict]) -> dict:
    """Distance from each real activity change to the nearest predicted one."""
    truth_b = sorted({float(x["start_s"]) for x in truth_intervals if float(x["start_s"]) > 0})
    pred_b = sorted({float(x["start_s"]) for x in pred_intervals if float(x["start_s"]) > 0})
    if not truth_b:
        return {"n": 0}
    if not pred_b:
        return {"n": len(truth_b), "matched": 0, "note": "arm emitted no boundaries"}

    errors = [min(abs(t - p) for p in pred_b) for t in truth_b]
    spurious = [p for p in pred_b if min(abs(p - t) for t in truth_b) > SPURIOUS_TOLERANCE_S]
    ordered = sorted(errors)
    return {
        "n": len(errors),
        "median_s": statistics.median(ordered),
        "p90_s": ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))],
        "max_s": ordered[-1],
        "within_2s_frac": sum(1 for e in errors if e <= 2) / len(errors),
        "spurious": len(spurious),
        "pred_boundaries": len(pred_b),
        "truth_boundaries": len(truth_b),
    }


def hardware_verdict(gpu: dict | None) -> dict:
    if not gpu:
        return {"verdict": "UNMEASURED", "reason": "no `gpu` block in the prediction file"}
    peak = gpu.get("peak_vram_mib")
    used = gpu.get("gpus_used", 1)
    if used and used > 1:
        return {"verdict": "DISQUALIFIED", "reason": f"needs {used} cards; the budget is one"}
    if peak is None:
        return {"verdict": "UNMEASURED", "reason": "`peak_vram_mib` missing"}
    if peak > VRAM_BUDGET_MIB:
        return {
            "verdict": "DISQUALIFIED",
            "reason": f"{peak} MiB peak exceeds the {VRAM_BUDGET_MIB} MiB single-card budget",
        }
    return {"verdict": "OK", "reason": f"{peak} MiB peak on one card"}


def gpu_seconds_per_video_hour(gpu: dict | None) -> float | None:
    if not gpu or not gpu.get("video_seconds"):
        return None
    return gpu["gpu_seconds"] / gpu["video_seconds"] * 3600


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _score_row(c: str, s: dict) -> str:
    """One row of the per-activity table — the same shape in either vocabulary."""
    if c == NON_ACTIVITY:
        bar = "—"
    elif s["usable"]:
        bar = "✅"
    elif s["passes"]:
        bar = "⚠️ gamed"
    else:
        bar = "❌"
    ratio = "n/a" if s["time_ratio"] is None else f"{s['time_ratio']:.2f}×"
    return (
        f"| `{c}` | {s['support']} | {_pct(s['recall'])} | {_pct(s['precision'])} | "
        f"{ratio} | {s['granularity_pp']:.1f} pp | {bar} |"
    )


def render(report: dict) -> str:
    m = report["manifest"]
    lines: list[str] = []
    add = lines.append

    add("# C.0 measurement report — `hala-prawe-v1`")
    add("")
    add(f"Generated {report['generated']} by `benchmarks/activity/tools/evaluate_arms.py`.")
    add("")
    # Derived from the manifest rather than written in prose, because this
    # paragraph outlived its own truth once: it still announced W3 as dropped
    # after W3 had been recovered and annotated.
    add("## Coverage")
    add("")
    annotated = [c for c in m["clips"] if c.get("annotated")]
    for c in annotated:
        add(f"- **{c['slot']}** {c['window_local']} — {c['shift_position']}")
    add("")
    positions = [c["shift_position"] for c in annotated]
    pre_break = sum("pre-break" in p for p in positions)
    if pre_break == len(annotated):
        add(
            "**Every annotated window is pre-break.** Every figure below describes the "
            "first half of a shift only and inherits that bias permanently. Quote it "
            "with the caveat attached or do not quote it."
        )
    elif pre_break:
        add(
            f"**{pre_break} of {len(annotated)} windows are pre-break.** The aggregate "
            f"still leans that way, and the one window that does not differs in both "
            f"shift and operator, so a per-window difference cannot be attributed to "
            f"either alone."
        )
    add("")
    add(m["split"].get("caveat", ""))
    add("")
    add("## Split")
    add("")
    split = m["split"]
    add(f"Protocol: **{split.get('protocol', 'single split')}**, declared {split['declared']}.")
    for fold in split.get("folds", []):
        add(f"- fold {fold['id']}: train {fold['train_dev']} → held out {fold['held_out']}")
    add("")
    add(split.get("reporting", ""))
    add("")

    collapse = report.get("collapse")
    if collapse:
        delivered = [c for c in collapse["classes"] if c != NON_ACTIVITY]
        add(f"## Delivery vocabulary: {len(delivered)} categories")
        add("")
        members = " + ".join(f"`{m}`" for m in collapse["members"])
        add(f"`{collapse['bucket']}` = {members} — declared in `{collapse['source']}`.")
        add("")
        add(
            f"Every figure in this section is over {len(delivered)} categories and is "
            f"**not comparable** with the per-arm sections below, which score all "
            f"{len(m['activities'])} separately. A merge cannot be undone by reading "
            "harder, so the two vocabularies never share a table."
        )
        add("")
        add(
            f"`{NON_ACTIVITY}` is **not** a member of the bucket and keeps its own row. "
            'It is neither work nor downtime, and folding the honest "cannot tell" '
            "into a work bucket would convert unknown time into measured time."
        )
        add("")

        add("### Held-out union")
        add("")
        for arm in report["arms"]:
            add(f"#### `{arm['name']}`")
            add("")
            add(
                "| Category | Support | Recall (the bar) | Precision | Time reported | "
                "1 error = | Verdict |"
            )
            add("|---|---:|---:|---:|---:|---:|:---:|")
            for c, s in arm["collapsed"]["scores"].items():
                if s["support"] == 0:
                    continue
                add(_score_row(c, s))
            add("")

        add("### Per held-out window")
        add("")
        add(
            "The union above is one number over folds that held out different "
            "material. Where those folds disagree, the mean describes neither — "
            "and on this fixture they disagree. Cells are recall (time reported, n)."
        )
        add("")
        for arm in report["arms"]:
            pw = arm["collapsed"]["per_window_scores"]
            cols = sorted(pw)
            add(f"#### `{arm['name']}`")
            add("")
            add("| Category | " + " | ".join(cols) + " |")
            add("|---|" + "---:|" * len(cols))
            for c, s in arm["collapsed"]["scores"].items():
                if s["support"] == 0:
                    continue
                cells = []
                for w in cols:
                    ws = pw[w].get(c)
                    if not ws or not ws["support"]:
                        cells.append("n/a")
                        continue
                    ratio = "n/a" if ws["time_ratio"] is None else f"{ws['time_ratio']:.2f}×"
                    cells.append(f"{_pct(ws['recall'])} ({ratio}, n={ws['support']})")
                add(f"| `{c}` | " + " | ".join(cells) + " |")
            add("")

    for arm in report["arms"]:
        add(f"## Arm: `{arm['name']}`")
        add("")
        if arm.get("what"):
            add(f"*{arm['what']}*")
            add("")
        hw = arm["hardware"]
        add(f"**Hardware verdict: {hw['verdict']}** — {hw['reason']}")
        cost = arm["gpu_seconds_per_video_hour"]
        box = arm.get("box") or "UNNAMED BOX"
        if cost is None:
            add("")
            add(
                "**Cost: UNMEASURED.** #117 requires GPU-seconds per video-hour measured "
                "on `cctv-vps` GPU 1 or `cctv-vps-2`, not on a workstation. Without it "
                "this arm cannot be recommended, whatever its accuracy."
            )
        else:
            add("")
            add(f"**Cost: {cost:.0f} GPU-seconds per video-hour**, measured on **{box}**.")
        add("")
        add(f"Abstention (`{NON_ACTIVITY}` predicted): {_pct(arm['abstention'])} of samples.")
        if arm["unpredicted"]:
            add("")
            add(
                f"⚠️ **{arm['unpredicted']} samples had no prediction** and are scored as "
                "errors. An arm that declines to answer does not get a smaller denominator."
            )
        add("")
        add("### Per-activity accuracy (held-out union)")
        add("")
        add(
            "| Activity | Support | Recall (the bar) | Precision | Time reported | "
            "1 error = | Verdict |"
        )
        add("|---|---:|---:|---:|---:|---:|:---:|")
        for c, s in arm["scores"].items():
            if s["support"] == 0:
                continue
            add(_score_row(c, s))
        add("")
        add(
            "*Time reported* is predicted seconds over true seconds for the activity — "
            f"the number a chronometraż client feels. Above {INFLATION_LIMIT:.2f}× a "
            "passing recall is marked **gamed**: the class was bought by over-calling "
            "it, and a work-study that over-reports productive time is worse than one "
            "that under-reports it."
        )
        add("")
        pw = arm.get("per_window_scores") or {}
        if len(pw) > 1:
            add("### Per held-out window")
            add("")
            add(
                "The union above is one number over folds that held out different "
                "material. Where those folds disagree, the mean describes neither."
            )
            add("")
            cols = sorted(pw)
            add("| Activity | " + " | ".join(cols) + " |")
            add("|---|" + "---:|" * len(cols))
            for c in arm["scores"]:
                if c == NON_ACTIVITY or not arm["scores"][c]["support"]:
                    continue
                cells = []
                for w in cols:
                    s = pw[w].get(c)
                    cells.append(
                        "n/a"
                        if not s or not s["support"]
                        else f"{_pct(s['recall'])} (n={s['support']})"
                    )
                add(f"| `{c}` | " + " | ".join(cells) + " |")
            add("")
        failing = [
            c
            for c, s in arm["scores"].items()
            if c != NON_ACTIVITY and s["support"] and not s["passes"]
        ]
        gamed = [
            c
            for c, s in arm["scores"].items()
            if c != NON_ACTIVITY and s["support"] and s["passes"] and not s["usable"]
        ]
        if failing:
            add(
                f"**Fails the bar on:** {', '.join('`' + c + '`' for c in failing)}. "
                "An 84% class fails even if the average clears."
            )
        if gamed:
            add(
                f"**Passes but unusable on:** {', '.join('`' + c + '`' for c in gamed)} — "
                "recall bought by over-calling the class."
            )
        if not failing and not gamed:
            add("**Every scored activity clears 85% without inflating its reported time.**")
        add("")
        add("### Confusion matrix")
        add("")
        preds = sorted({p for row in arm["confusion"].values() for p in row})
        add("| truth ↓ / pred → | " + " | ".join(f"`{p}`" for p in preds) + " |")
        add("|---" * (len(preds) + 1) + "|")
        for truth_label in sorted(arm["confusion"]):
            row = arm["confusion"][truth_label]
            cells = " | ".join(str(row.get(p, 0)) for p in preds)
            add(f"| `{truth_label}` | {cells} |")
        add("")
        add("### Boundary timing error")
        add("")
        b = arm["boundaries"]
        if b.get("n"):
            add(
                f"{b['truth_boundaries']} real activity changes, {b.get('pred_boundaries', 0)} "
                f"predicted. Median error **{b.get('median_s', 0):.1f} s**, p90 "
                f"{b.get('p90_s', 0):.1f} s, max {b.get('max_s', 0):.1f} s; "
                f"{_pct(b.get('within_2s_frac'))} land within 2 s. "
                f"Spurious boundaries (no real change within {SPURIOUS_TOLERANCE_S} s): "
                f"**{b.get('spurious', 0)}**."
            )
            add("")
            add(
                "The annotation's own boundaries are only accurate to ±1 s (2 s stride, "
                "boundary at the sample midpoint), so error below 1 s is not resolvable "
                "by this fixture and should not be read as precision."
            )
        else:
            add("No boundaries to score.")
        add("")

    add("## Arc-flash baseline on `spawanie`")
    add("")
    base = report["baseline"]
    add(
        "Reported at **two operating points**, because a single threshold tells a "
        "misleading story about this signal. *Conservative* is the clip-relative "
        "cut-off the annotation hints used. *Oracle F1* is the best threshold "
        "available in hindsight on that same clip — in-sample, unavailable in "
        "production, and deliberately generous: an arm that costs a GPU should have "
        "to beat the baseline's best day, not a strawman."
    )
    add("")
    labels = {"conservative": "Conservative", "oracle_f1": "Oracle F1 (in-sample)"}
    for point, per_window in base["points"].items():
        if not per_window:
            continue
        add(f"**{labels.get(point, point)}**")
        add("")
        add("| Window | Threshold | Recall | Precision | Time reported | F1 |")
        add("|---|---:|---:|---:|---:|---:|")
        for w, s in per_window.items():
            ratio = "n/a" if s.get("time_ratio") is None else f"{s['time_ratio']:.2f}×"
            add(
                f"| {w} | {s['threshold']:.2f} | {_pct(s['recall'])} | "
                f"{_pct(s['precision'])} | {ratio} | {s['f1']:.3f} |"
            )
        u = base["union"][point]
        ratio = "n/a" if u.get("time_ratio") is None else f"{u['time_ratio']:.2f}×"
        add("")
        add(
            f"Union on `spawanie`: recall **{_pct(u['recall'])}**, precision "
            f"{_pct(u['precision'])}, time reported {ratio}."
        )
        add("")
    add("Cost: **0 GPU-seconds** at either point.")
    add("")
    oracle = base["union"].get("oracle_f1", {})
    if oracle.get("recall") is not None and oracle["recall"] >= PASS_BAR:
        add(
            f"**The baseline clears the {PASS_BAR:.0%} recall bar on `spawanie` — and "
            "that is a finding about the bar, not about the baseline.** It reaches "
            f"{_pct(oracle['recall'])} recall by calling {oracle.get('time_ratio', 0):.2f}× "
            "as much time `spawanie` as actually was, at "
            f"{_pct(oracle['precision'])} precision. A recall-only bar is gameable by "
            "any arm willing to over-call the common class, so no arm should be "
            "promoted on recall alone. The `Time reported` column is what separates a "
            "measurement from a guess that happens to overlap the truth."
        )
    else:
        add(
            f"The baseline does not clear the {PASS_BAR:.0%} bar on `spawanie` even "
            "oracle-tuned, which is a property of the signal rather than of its tuning: "
            "it detects an **arc**, while `spawanie` as an activity includes "
            "positioning, tacking, chipping slag and the pauses between beads."
        )
    add("")
    add(
        "Either way the baseline is the **cost floor, not a candidate**: it cannot "
        "distinguish `ukladanie_pretow` from `postoj` at all, which is five of the "
        "seven activities and all of the hard part."
    )
    add("")
    add("## Go / no-go")
    add("")
    add(
        "_Not generated. The bar is numeric but the decision is human — "
        "recorded as a comment on issue #117 by @tkowalczyk._"
    )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--predictions", nargs="*", type=Path, default=[])
    ap.add_argument("--arc-csv", type=Path)
    ap.add_argument("--out", type=Path, help="markdown report (default: stdout)")
    ap.add_argument("--json-out", type=Path, help="also write the raw numbers as JSON")
    ap.add_argument(
        "--collapse",
        metavar="BUCKET=a,b,c",
        help="score an extra section under a collapsed vocabulary, overriding the "
        "manifest's `delivery_vocabulary` block",
    )
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    collapse = resolve_collapse(manifest, args.collapse)
    fixture_dir = args.manifest.parent
    classes = [a["id"] for a in manifest["activities"]]
    delivered_classes = collapse_classes(classes, collapse) if collapse else []

    truths: dict[str, dict] = {}
    for clip in manifest["clips"]:
        if not clip.get("annotated"):
            continue
        truths[clip["slot"]] = load_annotation(fixture_dir / clip["annotation_file"])
    if not truths:
        sys.exit("no annotated clips in the manifest — nothing to score against")

    # --- arms ---
    by_arm: dict[str, list[dict]] = {}
    for p in args.predictions:
        head = json.loads(p.read_text())
        # Names the file rather than raising a bare KeyError. The usual cause is
        # a shell glob that swallowed this tool's own --json-out sitting in the
        # same directory as the predictions.
        if "window" not in head:
            sys.exit(
                f"{p}: no `window` key - this is not a prediction file. "
                "If it is this tool's own --json-out, write it outside the "
                "directory the --predictions glob covers."
            )
        window = head["window"]
        if window not in truths:
            sys.exit(f"{p}: window {window} is not annotated")
        pred = load_prediction(p, truths[window])
        by_arm.setdefault(pred["arm"], []).append(pred)

    arms = []
    for name, preds in sorted(by_arm.items()):
        pairs: list[tuple[str, str]] = []
        # Also kept per window. The union over folds averages a fold that held
        # out familiar material with one that held out an unseen shift, and on
        # this fixture those differ by 70 points on `spawanie` - the mean of the
        # two describes neither, and reporting only the mean hides the finding.
        per_window_pairs: dict[str, list[tuple[str, str]]] = {}
        per_window_boundaries: dict[str, dict] = {}
        unpredicted = 0
        boundary_stats: list[dict] = []
        gpu_blocks = [p["gpu"] for p in preds if p["gpu"]]
        for pred in preds:
            truth = truths[pred["window"]]
            wp = per_window_pairs.setdefault(pred["window"], [])
            for t, gt in truth["grid"].items():
                got = pred["grid"].get(t)
                if got is None:
                    unpredicted += 1
                    got = "__brak_predykcji__"
                pairs.append((gt, got))
                wp.append((gt, got))
            b = boundary_errors(truth["intervals"], pred["intervals"])
            boundary_stats.append(b)
            per_window_boundaries[pred["window"]] = b

        windows = sorted(p["window"] for p in preds)
        held_out = {f["held_out"][0] for f in manifest["split"].get("folds", [])}
        if held_out and set(windows) != held_out:
            print(
                f"warning: arm `{name}` covers {windows} but the split's held-out "
                f"windows are {sorted(held_out)} — the union figure is incomplete",
                file=sys.stderr,
            )

        merged_boundaries = (
            boundary_stats[0]
            if len(boundary_stats) == 1
            else {
                "n": sum(b.get("n", 0) for b in boundary_stats),
                "truth_boundaries": sum(b.get("truth_boundaries", 0) for b in boundary_stats),
                "pred_boundaries": sum(b.get("pred_boundaries", 0) for b in boundary_stats),
                "median_s": statistics.median(
                    [b["median_s"] for b in boundary_stats if "median_s" in b] or [0]
                ),
                "p90_s": max((b.get("p90_s", 0) for b in boundary_stats), default=0),
                "max_s": max((b.get("max_s", 0) for b in boundary_stats), default=0),
                "within_2s_frac": statistics.mean(
                    [b["within_2s_frac"] for b in boundary_stats if "within_2s_frac" in b] or [0]
                ),
                "spurious": sum(b.get("spurious", 0) for b in boundary_stats),
            }
        )

        gpu = gpu_blocks[0] if gpu_blocks else None
        arms.append(
            {
                "name": name,
                "what": next((p["what"] for p in preds if p.get("what")), None),
                "windows": windows,
                "scores": per_activity_scores(pairs, classes),
                "per_window_boundaries": per_window_boundaries,
                "per_window_scores": {
                    w: per_activity_scores(wp, classes)
                    for w, wp in sorted(per_window_pairs.items())
                },
                "confusion": confusion(pairs),
                # The delivered vocabulary, scored from the same held-out pairs.
                # Kept in its own block rather than merged into `scores` so no
                # renderer can put the two vocabularies in one table.
                "collapsed": (
                    {
                        "scores": per_activity_scores(
                            collapse_pairs(pairs, collapse), delivered_classes
                        ),
                        "per_window_scores": {
                            w: per_activity_scores(collapse_pairs(wp, collapse), delivered_classes)
                            for w, wp in sorted(per_window_pairs.items())
                        },
                    }
                    if collapse
                    else None
                ),
                "boundaries": merged_boundaries,
                "hardware": hardware_verdict(gpu),
                "gpu_seconds_per_video_hour": gpu_seconds_per_video_hour(gpu),
                "box": (gpu or {}).get("box"),
                "abstention": sum(1 for _, p in pairs if p == NON_ACTIVITY) / len(pairs)
                if pairs
                else None,
                "unpredicted": unpredicted,
            }
        )

    # --- baseline ---
    arc_csv = args.arc_csv or (fixture_dir / "arc-timeline.csv")
    # Two operating points, because one threshold tells a misleading story. The
    # conservative point is the cut-off the annotation hints used; the oracle point
    # is the best F1 available in hindsight on that clip.
    points: dict[str, dict] = {"conservative": {}, "oracle_f1": {}}
    union_pairs: dict[str, list[tuple[str, str]]] = {"conservative": [], "oracle_f1": []}
    if arc_csv.exists():
        for slot, truth in truths.items():
            series = arc_metric_series(arc_csv, slot)
            thresholds = {
                "conservative": conservative_arc_threshold(series),
                "oracle_f1": tune_arc_threshold(series, truth)[0],
            }
            for point, threshold in thresholds.items():
                grid = arc_baseline_grid(series, truth, threshold)
                pairs = [(gt, grid.get(t, "__nie_spawanie__")) for t, gt in truth["grid"].items()]
                union_pairs[point] += pairs
                s = per_activity_scores(pairs, ["spawanie"])["spawanie"]
                denom = s["support"] + s["predicted"]
                points[point][slot] = {
                    "threshold": threshold,
                    **s,
                    "f1": (2 * s["tp"] / denom) if denom else 0.0,
                }
    union = {
        point: (
            per_activity_scores(pairs, ["spawanie"])["spawanie"]
            if pairs
            else {"recall": None, "precision": None, "time_ratio": None}
        )
        for point, pairs in union_pairs.items()
    }

    report = {
        "generated": manifest.get("verification", {}).get("date", "unknown"),
        "manifest": manifest,
        "arms": arms,
        "baseline": {"points": points, "union": union},
        "collapse": {**collapse, "classes": delivered_classes} if collapse else None,
    }

    text = render(report)
    if args.out:
        args.out.write_text(text)
        print(f"report:  {args.out}")
    else:
        print(text)
    if args.json_out:
        slim = {k: v for k, v in report.items() if k != "manifest"}
        args.json_out.write_text(json.dumps(slim, indent=2, ensure_ascii=False))
        print(f"numbers: {args.json_out}")

    if not arms:
        print(
            "\nNo arms scored — the baseline table above is the only content. "
            "Pass --predictions once an arm has run.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
