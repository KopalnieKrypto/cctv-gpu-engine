"""The shipped station head for `hala-prawe-v1`, and the card that describes it (#122).

The station-specific part of the offer is small on purpose: a frozen general
backbone does the looking and ships once inside the container image, and what
differs per station is a ~1 MB temporal head. Onboarding a station is "annotate
twenty minutes, train a head" - no new large model, no engine redeploy. This
module holds the parts of that which are pure and can be tested on a laptop; the
GPU command that uses them is `train_station_head.py`.

## The trap the card exists to avoid

The shipped weights train on every annotated window, so **they have no held-out
material and cannot be scored.** The accuracy quoted for them has to come from
the cross-validated runs, on models that are not the shipped one. That is normal
practice, and it is also exactly the seam where a number drifts loose from its
measurement.

This project has already lost that number once: 98.5% for `brak_na_stanowisku`
reached a client report after being copied by hand out of a detector docstring,
where it was the detector's own in-sample hit rate and no arm's score at all. So
every figure in the card is **read from `C0-report.json` at generation time** and
the build fails rather than emitting a card with a blank or a hand-set value.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

# Imported, never re-implemented. The head's output order and the scorer's class
# order have to be the same list: if they ever drift, the card quotes recall for
# one class against logits for another, and nothing in either file would say so.
from evaluate_arms import NON_ACTIVITY, collapse_classes  # noqa: E402


def assert_single_file(onnx_path: Path) -> None:
    """Refuse an export whose weights landed in a sibling file.

    torch writes them beside the graph as `<name>.onnx.data` unless told not to.
    A 36 KiB graph whose sha256 says nothing about the 1.8 MB of weights next to
    it is precisely the substitution `setup-models.sh` pins exist to catch, and
    that script verifies one file per model.
    """
    stray = onnx_path.with_suffix(".onnx.data")
    if stray.exists():
        sys.exit(
            f"{stray.name} was written alongside {onnx_path.name}: the weights are "
            "external, so the graph's sha256 verifies nothing. Export with "
            "`external_data=False` so the artefact is one self-contained file."
        )


def _require_figures(scores: dict, where: str) -> None:
    """Refuse to emit a card with a hole in it.

    A blank in a card is worse than a missing card: the reader fills it in from
    somewhere else, which is precisely how a detector's in-sample hit rate ended
    up quoted as a classifier's held-out accuracy.
    """
    for c, s in scores.items():
        if s.get("support") is None:
            sys.exit(f"{where}: `{c}` has no support count - the report is malformed.")
        # Zero support is not a hole. `nierozpoznane` never occurs on W1, so it
        # has no recall there, and that is a fact about the fixture rather than a
        # missing measurement. The card carries the null with its zero next to it.
        if s["support"] == 0:
            continue
        missing = [k for k in ("recall", "time_ratio") if s.get(k) is None]
        if missing:
            sys.exit(
                f"{where}: `{c}` has {s['support']} samples but no "
                f"{', '.join(missing)}. The card would carry a blank where a "
                "measurement belongs, so it is not written at all."
            )


def _figures(scores: dict) -> dict:
    return {
        c: {"recall": s["recall"], "time_ratio": s["time_ratio"], "support": s["support"]}
        for c, s in scores.items()
    }


def build_card(manifest: dict, report: dict, arm: str, training: dict) -> dict:
    """The model card. Every figure in it is read from `report`, never passed in."""
    entry = next((a for a in report["arms"] if a["name"] == arm), None)
    if entry is None:
        sys.exit(
            f"the report has no arm `{arm}` - every figure in a card is read from a "
            "measurement, so a card for an arm nobody scored has nowhere to come from. "
            f"Scored arms: {', '.join(a['name'] for a in report['arms']) or '(none)'}"
        )
    if not entry.get("collapsed"):
        sys.exit(
            f"arm `{arm}` was scored on the manifest's full vocabulary only. Re-run "
            "evaluate_arms.py against a manifest that declares `delivery_vocabulary`; "
            "quoting seven-category figures for a three-category head is the mix-up "
            "#121 exists to prevent."
        )
    scores = entry["collapsed"]["scores"]
    _require_figures(scores, f"{arm} held-out union")
    per_window = entry["collapsed"]["per_window_scores"]
    for w, ws in per_window.items():
        _require_figures(ws, f"{arm} held-out window {w}")

    annotated = [c for c in manifest["clips"] if c.get("annotated")]
    strides = {c.get("annotation_coverage", {}).get("stride_s") for c in annotated}
    if len(strides) != 1:
        sys.exit(
            f"the annotated windows disagree on the sampling stride ({sorted(strides)}). "
            "The head's receptive field is counted in samples, so there is no single "
            "stride for the card to state and 64 frames would mean two spans of time."
        )
    collapse = report["collapse"]
    preprocessing = training.get("preprocessing")
    if not preprocessing or "resize" not in preprocessing or "model_input" not in preprocessing:
        sys.exit(
            "the training record carries no preprocessing contract. The card must "
            "state both the resize target and the tensor the model actually "
            "receives - naming only one of them is how a 518 resize came to be "
            "read as a 518-pixel input."
        )
    comparison = training.get("comparison")
    # The decision and the artefact are not allowed to disagree. A card that
    # describes the model its own comparison rejected is worse than no card:
    # it carries the appearance of the check without its effect.
    if comparison and comparison["ships"] != training["trained_as"]:
        sys.exit(
            f"the comparison ships `{comparison['ships']}` but the artefact was "
            f"trained as `{training['trained_as']}`. Regressions: "
            f"{comparison['regressions']}. Train the winner, or re-run the "
            "comparison - do not write a card for the option that lost."
        )
    return {
        "artifact": {**training["artifact"], "trained_as": training["trained_as"]},
        "station": {
            "id": manifest["fixture"],
            "camera_id": manifest["camera"]["id"],
            "zone_native_px": manifest["station_roi"]["crop"],
            "stride_s": strides.pop(),
        },
        "vocabulary": {
            # What the ONNX emits and what the client is shown are not the same
            # list when the collapsed model ships: the head is seven-class and
            # the merge happens after argmax. A card naming only one of the two
            # leaves the consumer to guess which, so both are named.
            "model_outputs": training["output_classes"],
            "delivered_classes": collapse["classes"],
            "collapse_applied": (
                "none - the head emits them directly"
                if training["trained_as"] == "direct"
                else "after argmax, by the consumer"
            ),
            "bucket": collapse["bucket"],
            "bucket_contains": collapse["members"],
            "not_in_bucket": NON_ACTIVITY,
            "not_in_bucket_why": (
                "neither work nor downtime; folding it in would convert unknown "
                "time into measured time"
            ),
            "trained_as": training["trained_as"],
            "comparison": comparison,
        },
        "backbone": {
            "id": training["backbone"],
            # Resize and crop are different numbers, and quoting only the resize
            # is how this fixture's "518" arms came to be described as seeing 518
            # pixels when the processor centre-crops every one of them to 224.
            "preprocessing": {
                **preprocessing,
                "note": (
                    f"resize {preprocessing['resize']}x{preprocessing['resize']}, then "
                    f"centre-crop to {preprocessing['model_input'][0]}x"
                    f"{preprocessing['model_input'][1]} - the crop is what the model sees"
                ),
            },
            "frozen": True,
            "why_frozen": (
                "it is identical at every station and ships once inside the container "
                "image; only the ~1 MB head below is station-specific"
            ),
        },
        "head": {"hyperparameters": training["hyperparameters"]},
        "training_material": {
            # The shipped weights train on everything, which is the point - a
            # station gets twenty minutes of annotation and all of it should
            # reach the model. The cost is that these weights cannot be scored,
            # so the card states the consequence rather than leaving it implied.
            "windows": [
                {
                    "slot": c["slot"],
                    "window_local": c.get("window_local"),
                    "annotated_at": c.get("annotated_at"),
                    "samples": c.get("annotation_coverage", {}).get("labelled"),
                }
                for c in annotated
            ],
            "held_out": [],
            "all_annotated_material_used": True,
            "consequence": (
                "These weights saw every annotated window, so they have no held-out "
                "material and cannot be scored. Every figure under `accuracy` was "
                "measured on different models - the cross-validated folds - and "
                "describes the method, not this file."
            ),
        },
        "accuracy": {
            "arm": arm,
            "source": "C0-report.json",
            "from_these_weights": False,
            "measured_on": (
                "cross-validated held-out folds, on models that are not the shipped "
                "weights; read programmatically from C0-report.json at generation time"
            ),
            "union": _figures(scores),
            # The union is one number over folds that held out different material.
            # On this fixture they disagree by 45 points on the collective line,
            # so a card that quoted only the mean would describe a station that
            # behaves consistently - and this one does not.
            "per_window": {w: _figures(ws) for w, ws in sorted(per_window.items())},
        },
    }


def choose_vocabulary(direct: dict, collapsed: dict) -> dict:
    """Which of the two models ships, and why (#122).

    #121's collapsed figures are the floor. Training the delivered vocabulary
    directly ships only if it beats that floor **everywhere the card quotes it** -
    every delivered activity, on recall and on reported time. Any regression and
    the collapsed model ships instead.

    That is deliberately a one-sided bar rather than a net improvement: the
    burden of proof is on the new model, and a work-study client feels a class
    that got worse regardless of what got better next to it. `nierozpoznane` is
    excluded because the manifest defines it as not an activity - it is the
    abstention, scored and reported but never part of a verdict.
    """
    regressions: dict[str, list[str]] = {}
    deltas: dict[str, dict] = {}
    for c, floor in collapsed.items():
        if c == NON_ACTIVITY:
            continue
        got = direct.get(c)
        if got is None:
            regressions.setdefault(c, []).append("not predicted by the direct model")
            continue
        d_recall = got["recall"] - floor["recall"]
        # "Worse" for reported time is further from 1.0 in either direction. A
        # work-study can be wrong by over- or under-reporting, and the card
        # quotes the ratio itself, not its sign.
        d_ratio = abs(floor["time_ratio"] - 1.0) - abs(got["time_ratio"] - 1.0)
        deltas[c] = {"recall": d_recall, "time_ratio_toward_1": d_ratio}
        if d_recall < 0:
            regressions.setdefault(c, []).append("recall")
        if d_ratio < 0:
            regressions.setdefault(c, []).append("time_ratio")

    ships = "collapsed" if regressions else "direct"
    return {
        "ships": ships,
        "regressions": regressions,
        "deltas": deltas,
        "rule": (
            "the directly-trained vocabulary ships only if it regresses on no "
            "delivered activity, on neither recall nor reported time; "
            f"`{NON_ACTIVITY}` is excluded because it is not an activity"
        ),
    }


def remap_labels(y: np.ndarray, classes: list[str], collapse: dict) -> tuple[np.ndarray, list[str]]:
    """Re-index a label vector from the manifest's vocabulary into the delivered one.

    `y` indexes `classes` (the manifest's activity order). Training on the
    delivered vocabulary means remapping those indices and nothing else - the
    crops, the stride and the samples are identical, so a difference between the
    two runs is the label space and not the data.
    """
    members, bucket = set(collapse["members"]), collapse["bucket"]
    delivered = collapse_classes(classes, collapse)
    index = {c: i for i, c in enumerate(delivered)}
    return (
        np.asarray(
            [index[bucket if classes[i] in members else classes[i]] for i in y], dtype=np.int64
        ),
        delivered,
    )
