"""The `station_activity` section of `result.json` (issue #123).

Additive on purpose: nothing existing in the artefact changes, no consumer
breaks, and this repo ships without waiting for the platform.

The section answers a duration question — how much of the session went on
welding, how much on laying rods, how much on everything else — and the design
problem is making that answer impossible to read as more exact than it is. Three
things therefore travel with every total, and none of them are optional:

* **the measured time ratio**, predicted seconds over true seconds, carried from
  the model card. This fixture has already produced a baseline that met a 99.4%
  recall bar while reporting 2.18x the real welding time, so a bare "3 h 42 min
  of welding" invites a decision the measurement does not support.
* **coverage**, samples predicted over samples possible. Shares are computed
  against what the session could have produced, because a gap that silently
  leaves the denominator flatters every number above it.
* **the `nierozpoznane` share**, which counts as neither work nor downtime and
  keeps its own row rather than being folded into the collective bucket.

Each category is one object carrying its total, its share and its time ratio
together, rather than three parallel maps. A renderer can drop a key from a map
without noticing; it cannot easily show one field of an object and not its
sibling, and "never render a total without its measured error" is the load-bearing
requirement on the other side of this contract (gpu-exchange#210).

Interval boundaries land at the **sample midpoint**, the convention
`evaluate_arms.py` folds the annotation and every arm's predictions under. That
is not a free choice: every time ratio quoted here was measured against intervals
folded that way, so reporting under a different one would quietly invalidate the
error figure printed beside each total.
"""

from __future__ import annotations

from pipeline.station_card import StationCard
from pipeline.zones import Zone

# How interval edges are placed, stated in the artefact so a consumer can quote
# its own timing error rather than implying exactness. At a 2 s stride the
# annotation's own boundaries are only accurate to ±1 s.
BOUNDARY_CONVENTION = "sample_midpoint"


def _fold_to_intervals(
    categories: list[str], stride_s: float, session_s: float, complete: bool
) -> list[dict]:
    """Fold a per-sample category grid into intervals, boundaries at midpoints.

    Mirrors ``evaluate_arms._fold_to_intervals`` — deliberately, and the
    equivalence is tested rather than assumed.

    ``complete`` is the one departure, and it exists because the scorer never
    needs it: there, the grid always covers the clip, so the last interval is
    stretched to the full duration. Here samples can be missing, and stretching
    the final category over a gap would hand its unmeasured seconds to whichever
    category happened to be last. The shortfall stays in ``coverage`` instead.
    """
    intervals: list[dict] = []
    for index, category in enumerate(categories):
        t = index * stride_s
        if intervals and intervals[-1]["category"] == category:
            intervals[-1]["end_s"] = min(session_s, t + stride_s / 2)
            continue
        start = 0.0 if not intervals else max(0.0, t - stride_s / 2)
        intervals.append(
            {
                "category": category,
                "start_s": start,
                "end_s": min(session_s, t + stride_s / 2),
            }
        )
    if intervals and complete:
        # The last sample owns the half-stride at the end of the clip, exactly as
        # the first owns the one at the start.
        intervals[-1]["end_s"] = session_s
    return intervals


def build_station_activity(
    *,
    zone: Zone,
    card: StationCard,
    categories: list[str],
    samples_possible: int,
    head_sha256: str,
) -> dict:
    """One station zone's chronometraż, as the additive `result.json` section.

    ``categories`` are the delivered categories, one per predicted sample, in time
    order. ``samples_possible`` is how many the session could have produced at the
    card's stride — the denominator every share is taken against.
    """
    stride_s = float(card.stride_s)
    session_s = samples_possible * stride_s
    predicted = len(categories)
    complete = predicted == samples_possible
    intervals = _fold_to_intervals(categories, stride_s, session_s, complete)

    totals = dict.fromkeys(card.delivered_classes, 0.0)
    for interval in intervals:
        totals[interval["category"]] += interval["end_s"] - interval["start_s"]

    return {
        "zone_id": zone.id,
        "name": zone.name,
        "zone_native_px": dict(
            zip(("x", "y", "w", "h"), (int(v) for v in card.zone_rect), strict=True)
        ),
        "stride_s": card.stride_s,
        "boundary": BOUNDARY_CONVENTION,
        "session_s": session_s,
        "coverage": {
            "samples_predicted": predicted,
            "samples_possible": samples_possible,
            "fraction": (predicted / samples_possible) if samples_possible else 0.0,
        },
        # Order comes from the card's delivered vocabulary, and every category is
        # emitted even at zero: an absent key reads as "not measured", a zero
        # reads as "did not happen", and dropping an empty category would drop
        # its time ratio with it.
        "categories": [
            {
                "category": c,
                "total_s": totals[c],
                "share": (totals[c] / session_s) if session_s else 0.0,
                "time_ratio": card.time_ratios[c],
            }
            for c in card.delivered_classes
        ],
        # Which of the categories above is the abstention — neither work nor
        # downtime — named rather than left for the reader to know.
        "abstention": card.abstention,
        "intervals": intervals,
        "model": {
            "station_id": card.station_id,
            "version": card.version,
            # The weights that produced these numbers, not just their version. A
            # bind-mount over ./models is the documented way to run other weights,
            # and without this it would change every total in silence.
            "sha256": head_sha256,
            # Accuracy holds for shifts and operators the model has seen: on this
            # fixture a model trained on two morning recordings scored 27.5% on an
            # evening one. The reader has to be able to see whether the shift being
            # reported was covered.
            "trained_on": [
                {
                    "slot": w.slot,
                    "window_local": w.window_local,
                    "annotated_at": w.annotated_at,
                    "samples": w.samples,
                }
                for w in card.training_windows
            ],
        },
    }
