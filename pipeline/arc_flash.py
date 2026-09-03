"""An arc-flash cross-check that sits beside the station head's answer (#123).

Why this exists, stated plainly: on 2026-09-03 a production run reported
``spawanie: 0.0 s`` over an hour in which the welder is on camera, helmet down,
striking arcs — and nothing in ``result.json`` hinted at it. ``coverage.fraction``
was 1.0, every total carried its measured time ratio, and the number was still
wrong.

The cause is structural rather than one bad session. The head reads the station
through ``facebook/dinov2-base``'s processor, which resizes the crop to 518 and
then **centre-crops to 224**: it sees the middle 43% of each axis, 18.7% of the
authored rectangle's area. Work that drifts to the edge of the bench leaves the
model's field of view without leaving the rectangle the client was shown, and
``coverage`` cannot notice, because coverage counts samples predicted — not
whether the activity was in frame.

So the check reads the **whole** rectangle, which is the point: it looks exactly
where the head cannot.

This is not a second classifier and must never be read as one. It answers one
narrow question the pixels can answer independently and almost for free: *was
there an electric arc anywhere in the rectangle at all?* An arc is the only
activity in this vocabulary with an unmistakable optical signature — a small,
clipped, blue-white core that no hall lighting or sunlit concrete reproduces —
and C.0 already established that thresholding on it reaches 99.4% recall on
``spawanie``. That same C.0 run also established what it cannot do: it reported
2.18x the real welding time, so it is a **presence** test and never a duration.
Nothing here is added to, subtracted from, or used to correct a total.

Two departures from the C.0 arm, both forced by evidence:

* **Blue excess, not raw saturation.** The manifest's signal is the fraction of
  pixels above Y=235. Measured over the incident hour that signal is dominated
  by ambient drift, not by arcs: the idle minute at 12:44 has a *higher* median
  saturated fraction (0.942%) than the welding minute at 12:34 (0.642%), because
  a welder's dark body covers bright floor. Requiring the hot pixels to also be
  blue-dominant separates the two — the arc's plume pushes blue well above red,
  which neither daylight nor the hall's lamps do.
* **An absolute threshold, not a clip-relative one.** C.0 derives its cut-off
  per clip from that clip's own median and p99. That is right for proposing
  annotation hints and useless here: on a session with no welding at all it still
  flags its own top percentile, which is precisely the "did anyone weld" question
  this has to answer. The constant below is checked to mean the same thing in
  morning sun (W1), midday (this session) and evening dark (W3).
"""

from __future__ import annotations

import numpy as np

# The category an arc is evidence of. A card whose vocabulary does not contain it
# gets `not_applicable` rather than a guess — the optical signature is specific to
# arc welding and says nothing about laying rods.
ARC_LIT_CATEGORY = "spawanie"

# A pixel counts as arc-lit when it is both clipped-bright and blue-dominant.
# 235 is the manifest's own cut (nominal white); the blue excess is what makes it
# portable. Measured on the three annotated windows, `blue > red + 40` holds the
# per-sample precision at 79% (W1), 88% (W2) and 96% (W3) — the same constant in
# three very different lightings.
HOT_LUMA = 235
BLUE_EXCESS = 40

# Share of the rectangle that must be arc-lit before a sample is flagged, in
# percent. At 0.1 the incident hour flags 41 of 1831 samples across 28 separate
# minutes, while the nine-minute stretch where the crew stands around talking
# (12:40-12:48, verified by eye) flags none — its highest sample reaches 0.087.
ARC_PIXEL_PCT = 0.1

# What it takes to call the head contradicted. Deliberately conservative: a guard
# that cries wolf is worse than none, because the first false alarm is the reason
# nobody reads the second one.
#
# `MIN_FLAGGED_SAMPLES` and `MIN_BOUTS` together say "sustained and spread out",
# so one door opening onto a sunlit yard cannot trigger it. Bouts are counted with
# a gap because an arc is intermittent by nature — a welder strikes, breaks,
# repositions — so consecutive flags are the exception, not the rule.
MIN_FLAGGED_SAMPLES = 12
MIN_BOUTS = 5
BOUT_GAP_SAMPLES = 15

# How far under the arc evidence the head's own count has to fall. A healthy run
# lands far above this: the arc test over-reports (C.0's 2.18x), so a head that is
# working predicts *more* welding samples than are flagged, not a quarter of them.
CONTRADICTION_RATIO = 0.25

NOT_APPLICABLE = "not_applicable"
CONSISTENT = "consistent"
CONTRADICTED = "contradicted"


def arc_metric(crop_bgr: np.ndarray) -> float:
    """Percentage of ``crop_bgr`` that is clipped-bright **and** blue-dominant.

    ``crop_bgr`` is the native station rectangle exactly as the classifier
    receives it — the whole rectangle, before the processor's centre-crop throws
    57% of each axis away.

    Integer arithmetic in ``int32``: the luma coefficients are BT.601 scaled by
    256, which keeps this a few passes over the crop rather than a float
    conversion of it. ``int16`` is not enough headroom — ``green * 150`` overflows
    it, silently, and every metric comes back 0.0.
    """
    if crop_bgr.size == 0:
        return 0.0
    blue = crop_bgr[:, :, 0].astype(np.int32)
    green = crop_bgr[:, :, 1].astype(np.int32)
    red = crop_bgr[:, :, 2].astype(np.int32)
    luma = (red * 77 + green * 150 + blue * 29) >> 8
    lit = (luma > HOT_LUMA) & ((blue - red) > BLUE_EXCESS)
    return float(lit.sum()) * 100.0 / lit.size


def _count_bouts(flags: list[bool], gap: int = BOUT_GAP_SAMPLES) -> int:
    """How many separate stretches of arcing the flags describe.

    Flags closer together than ``gap`` samples belong to the same bout: one
    welder working one seam produces a burst of strikes, and counting each strike
    as its own bout would make a single event look like sustained evidence.
    """
    bouts = 0
    previous: int | None = None
    for index, flagged in enumerate(flags):
        if not flagged:
            continue
        if previous is None or index - previous > gap:
            bouts += 1
        previous = index
    return bouts


def build_arc_check(
    *,
    metrics: list[float],
    categories: list[str],
    stride_s: float,
    delivered_classes: tuple[str, ...],
) -> dict:
    """The arc evidence for one station session, and what it says about the head.

    ``metrics`` is :func:`arc_metric` per predicted sample, in the same order as
    ``categories``. A short ``metrics`` (an older caller, or a run that collected
    none) yields ``not_applicable`` rather than a verdict computed from a
    mismatched pairing.

    The verdict never edits a total. It marks one, which is the whole point: the
    failure this exists for produced a total that was wrong *and* well-formed, and
    a consumer had no way to tell.
    """
    flagged = [m >= ARC_PIXEL_PCT for m in metrics]
    samples_flagged = sum(flagged)
    bouts = _count_bouts(flagged)
    predicted = sum(1 for c in categories if c == ARC_LIT_CATEGORY)

    if ARC_LIT_CATEGORY not in delivered_classes or len(metrics) != len(categories):
        verdict = NOT_APPLICABLE
    elif (
        samples_flagged >= MIN_FLAGGED_SAMPLES
        and bouts >= MIN_BOUTS
        and predicted < samples_flagged * CONTRADICTION_RATIO
    ):
        verdict = CONTRADICTED
    else:
        verdict = CONSISTENT

    return {
        "signal": "hot_blue_pixel_pct",
        "category": ARC_LIT_CATEGORY,
        "threshold_pct": ARC_PIXEL_PCT,
        "samples": len(metrics),
        "samples_flagged": samples_flagged,
        # Seconds of *sampled* arc evidence, not an estimate of welding time. The
        # arc test over-reports duration by roughly 2x and under-samples it at a
        # 2 s stride; the two errors do not cancel and this is not a total.
        "flagged_s": samples_flagged * stride_s,
        "bouts": bouts,
        "predicted_samples": predicted,
        "verdict": verdict,
    }
