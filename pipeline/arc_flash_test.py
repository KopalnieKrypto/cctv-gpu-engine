"""Tests for the arc-flash cross-check (#123 follow-up).

The check exists because a station run reported `spawanie: 0.0 s` across an hour
of welding and looked complete doing it — full coverage, every time ratio in
place. So the tests that matter are the ones about *judgement*: it has to fire on
that shape, stay quiet on a thin or ambiguous one, and never touch a total.
"""

from __future__ import annotations

import numpy as np

from pipeline.arc_flash import (
    ARC_LIT_CATEGORY,
    CONSISTENT,
    CONTRADICTED,
    NOT_APPLICABLE,
    arc_metric,
    build_arc_check,
)

DELIVERED = ("spawanie", "ukladanie_pretow", "pozostale", "nierozpoznane")


def _crop(fill: tuple[int, int, int], size: int = 100) -> np.ndarray:
    """A ``size``x``size`` BGR crop of one colour."""
    return np.full((size, size, 3), fill, dtype=np.uint8)


# Bright and blue-dominant: what an arc's plume looks like to the decoder.
ARC_BGR = (255, 255, 200)
# Just as bright, no blue excess: a sunlit patch of concrete or a hall lamp.
WHITE_BGR = (250, 250, 250)


class TestArcMetric:
    def test_a_plain_scene_reads_zero(self) -> None:
        assert arc_metric(_crop((110, 110, 110))) == 0.0

    def test_an_arc_lit_patch_is_counted_in_pixels_not_in_share(self) -> None:
        """Pixels, so that widening the rectangle cannot dilute the evidence.

        It once was a share, calibrated at 900x800, and widening the bench to
        1670x800 spread the same arc over 1.86x the area and switched the guard
        off without a word.
        """
        crop = _crop((110, 110, 110), size=100)
        crop[:10, :10] = ARC_BGR
        assert arc_metric(crop) == 100.0

        # The same arc in a rectangle twice as wide is the same evidence.
        wider = _crop((110, 110, 110), size=100)
        wider = np.concatenate([wider, _crop((110, 110, 110), size=100)], axis=1)
        wider[:10, :10] = ARC_BGR
        assert arc_metric(wider) == 100.0

    def test_bright_but_colourless_light_is_not_an_arc(self) -> None:
        """The whole reason the check is not a plain saturation threshold.

        Measured over the incident hour, raw saturated-pixel fraction is
        dominated by ambient drift: the idle minute at 12:44 has a *higher*
        median than the welding minute at 12:34, because a welder's dark body
        covers bright floor. Blue excess is what separates them.
        """
        crop = _crop((110, 110, 110))
        crop[:50, :50] = WHITE_BGR  # a quarter of the rectangle, blown out
        assert arc_metric(crop) == 0.0

    def test_the_luma_sum_has_headroom_for_a_saturated_green_channel(self) -> None:
        """Regression: `green * 150` overflows int16 and zeroes every metric.

        It fails silently — no exception, no warning, just 0.0 everywhere and a
        guard that never fires. This cost a full calibration run to spot.
        """
        crop = _crop(ARC_BGR, size=100)
        assert arc_metric(crop) == 10000.0

    def test_an_empty_crop_is_not_a_division_by_zero(self) -> None:
        assert arc_metric(np.zeros((0, 0, 3), dtype=np.uint8)) == 0.0


def _series(flagged_at: list[int], total: int) -> list[float]:
    """A metric series that flags exactly the given sample indices."""
    return [1000.0 if i in set(flagged_at) else 5.0 for i in range(total)]


class TestVerdict:
    def test_it_contradicts_a_head_that_reported_no_welding_at_all(self) -> None:
        """The incident shape: 1831 samples, arc evidence spread across the hour,
        and `spawanie: 0.0 s` reported over the top of it."""
        flagged = list(range(0, 1800, 45))  # 40 flags, far apart
        check = build_arc_check(
            metrics=_series(flagged, 1831),
            categories=["pozostale"] * 1831,
            stride_s=2.0,
            delivered_classes=DELIVERED,
        )
        assert check["verdict"] == CONTRADICTED
        assert check["samples_flagged"] == 40
        assert check["predicted_samples"] == 0
        assert check["flagged_s"] == 80.0
        assert check["category"] == ARC_LIT_CATEGORY

    def test_a_head_that_found_the_welding_is_not_contradicted(self) -> None:
        flagged = list(range(0, 1800, 45))
        categories = ["pozostale"] * 1831
        for i in range(600):
            categories[i] = "spawanie"
        check = build_arc_check(
            metrics=_series(flagged, 1831),
            categories=categories,
            stride_s=2.0,
            delivered_classes=DELIVERED,
        )
        assert check["verdict"] == CONSISTENT

    def test_thin_evidence_is_not_enough_to_call_a_report_wrong(self) -> None:
        """A guard that cries wolf is worse than none — the first false alarm is
        why nobody reads the second one."""
        check = build_arc_check(
            metrics=_series([10, 200], 1831),
            categories=["pozostale"] * 1831,
            stride_s=2.0,
            delivered_classes=DELIVERED,
        )
        assert check["verdict"] == CONSISTENT
        assert check["samples_flagged"] == 2

    def test_one_burst_of_light_is_one_bout_and_cannot_trigger_it(self) -> None:
        """Twenty flags in ten seconds is a door opening onto a sunlit yard, not
        a shift of welding — the count clears the bar, the spread does not."""
        check = build_arc_check(
            metrics=_series(list(range(500, 520)), 1831),
            categories=["pozostale"] * 1831,
            stride_s=2.0,
            delivered_classes=DELIVERED,
        )
        assert check["bouts"] == 1
        assert check["samples_flagged"] == 20
        assert check["verdict"] == CONSISTENT

    def test_a_working_head_out_predicts_the_detector_and_is_left_alone(self) -> None:
        """The W1 shape, from the run that proved the retrained head works.

        Four flags in 599 samples while the head itself calls ~230 of them
        welding. The detector fires on a minority of true welding samples by
        design, so a low flag count is not evidence against the head - the ratio
        is what carries the verdict.
        """
        categories = ["pozostale"] * 599
        for i in range(230):
            categories[i] = "spawanie"
        check = build_arc_check(
            metrics=_series([40, 180, 300, 500], 599),
            categories=categories,
            stride_s=2.0,
            delivered_classes=DELIVERED,
        )
        assert check["verdict"] == CONSISTENT

    def test_a_short_clip_with_a_blind_head_is_still_caught(self) -> None:
        """The incident clip: 7 flags across 331 samples, and the head calls no
        welding at all. Under the old counts this passed as `consistent`."""
        check = build_arc_check(
            metrics=_series([12, 60, 110, 175, 230, 280, 320], 331),
            categories=["ukladanie_pretow"] * 331,
            stride_s=2.0,
            delivered_classes=DELIVERED,
        )
        assert check["verdict"] == CONTRADICTED
        assert check["samples_flagged"] == 7

    def test_a_vocabulary_without_welding_gets_no_verdict(self) -> None:
        """The optical signature is specific to an arc and says nothing about
        laying rods, so a card that delivers neither is not judged by it."""
        check = build_arc_check(
            metrics=_series(list(range(0, 1800, 45)), 1831),
            categories=["pozostale"] * 1831,
            stride_s=2.0,
            delivered_classes=("ukladanie_pretow", "pozostale"),
        )
        assert check["verdict"] == NOT_APPLICABLE

    def test_a_mismatched_pairing_is_refused_rather_than_scored(self) -> None:
        check = build_arc_check(
            metrics=[],
            categories=["pozostale"] * 1831,
            stride_s=2.0,
            delivered_classes=DELIVERED,
        )
        assert check["verdict"] == NOT_APPLICABLE

    def test_it_reports_what_it_saw_even_when_it_reaches_no_verdict(self) -> None:
        """`not_applicable` still has to carry its counts: a reader comparing two
        runs needs to see that one was never checked, not infer it from a gap."""
        check = build_arc_check(
            metrics=[],
            categories=["pozostale"] * 3,
            stride_s=2.0,
            delivered_classes=DELIVERED,
        )
        for key in ("signal", "threshold_lit_pixels", "samples", "samples_flagged", "bouts"):
            assert key in check
