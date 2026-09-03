"""Tests for the `station_activity` section of `result.json` (issue #123).

The section answers one question — how much of the session went on each category —
and the whole design problem is making it impossible to read that answer as more
exact than it is. So three things travel with every total and none of them are
optional: the **measured time ratio**, the **coverage**, and the **`nierozpoznane`
share**. Hiding any of them makes every number above it look better than it is.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from pipeline.station_activity import build_station_activity
from pipeline.station_card import StationCard, TrainingWindow
from pipeline.zones import Zone

REPO_ROOT = Path(__file__).resolve().parent.parent


def _card(**overrides) -> StationCard:
    fields = {
        "version": "1.0.0",
        "station_id": "hala-prawe-v1",
        "zone_rect": (1700, 1360, 900, 800),
        "stride_s": 2,
        "model_outputs": ("spawanie", "ukladanie_pretow", "postoj", "nierozpoznane"),
        "delivered_classes": ("spawanie", "ukladanie_pretow", "pozostale", "nierozpoznane"),
        "bucket": "pozostale",
        "bucket_members": frozenset({"postoj"}),
        "abstention": "nierozpoznane",
        "time_ratios": {
            "spawanie": 1.06,
            "ukladanie_pretow": 1.10,
            "pozostale": 0.84,
            "nierozpoznane": 0.5,
        },
        "window": 64,
        "model_input": (224, 224),
        "training_windows": (
            TrainingWindow(
                slot="W1",
                window_local="2026-08-28 09:00-09:20 Europe/Warsaw",
                annotated_at="2026-09-01",
                samples=599,
            ),
        ),
        **overrides,
    }
    return StationCard(**fields)


ZONE = Zone(
    id="spawanie",
    name="Stanowisko spawalnicze",
    polygon=[(1700, 1360), (2600, 1360), (2600, 2160), (1700, 2160)],
    rules={"type": "station"},
)


def _section(categories, samples_possible=None, card=None, sha256="abc123", arc_metrics=None):
    card = card or _card()
    return build_station_activity(
        zone=ZONE,
        card=card,
        categories=categories,
        samples_possible=samples_possible if samples_possible is not None else len(categories),
        head_sha256=sha256,
        arc_metrics=arc_metrics,
    )


def _arcing(count: int, flagged: list[int]) -> list[float]:
    """An arc-metric series flagging exactly ``flagged``, everything else quiet."""
    return [0.5 if i in set(flagged) else 0.0 for i in range(count)]


class TestIntervalBoundaries:
    """Boundaries land at the sample midpoint, matching the annotation.

    Not a free choice: every `time_ratio` the card quotes was measured against
    intervals folded this way. Reporting under a different convention would
    quietly invalidate the error figure printed beside each total.
    """

    def test_a_boundary_lands_halfway_between_the_two_samples(self) -> None:
        section = _section(["spawanie", "spawanie", "pozostale", "pozostale"])

        assert section["intervals"] == [
            {"category": "spawanie", "start_s": 0.0, "end_s": 3.0},
            {"category": "pozostale", "start_s": 3.0, "end_s": 8.0},
        ]

    def test_the_stride_and_the_convention_are_recorded_in_the_output(self) -> None:
        section = _section(["spawanie"])

        assert section["stride_s"] == 2
        assert section["boundary"] == "sample_midpoint"

    def test_it_folds_exactly_as_the_scorer_does(self) -> None:
        """The convention is shared with `evaluate_arms.py`, not re-derived here.

        Two implementations of one convention drift, and the drift would show up
        as a client-facing duration that no longer matches the measurement its
        error bar came from. So they are compared rather than trusted.
        """
        sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "activity" / "tools"))
        import evaluate_arms as ev

        labels = ["spawanie"] * 5 + ["pozostale"] * 2 + ["nierozpoznane"] + ["spawanie"] * 4
        stride, duration = 2, len(labels) * 2
        grid = {i * stride: label for i, label in enumerate(labels)}

        expected = ev._fold_to_intervals(grid, stride, duration)
        ours = _section(labels)["intervals"]

        assert [
            {"activity_id": iv["category"], "start_s": iv["start_s"], "end_s": iv["end_s"]}
            for iv in ours
        ] == expected


class TestTotalsAndShares:
    def test_each_category_carries_its_total_share_and_measured_time_ratio(self) -> None:
        section = _section(["spawanie", "spawanie", "pozostale", "pozostale"])

        by_name = {c["category"]: c for c in section["categories"]}
        assert by_name["spawanie"]["total_s"] == 3.0
        assert by_name["spawanie"]["share"] == pytest.approx(3.0 / 8.0)
        assert by_name["spawanie"]["time_ratio"] == 1.06
        assert by_name["pozostale"]["total_s"] == 5.0
        assert by_name["pozostale"]["share"] == pytest.approx(5.0 / 8.0)

    def test_every_delivered_category_is_present_even_at_zero(self) -> None:
        """An absent key reads as "not measured"; a zero reads as "did not happen".

        The renderer must never have to branch on a missing category, and a
        category dropped for being empty is a category whose time ratio also
        disappears.
        """
        section = _section(["spawanie"])

        assert [c["category"] for c in section["categories"]] == [
            "spawanie",
            "ukladanie_pretow",
            "pozostale",
            "nierozpoznane",
        ]
        idle = {c["category"]: c for c in section["categories"]}["ukladanie_pretow"]
        assert idle["total_s"] == 0.0
        assert idle["share"] == 0.0
        assert idle["time_ratio"] == 1.10

    def test_the_abstention_share_is_present_when_nothing_was_unrecognised(self) -> None:
        # It counts as neither work nor downtime, so a reader has to see it is
        # zero rather than infer it from silence.
        section = _section(["spawanie", "spawanie"])

        by_name = {c["category"]: c for c in section["categories"]}
        assert by_name["nierozpoznane"]["total_s"] == 0.0
        assert by_name["nierozpoznane"]["share"] == 0.0
        assert section["abstention"] == "nierozpoznane"

    def test_a_total_never_appears_without_a_time_ratio_beside_it(self) -> None:
        section = _section(["spawanie", "pozostale", "nierozpoznane"])

        for category in section["categories"]:
            assert category["time_ratio"] is not None
            assert "total_s" in category


class TestCoverage:
    """Samples predicted over samples possible.

    A gap that silently leaves the denominator flatters every number above it, so
    the shares are computed against what the session *could* have produced and the
    shortfall stays visible.
    """

    def test_full_coverage_is_reported_rather_than_left_implicit(self) -> None:
        section = _section(["spawanie"] * 4)

        assert section["coverage"] == {
            "samples_predicted": 4,
            "samples_possible": 4,
            "fraction": 1.0,
        }

    def test_a_gap_lowers_the_shares_instead_of_shrinking_the_denominator(self) -> None:
        # Ten samples were possible; four were predicted. Welding covers all four,
        # which is 40% of the session and not 100% of it.
        section = _section(["spawanie"] * 4, samples_possible=10)

        assert section["coverage"]["fraction"] == pytest.approx(0.4)
        assert section["session_s"] == 20.0
        welding = {c["category"]: c for c in section["categories"]}["spawanie"]
        assert welding["total_s"] == 7.0
        assert welding["share"] == pytest.approx(7.0 / 20.0)

    def test_a_session_with_no_samples_still_reports_every_field(self) -> None:
        section = _section([], samples_possible=0)

        assert section["coverage"] == {
            "samples_predicted": 0,
            "samples_possible": 0,
            "fraction": 0.0,
        }
        assert section["intervals"] == []
        assert all(c["total_s"] == 0.0 and c["share"] == 0.0 for c in section["categories"])


class TestProvenance:
    def test_it_names_the_model_and_the_recordings_it_was_trained_on(self) -> None:
        """Accuracy holds for shifts and operators the model has seen.

        On this fixture a model trained on two morning recordings scored 27.5% on
        an evening one. The reader has to be able to see whether the shift being
        reported was covered.
        """
        section = _section(["spawanie"])

        assert section["model"]["station_id"] == "hala-prawe-v1"
        assert section["model"]["version"] == "1.0.0"
        assert section["model"]["sha256"] == "abc123"
        assert section["model"]["trained_on"] == [
            {
                "slot": "W1",
                "window_local": "2026-08-28 09:00-09:20 Europe/Warsaw",
                "annotated_at": "2026-09-01",
                "samples": 599,
            }
        ]

    def test_it_names_the_zone_and_the_rectangle_it_measured(self) -> None:
        section = _section(["spawanie"])

        assert section["zone_id"] == "spawanie"
        assert section["name"] == "Stanowisko spawalnicze"
        assert section["zone_native_px"] == {"x": 1700, "y": 1360, "w": 900, "h": 800}


class TestArcCheck:
    """The pixels' own answer to "was there an arc", carried beside the head's.

    This exists because of a run that reported `spawanie: 0.0 s` over an hour of
    welding: full coverage, every time ratio in place, and wrong. The head reads
    the middle 43% of the rectangle, the check reads all of it.
    """

    def test_a_section_built_without_arc_metrics_still_reports_the_check(self) -> None:
        """An absent check would read as "no arc seen" rather than "not looked
        for" — the exact confusion `coverage` already exists to prevent."""
        section = _section(["pozostale"] * 40)

        assert section["arc_check"]["verdict"] == "not_applicable"
        assert section["arc_check"]["samples"] == 0

    def test_it_contradicts_a_zero_welding_total_the_pixels_disagree_with(self) -> None:
        section = _section(
            ["pozostale"] * 400,
            arc_metrics=_arcing(400, list(range(0, 400, 25))),
        )

        assert section["arc_check"]["verdict"] == "contradicted"
        assert section["arc_check"]["samples_flagged"] == 16
        assert section["arc_check"]["predicted_samples"] == 0

    def test_a_contradiction_marks_every_category_in_the_zone_unreliable(self) -> None:
        """Not just `spawanie`. The seconds the head missed were not dropped,
        they were counted as something else, so `pozostale` is the number that
        absorbed the error and must not read as measured."""
        section = _section(
            ["pozostale"] * 400,
            arc_metrics=_arcing(400, list(range(0, 400, 25))),
        )

        assert [c["reliable"] for c in section["categories"]] == [False] * 4

    def test_an_agreeing_run_is_marked_reliable_on_every_category(self) -> None:
        section = _section(
            ["spawanie"] * 200 + ["pozostale"] * 200,
            arc_metrics=_arcing(400, list(range(0, 400, 25))),
        )

        assert section["arc_check"]["verdict"] == "consistent"
        assert all(c["reliable"] for c in section["categories"])

    def test_the_check_never_moves_a_single_second(self) -> None:
        """The verdict marks a total, it does not correct one. An arc test that
        edited durations would be quoting its own 2.18x over-reporting as fact.
        """
        categories = ["pozostale"] * 400
        without = _section(categories)
        with_arc = _section(categories, arc_metrics=_arcing(400, list(range(0, 400, 25))))

        assert [c["total_s"] for c in with_arc["categories"]] == [
            c["total_s"] for c in without["categories"]
        ]
        assert with_arc["intervals"] == without["intervals"]
        assert with_arc["coverage"] == without["coverage"]
