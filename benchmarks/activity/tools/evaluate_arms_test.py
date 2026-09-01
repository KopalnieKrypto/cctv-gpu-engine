"""Contract tests for the C.0 scoring harness.

These pin the rules that decide whether a C.0 number means anything: an
unanswered sample is an error rather than a smaller denominator, the bar is
per-activity and never an average, and an arm that will not fit one card is
disqualified regardless of accuracy.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import evaluate_arms as ev  # noqa: E402


def _annotation(window: str, labels: list[str], stride: int = 2) -> dict:
    duration = len(labels) * stride
    samples = [{"t_s": i * stride, "activity_id": a} for i, a in enumerate(labels)]
    grid = {s["t_s"]: s["activity_id"] for s in samples}
    return {
        "window": window,
        "stride_s": stride,
        "duration_s": duration,
        "grid": grid,
        "intervals": ev._fold_to_intervals(grid, stride, duration),
        "count": len(samples),
    }


class TestPerActivityScores:
    def test_bar_is_per_activity_not_an_average(self):
        # 18 correct `spawanie` and 2 of 4 `postoj`: average clears 85%, postoj does not.
        pairs = [("spawanie", "spawanie")] * 18 + [("postoj", "postoj")] * 2
        pairs += [("postoj", "spawanie")] * 2
        scores = ev.per_activity_scores(pairs, ["spawanie", "postoj"])
        assert scores["spawanie"]["passes"] is True
        assert scores["postoj"]["recall"] == 0.5
        assert scores["postoj"]["passes"] is False

    def test_eighty_four_percent_fails(self):
        pairs = [("spawanie", "spawanie")] * 84 + [("spawanie", "postoj")] * 16
        scores = ev.per_activity_scores(pairs, ["spawanie"])
        assert scores["spawanie"]["recall"] == pytest.approx(0.84)
        assert scores["spawanie"]["passes"] is False

    def test_granularity_is_reported_for_thin_classes(self):
        pairs = [("postoj", "postoj")] * 2
        scores = ev.per_activity_scores(pairs, ["postoj"])
        # n=2 means one error moves the score 50 points - the figure cannot resolve 85%.
        assert scores["postoj"]["granularity_pp"] == pytest.approx(50.0)

    def test_precision_is_reported_separately_from_the_bar(self):
        # Over-calls spawanie: perfect recall, poor precision. The bar still passes,
        # which is exactly why precision has to be visible next to it.
        pairs = [("spawanie", "spawanie")] * 10 + [("postoj", "spawanie")] * 10
        scores = ev.per_activity_scores(pairs, ["spawanie", "postoj"])
        assert scores["spawanie"]["recall"] == 1.0
        assert scores["spawanie"]["precision"] == pytest.approx(0.5)
        assert scores["spawanie"]["passes"] is True

    def test_recall_bought_by_over_calling_is_not_usable(self):
        # The arc baseline's real failure mode: 100% recall by calling twice as
        # much time `spawanie` as happened. It clears the client's bar and must
        # still not be promoted.
        pairs = [("spawanie", "spawanie")] * 10 + [("postoj", "spawanie")] * 10
        s = ev.per_activity_scores(pairs, ["spawanie"])["spawanie"]
        assert s["time_ratio"] == pytest.approx(2.0)
        assert s["inflated"] is True
        assert s["passes"] is True  # the client's bar, reported unchanged
        assert s["usable"] is False  # but not a measurement

    def test_honest_arm_is_usable(self):
        pairs = [("spawanie", "spawanie")] * 18 + [("spawanie", "postoj")] * 2
        s = ev.per_activity_scores(pairs, ["spawanie"])["spawanie"]
        assert s["time_ratio"] == pytest.approx(0.9)
        assert s["inflated"] is False
        assert s["usable"] is True

    def test_under_reporting_is_not_flagged_as_gamed(self):
        # Under-reporting fails the bar on recall, which is the correct and
        # sufficient failure - it must not also be labelled as inflation.
        pairs = [("spawanie", "spawanie")] * 5 + [("spawanie", "postoj")] * 15
        s = ev.per_activity_scores(pairs, ["spawanie"])["spawanie"]
        assert s["inflated"] is False
        assert s["passes"] is False


class TestHardwareVerdict:
    def test_over_budget_is_disqualified(self):
        v = ev.hardware_verdict({"peak_vram_mib": 13000, "gpus_used": 1})
        assert v["verdict"] == "DISQUALIFIED"

    def test_two_cards_is_disqualified_even_when_small(self):
        v = ev.hardware_verdict({"peak_vram_mib": 4000, "gpus_used": 2})
        assert v["verdict"] == "DISQUALIFIED"

    def test_missing_gpu_block_is_unmeasured_not_ok(self):
        assert ev.hardware_verdict(None)["verdict"] == "UNMEASURED"
        assert ev.hardware_verdict({"box": "cctv-vps"})["verdict"] == "UNMEASURED"

    def test_within_budget_passes(self):
        v = ev.hardware_verdict({"peak_vram_mib": 7866, "gpus_used": 1})
        assert v["verdict"] == "OK"

    def test_cost_is_normalised_per_video_hour(self):
        gpu = {"gpu_seconds": 600, "video_seconds": 1200}
        assert ev.gpu_seconds_per_video_hour(gpu) == pytest.approx(1800.0)


class TestBoundaryErrors:
    def test_exact_boundaries_score_zero(self):
        truth = [
            {"activity_id": "a", "start_s": 0, "end_s": 10},
            {"activity_id": "b", "start_s": 10, "end_s": 20},
        ]
        b = ev.boundary_errors(truth, truth)
        assert b["median_s"] == 0
        assert b["spurious"] == 0

    def test_invented_boundaries_are_counted_as_spurious(self):
        truth = [
            {"activity_id": "a", "start_s": 0, "end_s": 10},
            {"activity_id": "b", "start_s": 10, "end_s": 20},
        ]
        pred = truth + [{"activity_id": "a", "start_s": 15, "end_s": 20}]
        b = ev.boundary_errors(truth, pred)
        assert b["spurious"] == 1

    def test_an_arm_emitting_no_boundaries_is_flagged(self):
        truth = [
            {"activity_id": "a", "start_s": 0, "end_s": 10},
            {"activity_id": "b", "start_s": 10, "end_s": 20},
        ]
        b = ev.boundary_errors(truth, [{"activity_id": "a", "start_s": 0, "end_s": 20}])
        assert b["matched"] == 0


class TestPredictionLoading:
    def test_intervals_and_samples_agree(self, tmp_path: Path):
        truth = _annotation("W1", ["spawanie", "spawanie", "postoj"])
        as_intervals = tmp_path / "a.json"
        as_intervals.write_text(
            json.dumps(
                {
                    "arm": "x",
                    "window": "W1",
                    "intervals": [
                        {"activity_id": "spawanie", "start_s": 0, "end_s": 3},
                        {"activity_id": "postoj", "start_s": 3, "end_s": 6},
                    ],
                }
            )
        )
        as_samples = tmp_path / "b.json"
        as_samples.write_text(
            json.dumps(
                {
                    "arm": "x",
                    "window": "W1",
                    "samples": [
                        {"t_s": 0, "activity_id": "spawanie"},
                        {"t_s": 2, "activity_id": "spawanie"},
                        {"t_s": 4, "activity_id": "postoj"},
                    ],
                }
            )
        )
        assert (
            ev.load_prediction(as_intervals, truth)["grid"]
            == (ev.load_prediction(as_samples, truth)["grid"])
        )

    def test_missing_prediction_is_an_error_not_a_smaller_denominator(self, tmp_path: Path):
        truth = _annotation("W1", ["spawanie", "spawanie", "postoj"])
        p = tmp_path / "partial.json"
        p.write_text(
            json.dumps(
                {
                    "arm": "x",
                    "window": "W1",
                    # Answers only the first sample.
                    "intervals": [{"activity_id": "spawanie", "start_s": 0, "end_s": 1}],
                }
            )
        )
        pred = ev.load_prediction(p, truth)
        pairs = [(gt, pred["grid"].get(t, "__brak_predykcji__")) for t, gt in truth["grid"].items()]
        scores = ev.per_activity_scores(pairs, ["spawanie"])
        # Support stays 2 - the unanswered sample counts against recall.
        assert scores["spawanie"]["support"] == 2
        assert scores["spawanie"]["recall"] == 0.5


class TestManifestSplit:
    def test_missing_split_block_refuses_to_score(self, tmp_path: Path):
        m = tmp_path / "manifest.json"
        m.write_text(json.dumps({"activities": [], "clips": []}))
        with pytest.raises(SystemExit):
            ev.load_manifest(m)


class TestArcBaseline:
    def test_threshold_is_tuned_per_clip(self):
        # Two clips with the same shape at different brightness: one fixed
        # threshold cannot serve both, which is the whole clip-relative caveat.
        dim = {t: (5.0 if t < 10 else 0.1) for t in range(20)}
        bright = {t: (50.0 if t < 10 else 1.0) for t in range(20)}
        truth = _annotation("W", ["spawanie"] * 5 + ["postoj"] * 5)
        t_dim, f1_dim = ev.tune_arc_threshold(dim, truth)
        t_bright, f1_bright = ev.tune_arc_threshold(bright, truth)
        assert f1_dim == pytest.approx(1.0)
        assert f1_bright == pytest.approx(1.0)
        assert t_bright > t_dim
