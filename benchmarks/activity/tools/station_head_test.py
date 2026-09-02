"""Contract tests for the shipped station head and its model card (#122).

The card exists because this project has already published a figure with nothing
behind it - 98.5% for one class reached a client report after being copied by
hand from a detector docstring. So these pin the rules that keep a quoted number
attached to its measurement: every figure is read from `C0-report.json`, the card
says out loud that the shipped weights were never held out from anything, and a
missing measurement is an error rather than a blank.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

import station_head as sh  # noqa: E402

ACTIVITY_IDS = [
    "spawanie",
    "ukladanie_pretow",
    "sciaganie_elementu",
    "inna_czynnosc",
    "postoj",
    "brak_na_stanowisku",
    "nierozpoznane",
]

COLLAPSE = {
    "bucket": "pozostale",
    "members": ["sciaganie_elementu", "inna_czynnosc", "postoj", "brak_na_stanowisku"],
}


class TestVocabularyRemap:
    def test_labels_remap_onto_the_delivered_classes(self):
        # `y` indexes the manifest's activity order. Training on three categories
        # means remapping those indices, never re-reading the annotation - the
        # pixels and the samples are identical, only the label space changes.
        y = np.array([ACTIVITY_IDS.index(a) for a in ACTIVITY_IDS], dtype=np.int64)
        y2, delivered = sh.remap_labels(y, ACTIVITY_IDS, COLLAPSE)
        assert delivered == ["spawanie", "ukladanie_pretow", "pozostale", "nierozpoznane"]
        assert [delivered[i] for i in y2] == [
            "spawanie",
            "ukladanie_pretow",
            "pozostale",
            "pozostale",
            "pozostale",
            "pozostale",
            "nierozpoznane",
        ]


def _figs(**kw) -> dict:
    """`{category: {recall, time_ratio}}` in the shape C0-report.json carries."""
    return {c: {"recall": r, "time_ratio": t} for c, (r, t) in kw.items()}


class TestShippingDecision:
    def test_worse_recall_on_one_category_ships_the_collapsed_model(self):
        # #121's collapsed figures are the floor. Training on three categories has
        # to beat them everywhere it is quoted, or the thing that already works
        # ships - the burden of proof is on the new model, not on the client.
        collapsed = _figs(
            spawanie=(0.89, 1.06), ukladanie_pretow=(0.88, 1.10), pozostale=(0.45, 0.84)
        )
        direct = _figs(spawanie=(0.91, 1.02), ukladanie_pretow=(0.83, 1.05), pozostale=(0.52, 0.95))
        decision = sh.choose_vocabulary(direct, collapsed)
        assert decision["ships"] == "collapsed"
        assert "ukladanie_pretow" in decision["regressions"]

    def test_beating_the_floor_everywhere_ships_the_direct_model(self):
        collapsed = _figs(
            spawanie=(0.89, 1.06), ukladanie_pretow=(0.88, 1.10), pozostale=(0.45, 0.84)
        )
        direct = _figs(spawanie=(0.91, 1.02), ukladanie_pretow=(0.90, 1.05), pozostale=(0.52, 0.95))
        decision = sh.choose_vocabulary(direct, collapsed)
        assert decision["ships"] == "direct"
        assert decision["regressions"] == {}

    def test_recall_gained_by_drifting_the_reported_time_is_a_regression(self):
        # The failure this whole harness exists to catch: recall bought by
        # over-calling the class. A better recall next to a worse time ratio is
        # not an improvement a chronometraz client can use.
        collapsed = _figs(
            spawanie=(0.89, 1.06), ukladanie_pretow=(0.88, 1.10), pozostale=(0.45, 0.84)
        )
        direct = _figs(spawanie=(0.95, 1.40), ukladanie_pretow=(0.90, 1.05), pozostale=(0.52, 0.95))
        decision = sh.choose_vocabulary(direct, collapsed)
        assert decision["ships"] == "collapsed"
        assert decision["regressions"]["spawanie"] == ["time_ratio"]

    def test_under_reporting_counts_as_drift_too(self):
        # 0.60x is as far from the truth as 1.40x. The card quotes the ratio,
        # not its sign.
        collapsed = _figs(spawanie=(0.89, 1.06))
        direct = _figs(spawanie=(0.95, 0.60))
        assert sh.choose_vocabulary(direct, collapsed)["ships"] == "collapsed"

    def test_the_abstention_is_not_part_of_the_verdict(self):
        # `nierozpoznane` is not an activity - the manifest says so, and every
        # other verdict in this fixture excludes it.
        collapsed = _figs(spawanie=(0.89, 1.06), nierozpoznane=(0.30, 1.00))
        direct = _figs(spawanie=(0.91, 1.02), nierozpoznane=(0.00, 3.00))
        assert sh.choose_vocabulary(direct, collapsed)["ships"] == "direct"

    def test_a_category_the_direct_model_never_predicts_is_a_regression(self):
        collapsed = _figs(spawanie=(0.89, 1.06), pozostale=(0.45, 0.84))
        direct = _figs(spawanie=(0.91, 1.02))
        decision = sh.choose_vocabulary(direct, collapsed)
        assert decision["ships"] == "collapsed"
        assert "pozostale" in decision["regressions"]


def _score(recall: float, ratio: float, support: int) -> dict:
    return {"recall": recall, "time_ratio": ratio, "support": support, "precision": 0.8}


def _report(arm: str = "tcn-pixel-518", *, collapsed: bool = True) -> dict:
    """The shape `evaluate_arms.py --json-out` writes."""
    union = {
        "spawanie": _score(0.890, 1.06, 744),
        "ukladanie_pretow": _score(0.882, 1.10, 552),
        "pozostale": _score(0.451, 0.84, 441),
        "nierozpoznane": _score(0.0, 0.50, 62),
    }
    per_window = {
        w: {c: _score(s["recall"], s["time_ratio"], s["support"] // 3) for c, s in union.items()}
        for w in ("W1", "W2", "W3")
    }
    entry = {"name": arm, "scores": {}, "per_window_scores": {}}
    if collapsed:
        entry["collapsed"] = {"scores": union, "per_window_scores": per_window}
    return {
        "generated": "2026-09-02",
        "arms": [entry],
        "collapse": {**COLLAPSE, "classes": [*ACTIVITY_IDS[:2], "pozostale", "nierozpoznane"]},
    }


def _manifest() -> dict:
    return {
        "fixture": "hala-prawe-v1",
        "activities": [{"id": a} for a in ACTIVITY_IDS],
        "camera": {"id": "c88d18d9", "pose": {"input_size": "1280x736"}},
        "station_roi": {"crop": {"w": 900, "h": 800, "x": 1700, "y": 1400}},
        "delivery_vocabulary": COLLAPSE,
        "clips": [
            {
                "slot": w,
                "annotated": True,
                "annotated_at": "2026-09-01",
                "window_local": f"2026-08-28 {h}:00-{h}:20 Europe/Warsaw",
                "annotation_coverage": {"labelled": n, "total": n, "stride_s": 2},
            }
            for w, h, n in (("W1", "09", 599), ("W2", "10", 600), ("W3", "18", 600))
        ],
    }


TRAINING = {
    "backbone": "facebook/dinov2-base",
    "image_size": 518,
    "hyperparameters": {"window": 64, "channels": 96, "epochs": 60, "seed": 117},
    "trained_as": "collapsed",
    "artifact": {"name": "station-head-hala-prawe-v1.0.0.onnx", "version": "1.0.0"},
}


class TestModelCard:
    def test_union_figures_are_read_from_the_report(self):
        # Not typed, not passed in: the only place a number may come from is the
        # committed measurement, read at generation time.
        card = sh.build_card(_manifest(), _report(), "tcn-pixel-518", TRAINING)
        assert card["accuracy"]["union"]["spawanie"]["recall"] == 0.890
        assert card["accuracy"]["union"]["pozostale"]["time_ratio"] == 0.84
        assert card["accuracy"]["arm"] == "tcn-pixel-518"

    def test_no_matching_arm_in_the_report_is_an_error(self):
        # The 98.5% that reached a client report had no arm behind it. A card
        # whose arm is absent from the measurement must not be buildable.
        with pytest.raises(SystemExit):
            sh.build_card(_manifest(), _report(arm="something-else"), "tcn-pixel-518", TRAINING)

    def test_an_arm_scored_on_seven_categories_only_is_an_error(self):
        # No `collapsed` block means the report predates #121 or was generated
        # without the delivery vocabulary. Quoting its seven-category figures for
        # a three-category head is exactly the mix-up #121 forbids.
        with pytest.raises(SystemExit):
            sh.build_card(_manifest(), _report(collapsed=False), "tcn-pixel-518", TRAINING)

    def test_a_missing_figure_fails_rather_than_emitting_a_blank(self):
        report = _report()
        report["arms"][0]["collapsed"]["scores"]["pozostale"]["recall"] = None
        with pytest.raises(SystemExit):
            sh.build_card(_manifest(), report, "tcn-pixel-518", TRAINING)

    def test_card_states_the_shipped_weights_were_never_held_out(self):
        # The shipped head trains on every annotated window, so it cannot be
        # scored at all. The card has to say that next to the numbers, or the
        # numbers read as if they described these weights.
        card = sh.build_card(_manifest(), _report(), "tcn-pixel-518", TRAINING)
        material = card["training_material"]
        assert material["held_out"] == []
        assert material["all_annotated_material_used"] is True
        assert card["accuracy"]["from_these_weights"] is False
        assert "cross-valid" in card["accuracy"]["measured_on"].lower()

    def test_card_lists_the_windows_it_trained_on_with_their_counts(self):
        card = sh.build_card(_manifest(), _report(), "tcn-pixel-518", TRAINING)
        windows = {w["slot"]: w for w in card["training_material"]["windows"]}
        assert set(windows) == {"W1", "W2", "W3"}
        assert windows["W1"]["samples"] == 599
        assert windows["W3"]["samples"] == 600

    def test_card_carries_per_window_figures_not_only_the_union(self):
        # The folds disagree on this fixture - `pozostale` runs 27% / 72% / 52%
        # across the three windows. A card quoting only the union would describe
        # a station that behaves consistently, which this one does not.
        card = sh.build_card(_manifest(), _report(), "tcn-pixel-518", TRAINING)
        per_window = card["accuracy"]["per_window"]
        assert set(per_window) == {"W1", "W2", "W3"}
        for window in per_window.values():
            assert set(window) >= {"spawanie", "ukladanie_pretow", "pozostale"}
            assert window["spawanie"]["recall"] is not None
            assert window["spawanie"]["time_ratio"] is not None

    def test_a_missing_per_window_figure_also_fails(self):
        report = _report()
        report["arms"][0]["collapsed"]["per_window_scores"]["W2"]["spawanie"]["time_ratio"] = None
        with pytest.raises(SystemExit):
            sh.build_card(_manifest(), report, "tcn-pixel-518", TRAINING)

    def test_card_identifies_the_station_and_its_zone(self):
        card = sh.build_card(_manifest(), _report(), "tcn-pixel-518", TRAINING)
        station = card["station"]
        assert station["id"] == "hala-prawe-v1"
        assert station["camera_id"] == "c88d18d9"
        assert station["zone_native_px"] == {"w": 900, "h": 800, "x": 1700, "y": 1400}
        assert station["stride_s"] == 2

    def test_windows_that_disagree_on_the_stride_are_an_error(self):
        # The head's receptive field is counted in samples. If two windows were
        # sampled at different strides there is no single "the stride" to state,
        # and 64 frames would mean two different spans of time.
        manifest = _manifest()
        manifest["clips"][1]["annotation_coverage"]["stride_s"] = 4
        with pytest.raises(SystemExit):
            sh.build_card(manifest, _report(), "tcn-pixel-518", TRAINING)

    def test_card_states_the_vocabulary_and_what_the_bucket_contains(self):
        card = sh.build_card(_manifest(), _report(), "tcn-pixel-518", TRAINING)
        vocab = card["vocabulary"]
        assert vocab["classes"] == ["spawanie", "ukladanie_pretow", "pozostale", "nierozpoznane"]
        assert vocab["bucket"] == "pozostale"
        assert vocab["bucket_contains"] == COLLAPSE["members"]
        assert vocab["not_in_bucket"] == "nierozpoznane"

    def test_card_records_the_backbone_and_the_head_hyperparameters(self):
        # "Annotate twenty minutes, train a head" only holds if the head can be
        # rebuilt from what the card says.
        card = sh.build_card(_manifest(), _report(), "tcn-pixel-518", TRAINING)
        assert card["backbone"]["id"] == "facebook/dinov2-base"
        assert card["backbone"]["image_size"] == 518
        assert card["backbone"]["frozen"] is True
        assert card["head"]["hyperparameters"]["seed"] == 117

    def test_card_carries_the_versioned_artefact_name(self):
        # A station has to be rollable back to a previous head, so the name
        # carries a version like `activity-mlp-v1.0.0.onnx` already does.
        card = sh.build_card(_manifest(), _report(), "tcn-pixel-518", TRAINING)
        assert card["artifact"]["version"] == "1.0.0"
        assert "1.0.0" in card["artifact"]["name"]
        assert card["artifact"]["name"].endswith(".onnx")

    def test_card_records_the_vocabulary_comparison_and_which_one_shipped(self):
        # The comparison against #121's collapsed floor is the reason one of the
        # two models is in the file, so it belongs in the card rather than in a
        # commit message nobody reads next to the weights.
        training = {
            **TRAINING,
            "comparison": sh.choose_vocabulary(
                _figs(spawanie=(0.80, 1.06), ukladanie_pretow=(0.90, 1.05)),
                _figs(spawanie=(0.89, 1.06), ukladanie_pretow=(0.88, 1.10)),
            ),
        }
        card = sh.build_card(_manifest(), _report(), "tcn-pixel-518", training)
        comparison = card["vocabulary"]["comparison"]
        assert comparison["ships"] == "collapsed"
        assert "spawanie" in comparison["regressions"]
        assert comparison["deltas"]["spawanie"]["recall"] < 0

    def test_the_card_never_ships_the_option_that_lost(self):
        # The decision and the artefact cannot disagree: whichever model the
        # comparison rejected must not be the one the card describes.
        training = {
            **TRAINING,
            "trained_as": "direct",
            "comparison": sh.choose_vocabulary(
                _figs(spawanie=(0.80, 1.06)), _figs(spawanie=(0.89, 1.06))
            ),
        }
        with pytest.raises(SystemExit):
            sh.build_card(_manifest(), _report(), "tcn-pixel-518", training)
