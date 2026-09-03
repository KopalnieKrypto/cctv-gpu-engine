"""Tests for the station model card reader (issue #123).

The card is the only place the station classifier learns what it is allowed to
report: which rectangle the head was fitted on, which classes it emits, how they
collapse into the delivered vocabulary — and, load-bearingly, the **measured time
ratio** for each delivered category.

The happy-path test reads the real shipped card rather than a synthetic one. A
parser that agrees with a fixture invented alongside it proves nothing about the
artefact production actually loads, and this project has already published a
figure that drifted loose from its measurement.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.station_card import StationCard, StationCardError

REPO_ROOT = Path(__file__).resolve().parent.parent
SHIPPED_CARD = REPO_ROOT / "models" / "station-head-hala-prawe-v1-v1.0.0.card.json"


def _shipped_document() -> dict:
    """The real card, as a mutable document to take one thing away from."""
    return json.loads(SHIPPED_CARD.read_text(encoding="utf-8"))


def _current_document() -> dict:
    """The real card brought up to the no-centre-crop contract.

    The shipped v1.0.0 card predates it: its `preprocessing` names a 518 resize
    and lets DINOv2's processor keep its own 224 `crop_size`, so the head read the
    middle 43% of the rectangle the card advertised. That card is now refused by
    construction (see :class:`TestACardFromBeforeTheCropWasRemoved`), so the
    happy-path tests read the real artefact and apply the one field a re-exported
    head will carry, rather than a document invented from nothing.
    """
    doc = _shipped_document()
    doc["backbone"]["preprocessing"] = {"model_input": [420, 882], "center_crop": False}
    return doc


class TestTheShippedCard:
    def test_it_carries_everything_the_section_reports(self) -> None:
        card = StationCard.from_dict(_current_document())

        assert card.version == "1.0.0"
        assert card.station_id == "hala-prawe-v1"
        assert card.stride_s == 2

        # The head emits seven classes and the delivered vocabulary is four; a
        # reader that knows only one of the two cannot collapse an argmax.
        assert card.model_outputs[0] == "spawanie"
        assert len(card.model_outputs) == 7
        assert card.delivered_classes == (
            "spawanie",
            "ukladanie_pretow",
            "pozostale",
            "nierozpoznane",
        )
        assert card.abstention == "nierozpoznane"

        # The measured over-reporting, per delivered category. A total rendered
        # without this is a number the client cannot use safely.
        assert card.time_ratios["spawanie"] == 1.0618279569892473
        assert set(card.time_ratios) == set(card.delivered_classes)

        assert card.window == 64
        assert card.model_input == (420, 882)

        assert [w.slot for w in card.training_windows] == ["W1", "W2", "W3"]
        assert card.training_windows[0].window_local == "2026-08-28 09:00-09:20 Europe/Warsaw"
        assert card.training_windows[0].samples == 599

    def test_the_rectangle_is_the_one_the_head_was_fitted_on(self) -> None:
        """x, y, w, h in native pixels — the field the inference path gates on.

        y is 1360 and not the 1400 the card shipped with. `crop=900:800:1700:1400`
        does not fit a 2160-tall frame, and ffmpeg's crop filter slides the
        rectangle back inside rather than failing, so every training crop was cut
        at `in_h - h`. The card was regenerated against the released weights
        (`--card-only`, sha256 unchanged) once that was measured.
        """
        assert StationCard.from_dict(_current_document()).zone_rect == (1700, 1360, 900, 800)


class TestACardWithAHoleInIt:
    """A missing time ratio stops the run; it never degrades to a bare total.

    The client-facing consequence is the whole reason: the same fixture produced
    a baseline that met a 99.4% recall bar while reporting 2.18x the real welding
    time. A duration printed without its measured error invites a decision the
    measurement does not support, so there is no path here that emits one.
    """

    def test_a_null_time_ratio_is_refused(self) -> None:
        doc = _current_document()
        doc["accuracy"]["union"]["ukladanie_pretow"]["time_ratio"] = None

        with pytest.raises(StationCardError) as exc:
            StationCard.from_dict(doc)

        assert "ukladanie_pretow" in str(exc.value)
        assert "time_ratio" in str(exc.value)

    def test_a_category_absent_from_the_measurements_is_refused(self) -> None:
        doc = _current_document()
        del doc["accuracy"]["union"]["pozostale"]

        with pytest.raises(StationCardError) as exc:
            StationCard.from_dict(doc)

        assert "pozostale" in str(exc.value)


class TestACardFromBeforeTheCropWasRemoved:
    """The v1.0.0 contract is refused, loudly, rather than reinterpreted.

    Until 2026-09-03 `preprocessing` named a `resize` and a `model_input`, and the
    gap between them was a centre-crop nobody chose: DINOv2's processor keeps its
    own `crop_size` default of 224, so a card saying "resize 518" produced a head
    that read the middle 43.2% of each axis — 18.7% of the rectangle's area. The
    card, the panel and `zone_native_px` all reported the whole rectangle.

    This build feeds the whole rectangle. Loading such a head anyway would score
    it on a framing and a scale it was never fitted on, and it would not fail
    while doing it — it would return confident logits, which is how a run came to
    report `spawanie: 0.0 s` across an hour of welding.
    """

    def test_the_real_shipped_card_no_longer_loads(self) -> None:
        with pytest.raises(StationCardError) as exc:
            StationCard.load(SHIPPED_CARD)

        assert "center_crop" in str(exc.value)

    def test_a_card_that_still_asks_for_a_centre_crop_is_refused(self) -> None:
        doc = _current_document()
        doc["backbone"]["preprocessing"]["center_crop"] = True

        with pytest.raises(StationCardError) as exc:
            StationCard.from_dict(doc)

        assert "center_crop" in str(exc.value)

    def test_a_card_without_a_tensor_size_is_refused(self) -> None:
        doc = _current_document()
        del doc["backbone"]["preprocessing"]["model_input"]

        with pytest.raises(StationCardError) as exc:
            StationCard.from_dict(doc)

        assert "model_input" in str(exc.value)
