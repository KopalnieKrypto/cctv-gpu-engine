"""The station model card, read rather than assumed (issue #123).

The card that ships beside a station head (issue #122) is the classifier's whole
contract: the rectangle the head was fitted on, the classes it emits, how those
collapse into the delivered vocabulary, and the **measured time ratio** for each
delivered category. Nothing in this module invents a number — every figure is
read out of the card, which was itself generated from ``C0-report.json`` rather
than typed.

That indirection exists because this project has already lost a figure to the
gap: 98.5% for one class reached a client report after being copied by hand out
of a detector docstring. The card is generated so no hand can touch it, and this
reader refuses a card with a hole in it for the same reason.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# The rectangle the head was fitted on, in native source pixels: (x, y, w, h).
Rect = tuple[int, int, int, int]


class StationError(ValueError):
    """Base for every reason the station path refuses to produce numbers.

    Lives here because this is the bottom of the station import stack, so every
    other station module can subclass it without a cycle. A subclass of
    :class:`ValueError` so callers already catching bad input keep working, and
    one type the CLI can catch to turn any of them into a sentence rather than a
    traceback.
    """


class StationCardError(StationError):
    """Raised when a card cannot support the totals that would be quoted from it.

    Raised for a malformed card and — the case that matters — for a card whose
    measured time ratios have a hole in them. A blank there is worse than a
    missing card: the reader fills it in from somewhere else, which is exactly how
    a detector's in-sample hit rate once ended up quoted as held-out accuracy.
    """


def _time_ratios(union: dict[str, Any], delivered: tuple[str, ...]) -> dict[str, float]:
    """The measured time ratio for every delivered category, or refuse.

    ``time_ratio`` is predicted seconds over true seconds, measured on the
    cross-validated folds. It is what turns "3 h 42 min of welding" into a figure
    a work-study client can act on, so a category without one takes the whole
    section down rather than emitting a total that looks exact.
    """
    ratios: dict[str, float] = {}
    for category in delivered:
        scores = union.get(category)
        if scores is None:
            raise StationCardError(
                f"the card quotes no measurement for the delivered category "
                f"`{category}`. Every total is reported with its measured time "
                "ratio, so a category the folds never scored cannot be reported."
            )
        ratio = scores.get("time_ratio")
        if ratio is None:
            raise StationCardError(
                f"`{category}` has no measured time_ratio in the card. A duration "
                "without its measured over-reporting invites a decision the "
                "measurement does not support, so no totals are emitted at all."
            )
        ratios[category] = float(ratio)
    return ratios


def _model_input(preprocessing: dict[str, Any]) -> tuple[int, int]:
    """The ``(height, width)`` the rectangle is resized to, or refuse the card.

    A card must state ``center_crop: false`` to be loaded at all, and that is not
    ceremony. Until 2026-09-03 the preprocessing named a 518 resize and let the
    processor's own ``crop_size`` default to 224, so the head read the middle
    43.2% of each axis while the card, the panel and `zone_native_px` all
    reported the full rectangle. Nothing failed; a run just reported
    ``spawanie: 0.0 s`` over an hour of welding.

    So a card carrying the old two-step contract - or one that quietly omits the
    flag - is refused rather than interpreted. The alternative is a head fitted
    on centre-crops being fed whole rectangles, which produces confident numbers
    about a feature space it never saw.

    Raises:
        StationCardError: when the flag is absent or true, or when
            ``model_input`` is not a usable pair.
    """
    if preprocessing.get("center_crop") is not False:
        raise StationCardError(
            "the card's `backbone.preprocessing` does not declare "
            "`center_crop: false`. Cards written before 2026-09-03 resized to "
            "`resize` and then centre-cropped to `model_input`, so the head saw "
            "the middle 43% of the station rectangle and not the rectangle. This "
            "build feeds the whole rectangle, so such a head would be scored on "
            "pixels at a scale and framing it was never fitted on. Re-export the "
            "head and its card."
        )
    try:
        height, width = (int(v) for v in preprocessing["model_input"])
    except (KeyError, TypeError, ValueError) as exc:
        raise StationCardError(
            "the card's `backbone.preprocessing.model_input` is not a "
            "`[height, width]` pair, so there is no tensor size to resize the "
            "station rectangle to."
        ) from exc
    if height < 1 or width < 1:
        raise StationCardError(
            f"the card's `model_input` is {height}x{width}, which is not an image."
        )
    return (height, width)


@dataclass(frozen=True)
class TrainingWindow:
    """One annotated recording the shipped weights were trained on."""

    slot: str
    window_local: str | None
    annotated_at: str | None
    samples: int | None


@dataclass(frozen=True)
class StationCard:
    """What a station head is allowed to claim about itself."""

    version: str
    station_id: str
    zone_rect: Rect
    stride_s: int
    model_outputs: tuple[str, ...]
    delivered_classes: tuple[str, ...]
    bucket: str
    bucket_members: frozenset[str]
    abstention: str
    time_ratios: dict[str, float]
    window: int
    #: ``(height, width)`` the whole station rectangle is resized to. One number
    #: pair, not two steps: there is no separate resize target any more, because a
    #: second step is what hid 57% of the rectangle for the life of v1.0.0.
    model_input: tuple[int, int]
    training_windows: tuple[TrainingWindow, ...]

    def deliver(self, model_output: str) -> str:
        """Map one of the head's classes to the category the client is shown.

        The collapse happens **after** argmax, by the consumer — the shipped head
        is seven-class and the delivered vocabulary is four. Summing logits across
        the bucket's members before the argmax would change which class wins.

        ``nierozpoznane`` is never a member: the manifest defines it as neither
        work nor downtime, and folding the honest cannot-tell into a collective
        work bucket would convert unknown time into measured time.
        """
        return self.bucket if model_output in self.bucket_members else model_output

    @classmethod
    def load(cls, path: str | Path) -> StationCard:
        """Read and validate the card at ``path``."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StationCard:
        """Build a card from an already-parsed document."""
        station = data["station"]
        crop = station["zone_native_px"]
        vocabulary = data["vocabulary"]
        delivered = tuple(vocabulary["delivered_classes"])
        preprocessing = data["backbone"]["preprocessing"]
        union = data["accuracy"]["union"]
        return cls(
            version=data["artifact"]["version"],
            station_id=station["id"],
            zone_rect=(int(crop["x"]), int(crop["y"]), int(crop["w"]), int(crop["h"])),
            stride_s=int(station["stride_s"]),
            model_outputs=tuple(vocabulary["model_outputs"]),
            delivered_classes=delivered,
            bucket=vocabulary["bucket"],
            bucket_members=frozenset(vocabulary["bucket_contains"]),
            abstention=vocabulary["not_in_bucket"],
            time_ratios=_time_ratios(union, delivered),
            window=int(data["head"]["hyperparameters"]["window"]),
            model_input=_model_input(preprocessing),
            training_windows=tuple(
                TrainingWindow(
                    slot=w["slot"],
                    window_local=w.get("window_local"),
                    annotated_at=w.get("annotated_at"),
                    samples=w.get("samples"),
                )
                for w in data["training_material"]["windows"]
            ),
        )
