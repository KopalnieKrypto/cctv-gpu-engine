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


class StationCardError(ValueError):
    """Raised when a card cannot support the totals that would be quoted from it.

    A subclass of :class:`ValueError` so callers already catching bad input keep
    working. Raised for a malformed card and — the case that matters — for a card
    whose measured time ratios have a hole in them. A blank there is worse than a
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
    resize_px: int
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
            resize_px=int(preprocessing["resize"]),
            model_input=tuple(int(v) for v in preprocessing["model_input"]),  # type: ignore[arg-type]
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
