"""Native-resolution frame tiling for pose detection (issue #110).

#101 established that the effective detection floor is ~60 px of person height
at model input, and that the 80-120 px native band — the largest bucket in the
`magazyn-hall-v1` ground truth — is 0% at every full-frame arm because a 3840 px
frame squeezed into a 1280 px input shrinks those people below the floor. Tiling
is the only lever left: crop the frame into overlapping **native-resolution**
tiles sized to the model input, detect each tile at full pixel scale, translate
detections back to full-frame coordinates, and merge across the overlaps.

The geometry and merge are pure; the per-tile pose call is supplied by an inner
detector object, so :class:`TiledPoseDetector` is testable without a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pipeline.postprocessing import Detection
from pipeline.preprocessing import InputSize

# Cross-tile duplicates are suppressed above this Intersection-over-Smaller. It
# is deliberately looser than the 0.45 NMS IoU used within a single frame: a
# person clipped by a tile seam produces a *partial* box, and IoS measures how
# much of the smaller box the larger one swallows — the right question for the
# containment that tiling creates, where IoU under-scores the pair.
DEFAULT_IOS_THRESHOLD = 0.6

# A candidate box that swallows an already-kept box is treated as that person's
# *whole body* (and the kept box as a clipped partial) only when it is at least
# this many times larger. Below the ratio the two boxes are near-equal — two
# people, one nested behind the other, or the same box twice — and greedy
# confidence order, not size, decides which survives. Tuned to sit under the
# full-frame-pass-vs-tile-partial area gap (a half-body partial is ~0.5x its
# whole box) while staying above the jitter between two views of one whole body.
DEFAULT_MIN_WHOLE_BOX_RATIO = 1.5


@dataclass(frozen=True)
class Tile:
    """A native-frame crop rectangle in ``xyxy`` pixel coordinates."""

    x1: int
    y1: int
    x2: int
    y2: int


def _axis_starts(length: int, tile: int, overlap: float) -> list[int]:
    """Start offsets so ``tile``-wide windows cover ``[0, length)`` with overlap.

    The last window is flush against the far edge rather than hanging past it,
    so every tile stays the same native size (no ragged, letterbox-padded edge
    tile) and no strip of the frame goes unscanned.
    """
    if tile >= length:
        return [0]
    step = max(1, round(tile * (1.0 - overlap)))
    starts = list(range(0, length - tile + 1, step))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def tile_grid(
    *,
    frame_w: int,
    frame_h: int,
    tile_w: int,
    tile_h: int,
    overlap: float,
) -> list[Tile]:
    """Slice a frame into overlapping native-resolution tiles.

    ``overlap`` is the fraction of a tile shared with its neighbour (0.2 = 20%),
    so a person straddling a tile boundary is whole inside at least one tile.
    Tiles are clamped to the frame; when the frame is smaller than a tile in an
    axis a single tile spans that axis.
    """
    x_starts = _axis_starts(frame_w, tile_w, overlap)
    y_starts = _axis_starts(frame_h, tile_h, overlap)
    return [
        Tile(
            x1=x,
            y1=y,
            x2=min(x + tile_w, frame_w),
            y2=min(y + tile_h, frame_h),
        )
        for y in y_starts
        for x in x_starts
    ]


def _tiles_overlap(tile: Tile, bounds: tuple[float, float, float, float]) -> bool:
    """Whether ``tile`` shares positive area with the ``xyxy`` ``bounds`` box."""
    bx1, by1, bx2, by2 = bounds
    return tile.x1 < bx2 and bx1 < tile.x2 and tile.y1 < by2 and by1 < tile.y2


def restrict_tiles_to_bounds(
    tiles: list[Tile],
    bounds: list[tuple[float, float, float, float]],
) -> list[Tile]:
    """Keep tiles that share pixels with any zone bounding box (with-zones arm).

    A tile only grazing a bbox edge shares zero area and carries no in-zone
    pixels, so it is dropped: the point of the with-zones arm is to not spend a
    pose call on frame regions no authored zone covers.
    """
    return [tile for tile in tiles if any(_tiles_overlap(tile, box) for box in bounds)]


def intersection_over_smaller(a: list[float], b: list[float]) -> float:
    """Intersection area over the smaller box's area, for two ``xyxy`` boxes.

    Unlike IoU this reaches 1.0 when one box sits fully inside the other — the
    signature of a person clipped by a tile seam whose partial box nests inside
    their whole box in the neighbouring tile.
    """
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    smaller = min(area_a, area_b)
    return inter / smaller if smaller > 0 else 0.0


def _bbox_area(bbox: list[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def merge_detections(
    detections: list[Detection],
    ios_threshold: float = DEFAULT_IOS_THRESHOLD,
    min_whole_box_ratio: float = DEFAULT_MIN_WHOLE_BOX_RATIO,
) -> list[Detection]:
    """Greedily dedup pooled tile detections, then drop cross-tile split duplicates.

    Phase 1 is the original greedy pass: highest confidence wins, and a candidate
    is suppressed when its IoS with an already-kept detection is at or above
    ``ios_threshold``. So the whole-body box beats the seam-clipped partial of the
    same person, and two distinct people standing close (neither containing the
    other) both survive.

    Phase 2 closes the precision leak #112 named: when a large person straddles a
    tile seam, each tile emits a *partial* box and a marginally-higher-confidence
    partial can be kept ahead of the whole-body box (from the full-frame pass),
    which is then suppressed — leaving two barely-overlapping partials of one
    person that phase 1 keeps as a double count. The suppressed whole box is the
    evidence they are one object: any suppressed box that contains two or more
    kept boxes, each at least ``min_whole_box_ratio``× smaller, marks those kept
    boxes as fragments of a single person. The strongest fragment stays; the rest
    are dropped. The coarse whole box is *not* reintroduced — measured worse at
    IoU 0.5 than the native-resolution fragment it contains (#112), so promoting
    it regresses the ≥260 px band rather than helping it.
    """
    kept: list[Detection] = []
    suppressed: list[Detection] = []
    for candidate in sorted(detections, key=lambda d: d.confidence, reverse=True):
        if all(intersection_over_smaller(candidate.bbox, k.bbox) < ios_threshold for k in kept):
            kept.append(candidate)
        else:
            suppressed.append(candidate)

    dropped: set[int] = set()
    for container in suppressed:
        container_area = _bbox_area(container.bbox)
        members = [
            det
            for det in kept
            if id(det) not in dropped
            and intersection_over_smaller(det.bbox, container.bbox) >= ios_threshold
            and container_area >= min_whole_box_ratio * _bbox_area(det.bbox)
        ]
        if len(members) >= 2:
            # Fragments of one seam-split person: keep the strongest, drop the
            # duplicates. ``>= 2`` guards a container that swallows a single box —
            # that is an ordinary contained partial phase 1 already handled, not a
            # split we should collapse further.
            members.sort(key=lambda d: d.confidence, reverse=True)
            for duplicate in members[1:]:
                dropped.add(id(duplicate))
    return [det for det in kept if id(det) not in dropped]


def _translate_detection(det: Detection, dx: int, dy: int) -> None:
    """Shift a tile-local detection into full-frame pixels, in place."""
    det.bbox[0] += dx
    det.bbox[1] += dy
    det.bbox[2] += dx
    det.bbox[3] += dy
    for keypoint in det.keypoints:
        keypoint.x += dx
        keypoint.y += dy


@dataclass
class TiledPoseDetector:
    """Run an inner single-tile pose detector across a native-resolution grid.

    Presents the same ``detect(img_bgr) -> list[Detection]`` surface as
    :class:`pipeline.pose_detector.PoseDetector`, so the benchmark harness and
    pipeline drive it unchanged. ``detector`` is any object exposing that method
    plus ``input_size`` — in production a zone-less ``PoseDetector`` whose model
    input equals the tile size, so each tile letterboxes at scale 1.0 and people
    keep their native pixel height.

    ``zone_bounds`` (with-zones arm) restricts inference to tiles that intersect
    an authored zone's bounding box; ``None`` tiles the whole frame.

    ``full_frame_pass`` (hybrid arm) adds one whole-frame pose call whose
    detections join the tile pool before merging. A person too large to fit one
    tile is split into seam-clipped partials by the grid; the whole-frame pass
    frames them once, and the higher-confidence whole box suppresses the partials
    in the merge — recovering the large near-field people grid-only tiling loses.
    """

    detector: Any
    tile_w: int
    tile_h: int
    overlap: float
    zone_bounds: list[tuple[float, float, float, float]] | None = None
    ios_threshold: float = DEFAULT_IOS_THRESHOLD
    full_frame_pass: bool = False
    min_whole_box_ratio: float = DEFAULT_MIN_WHOLE_BOX_RATIO

    @property
    def input_size(self) -> InputSize:
        """The inner model's input shape, so the harness's shape guard passes."""
        return self.detector.input_size

    def detect(self, img_bgr: np.ndarray) -> list[Detection]:
        frame_h, frame_w = img_bgr.shape[:2]
        tiles = tile_grid(
            frame_w=frame_w,
            frame_h=frame_h,
            tile_w=self.tile_w,
            tile_h=self.tile_h,
            overlap=self.overlap,
        )
        if self.zone_bounds is not None:
            tiles = restrict_tiles_to_bounds(tiles, self.zone_bounds)
        pooled: list[Detection] = []
        if self.full_frame_pass:
            # The whole frame letterboxed to the tile size — offset (0, 0), so its
            # detections are already in full-frame coordinates. Pooled before the
            # tiles so the whole-body box, kept on confidence, wins the merge.
            pooled.extend(self.detector.detect(img_bgr))
        for tile in tiles:
            crop = img_bgr[tile.y1 : tile.y2, tile.x1 : tile.x2]
            for det in self.detector.detect(crop):
                _translate_detection(det, tile.x1, tile.y1)
                pooled.append(det)
        return merge_detections(pooled, self.ios_threshold, self.min_whole_box_ratio)
