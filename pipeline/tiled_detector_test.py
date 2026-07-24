"""Behavior tests for native-resolution frame tiling (issue #110).

Tiling is the only detection lever left for the 80-120 px person band that #101
proved is 0% at every full-frame arm. The unit here is the geometry and merge
that turn one 4K frame into overlapping native tiles and back into deduplicated
full-frame detections; the GPU pose call is a boundary these tests inject.
"""

from __future__ import annotations

import numpy as np

from pipeline.postprocessing import Detection, Keypoint
from pipeline.tiled_detector import (
    DEFAULT_TILE_OVERLAP,
    Tile,
    TiledPoseDetector,
    build_hybrid_detector,
    intersection_over_smaller,
    merge_detections,
    restrict_tiles_to_bounds,
    tile_grid,
)


def test_tile_grid_covers_a_4k_frame_with_native_sized_overlapping_tiles():
    tiles = tile_grid(
        frame_w=3840,
        frame_h=2160,
        tile_w=1280,
        tile_h=736,
        overlap=0.2,
    )

    # Every tile is a native-resolution 1280x736 window fully inside the frame:
    # native means the person keeps their pixel height, which is the whole point.
    for tile in tiles:
        assert (tile.x2 - tile.x1, tile.y2 - tile.y1) == (1280, 736)
        assert 0 <= tile.x1 and tile.x2 <= 3840
        assert 0 <= tile.y1 and tile.y2 <= 2160

    # The grid reaches all four corners, so no strip of the frame is unscanned.
    assert (0, 0) == (tiles[0].x1, tiles[0].y1)
    assert max(tile.x2 for tile in tiles) == 3840
    assert max(tile.y2 for tile in tiles) == 2160


def test_tile_grid_neighbours_overlap_so_a_boundary_person_is_whole_in_one_tile():
    tiles = tile_grid(frame_w=3840, frame_h=2160, tile_w=1280, tile_h=736, overlap=0.2)

    xs = sorted({tile.x1 for tile in tiles})
    # Consecutive columns step by less than a tile width: the shared band is
    # where a person split by one seam lands whole inside the neighbour.
    steps = [b - a for a, b in zip(xs, xs[1:], strict=False)]
    assert steps and all(0 < step < 1280 for step in steps)
    # The requested 20% overlap is a floor, not exact: the flush-to-edge last
    # column can only widen the overlap, never shrink it below the target.
    assert max(steps) <= round(1280 * (1 - 0.2))


def test_tile_grid_returns_one_clamped_tile_when_the_frame_is_smaller_than_a_tile():
    tiles = tile_grid(frame_w=800, frame_h=600, tile_w=1280, tile_h=736, overlap=0.2)

    assert tiles == [Tile(0, 0, 800, 600)]


def test_restrict_tiles_keeps_only_tiles_touching_a_zone_bbox():
    # The with-zones arm focuses compute: tile only where an authored zone lives,
    # spending nothing on frame regions no worker can occupy.
    tiles = [
        Tile(0, 0, 100, 100),  # fully above-left of the zone -> dropped
        Tile(140, 140, 240, 240),  # clips the zone's top-left corner -> kept
        Tile(200, 200, 300, 300),  # inside the zone -> kept
        Tile(500, 500, 600, 600),  # fully below-right of the zone -> dropped
    ]

    kept = restrict_tiles_to_bounds(tiles, [(150, 150, 400, 400)])

    assert kept == [Tile(140, 140, 240, 240), Tile(200, 200, 300, 300)]


def test_restrict_tiles_treats_edge_touching_tiles_as_outside():
    # A tile whose edge only grazes the bbox line shares zero area with the zone,
    # so it carries no in-zone pixels and is not worth a pose call.
    tiles = [Tile(100, 0, 200, 100)]

    assert restrict_tiles_to_bounds(tiles, [(0, 0, 100, 100)]) == []


def _det(bbox, confidence):
    return Detection(
        bbox=list(bbox), confidence=confidence, keypoints=[Keypoint(0.0, 0.0, 0.0)] * 17
    )


def test_intersection_over_smaller_is_one_when_a_box_contains_another():
    # The dedup case tiling creates: a person clipped by a tile seam yields a
    # small partial box sitting inside their full box in the neighbour tile. IoU
    # would score that pair low (0.04 here) and keep both; IoS scores it 1.0.
    outer = [0, 0, 100, 100]
    inner = [10, 10, 30, 30]

    assert intersection_over_smaller(outer, inner) == 1.0
    assert intersection_over_smaller(inner, outer) == 1.0


def test_intersection_over_smaller_is_zero_for_disjoint_boxes():
    assert intersection_over_smaller([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_merge_dedups_a_person_split_across_two_overlapping_tiles():
    # Same person: whole box (from the tile that framed them) plus a clipped
    # partial box (from the neighbour that cut them at the seam). One survives.
    whole = _det([100, 100, 160, 260], 0.90)
    clipped = _det([100, 100, 130, 260], 0.55)  # left half, cut by the seam
    other_person = _det([900, 900, 960, 1060], 0.80)

    merged = merge_detections([clipped, whole, other_person], ios_threshold=0.6)

    assert [d.bbox for d in merged] == [[100, 100, 160, 260], [900, 900, 960, 1060]]
    assert [d.confidence for d in merged] == [0.90, 0.80]


def test_merge_keeps_both_when_overlap_stays_under_the_threshold():
    # Two workers standing close: boxes touch but neither contains the other, so
    # IoS stays below threshold and both people are counted.
    left = _det([0, 0, 100, 200], 0.9)
    right = _det([90, 0, 190, 200], 0.8)  # 10% width overlap

    merged = merge_detections([left, right], ios_threshold=0.6)

    assert len(merged) == 2


def test_merge_removes_a_cross_tile_duplicate_that_ios_alone_leaves_uncaught():
    # The un-caught cross-tile duplicate (#112): a big near-field person spans a
    # seam and is clipped into a left and a right partial by two tiles; the two
    # partials barely overlap each other (IoS well under threshold), so IoS-only
    # greedy keeps BOTH — one person counted twice, the precision leak. The
    # full-frame pass frames them whole, but at a lower score than the strongest
    # partial, so greedy suppresses the whole box. That suppressed whole box is
    # the evidence the two partials are one person: it contains both, each much
    # smaller, so the weaker partial is dropped as a duplicate and one detection
    # remains — the strongest, native-resolution fragment, not the coarse whole.
    partial_left = _det([1000, 500, 1160, 1400], 0.92)  # strongest, a clipped half
    whole = _det([1000, 500, 1300, 1400], 0.88)  # full-frame pass: contains both halves
    partial_right = _det([1140, 500, 1300, 1400], 0.80)  # the other tile's half

    merged = merge_detections([partial_left, whole, partial_right], ios_threshold=0.6)

    assert [d.bbox for d in merged] == [[1000, 500, 1160, 1400]]
    assert [d.confidence for d in merged] == [0.92]


def test_merge_keeps_the_native_partial_over_a_lower_confidence_whole_box():
    # The big-person whole-box-vs-partial case (#112). The issue floated promoting
    # the whole box over a higher-confidence partial, but the measured benchmark
    # says the coarse full-frame-pass whole box matches the GT *worse* at IoU 0.5
    # than the native-resolution tile partial, so promoting it regresses the
    # ≥260 px band. With no second fragment to mark a duplicate, the single
    # contained partial is left exactly as greedy chose it — the whole box is
    # suppressed, never promoted.
    partial = _det([0, 0, 90, 180], 0.90)  # native tile half, higher confidence
    whole = _det([0, 0, 180, 180], 0.85)  # coarse full-frame box, contains it

    merged = merge_detections([partial, whole], ios_threshold=0.6)

    assert [d.bbox for d in merged] == [[0, 0, 90, 180]]
    assert [d.confidence for d in merged] == [0.90]


class _FakeTileDetector:
    """A per-tile pose call stand-in: returns canned crop-local detections.

    Tiles are visited row-major, so the Nth ``detect`` call is the Nth tile;
    each entry in ``per_call`` is that tile's crop-local detection list.
    """

    input_size = (100, 100)

    def __init__(self, per_call):
        self.per_call = [list(dets) for dets in per_call]
        self.crops = []

    def detect(self, crop):
        self.crops.append(crop.shape[:2])
        return self.per_call[len(self.crops) - 1]


def _kp_det(bbox, confidence, keypoint):
    kx, ky = keypoint
    return Detection(bbox=list(bbox), confidence=confidence, keypoints=[Keypoint(kx, ky, 1.0)])


def test_detect_translates_each_tile_back_to_full_frame_coordinates():
    # Two disjoint tiles (200x100 frame, 100x100 tiles, no overlap). Each tile's
    # crop-local detection must land at its tile offset in the full frame, and
    # keypoints move with the box.
    fake = _FakeTileDetector(
        per_call=[
            [_kp_det([10, 10, 30, 30], 0.9, (20, 20))],  # tile at x=0
            [_kp_det([10, 10, 30, 30], 0.8, (20, 20))],  # tile at x=100
        ]
    )
    detector = TiledPoseDetector(detector=fake, tile_w=100, tile_h=100, overlap=0.0)
    frame = np.zeros((100, 200, 3), dtype=np.uint8)

    detections = detector.detect(frame)

    assert [d.bbox for d in detections] == [[10, 10, 30, 30], [110, 10, 130, 30]]
    assert [(d.keypoints[0].x, d.keypoints[0].y) for d in detections] == [(20, 20), (120, 20)]
    assert fake.crops == [(100, 100), (100, 100)]  # both tiles were inferred


def test_detect_merges_the_same_person_seen_in_two_overlapping_tiles():
    # 150x100 frame, 100x100 tiles, 50% overlap -> tiles at x=0 and x=50 share
    # the band [50,100). A worker standing there is framed whole by both tiles;
    # after translation both boxes land on the same full-frame pixels.
    fake = _FakeTileDetector(
        per_call=[
            [_kp_det([60, 10, 90, 90], 0.90, (75, 50))],  # tile x=0 -> [60,10,90,90]
            [_kp_det([10, 10, 40, 90], 0.70, (25, 50))],  # tile x=50 -> [60,10,90,90]
        ]
    )
    detector = TiledPoseDetector(detector=fake, tile_w=100, tile_h=100, overlap=0.5)
    frame = np.zeros((100, 150, 3), dtype=np.uint8)

    detections = detector.detect(frame)

    assert len(fake.crops) == 2  # both tiles inferred; dedup happens after
    assert [d.bbox for d in detections] == [[60, 10, 90, 90]]
    assert detections[0].confidence == 0.90  # the whole-frame view kept its box


def test_detect_with_full_frame_pass_recovers_a_person_the_tiles_split():
    # 200x100 frame, two 100x100 tiles. A big near-field person spans both tiles;
    # each tile only sees a clipped half, but a whole-frame pass frames them once.
    # The full-frame pass runs first (call 0 = whole image), then the two tiles.
    fake = _FakeTileDetector(
        per_call=[
            [_kp_det([0, 0, 180, 90], 0.90, (90, 45))],  # full-frame pass: whole body
            [_kp_det([0, 0, 90, 90], 0.50, (45, 45))],  # tile x=0: left half (clipped)
            [_kp_det([0, 0, 80, 90], 0.50, (40, 45))],  # tile x=100: right half -> [100,0,180,90]
        ]
    )
    detector = TiledPoseDetector(
        detector=fake,
        tile_w=100,
        tile_h=100,
        overlap=0.0,
        full_frame_pass=True,
    )
    frame = np.zeros((100, 200, 3), dtype=np.uint8)

    detections = detector.detect(frame)

    assert fake.crops == [(100, 200), (100, 100), (100, 100)]  # whole frame, then two tiles
    # The whole-body box wins; both seam-clipped partials are suppressed as
    # contained within it, so the split person is one detection, not two halves.
    assert [d.bbox for d in detections] == [[0, 0, 180, 90]]
    assert detections[0].confidence == 0.90


def test_detect_with_zone_bounds_only_infers_tiles_inside_the_zone():
    # 200x100 frame, two 100x100 tiles. The zone bbox covers only the left tile,
    # so the right tile never costs a pose call — the with-zones compute saving.
    fake = _FakeTileDetector(per_call=[[_kp_det([10, 10, 30, 30], 0.9, (20, 20))]])
    detector = TiledPoseDetector(
        detector=fake,
        tile_w=100,
        tile_h=100,
        overlap=0.0,
        zone_bounds=[(0, 0, 50, 100)],
    )
    frame = np.zeros((100, 200, 3), dtype=np.uint8)

    detections = detector.detect(frame)

    assert fake.crops == [(100, 100)]  # only the in-zone tile was inferred
    assert [d.bbox for d in detections] == [[10, 10, 30, 30]]


def test_build_hybrid_detector_wraps_a_base_detector_at_its_native_tile_size():
    # #111 wiring: a "hybrid" camera runs the tiling detector over native
    # 1280x736 tiles plus one whole-frame pass. The factory reads the tile size
    # off the base detector's declared input, so the tiles letterbox at scale 1.0.
    base = _FakeTileDetector(per_call=[])
    base.input_size = (1280, 736)

    detector = build_hybrid_detector(base)

    assert isinstance(detector, TiledPoseDetector)
    assert detector.detector is base
    assert (detector.tile_w, detector.tile_h) == (1280, 736)
    assert detector.overlap == DEFAULT_TILE_OVERLAP
    assert detector.full_frame_pass is True  # the hybrid arm's whole-frame pass
    assert detector.zone_bounds is None  # whole-frame grid by default


def test_build_hybrid_detector_restricts_reach_to_supplied_zone_bounds():
    # With authored zones + restrict_to_zones, the tiling reach is bounded to the
    # zone bboxes so no pose call is spent on frame regions no zone covers.
    base = _FakeTileDetector(per_call=[])
    base.input_size = (1280, 736)
    bounds = [(0.0, 0.0, 500.0, 500.0), (1000.0, 800.0, 2000.0, 1600.0)]

    detector = build_hybrid_detector(base, zone_bounds=bounds)

    assert detector.zone_bounds == bounds
    assert detector.full_frame_pass is True


def test_hybrid_detector_delegates_model_provenance_to_its_base():
    # result.json diagnostics (#98) read model_path/model_sha256/input_size off
    # the detector; the tiling wrapper must forward them from the base so a hybrid
    # job records the same weights provenance a plain job does.
    base = _FakeTileDetector(per_call=[])
    base.input_size = (1280, 736)
    base.model_path = "models/yolo11s-pose-1280x736.onnx"
    base.model_sha256 = "a" * 64

    detector = build_hybrid_detector(base)

    assert detector.model_path == "models/yolo11s-pose-1280x736.onnx"
    assert detector.model_sha256 == "a" * 64
    assert detector.input_size == (1280, 736)
