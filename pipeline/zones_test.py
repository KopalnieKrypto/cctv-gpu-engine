"""Tests for zone configuration + foot-point assignment (issue #78, slice 0).

A *zone* is a named ROI polygon over a workstation, supplied as a config file
(`--zones zones.json`) and never hardcoded. These tests pin the public
behaviour: load + validate the config, and decide which zone (if any) a
detection belongs to by testing its **foot point** (midpoint of the bbox
bottom edge) against each polygon. Membership is exercised inside / outside /
on-edge, and malformed configs must raise a typed error rather than a bare
KeyError deep in the pipeline.
"""

from __future__ import annotations

import json

import pytest

from pipeline.postprocessing import Detection, Keypoint
from pipeline.zones import (
    UnsupportedRuleTypeError,
    Zone,
    ZoneConfig,
    ZoneConfigError,
    build_zone_ruleset,
)


def _square_config() -> ZoneConfig:
    """A single 100×100 square zone anchored at the origin."""
    return ZoneConfig.from_dict(
        {
            "zones": [
                {
                    "id": "bending-1",
                    "name": "Giętarka 1",
                    "polygon": [[0, 0], [100, 0], [100, 100], [0, 100]],
                    "rules": {"type": "bending"},
                }
            ]
        }
    )


def _det(bbox: tuple[float, float, float, float]) -> Detection:
    return Detection(
        bbox=list(bbox),
        confidence=0.9,
        keypoints=[Keypoint(0.0, 0.0, 0.0)] * 17,
    )


class TestPointMembership:
    def test_point_inside_polygon_returns_zone_id(self):
        config = _square_config()

        assert config.zone_for_point(50.0, 50.0) == "bending-1"

    def test_point_outside_polygon_returns_none(self):
        config = _square_config()

        assert config.zone_for_point(200.0, 200.0) is None

    def test_point_on_edge_counts_as_inside(self):
        config = _square_config()

        # A foot planted exactly on the boundary is in the zone, not out of it.
        assert config.zone_for_point(50.0, 0.0) == "bending-1"
        assert config.zone_for_point(0.0, 50.0) == "bending-1"

    def test_point_on_vertex_counts_as_inside(self):
        config = _square_config()

        assert config.zone_for_point(0.0, 0.0) == "bending-1"


class TestDetectionAssignment:
    def test_foot_point_not_center_decides_membership(self):
        # Zone is the lower band y∈[100,200]. This bbox's *center* (y=90) is
        # above the zone, but its foot point (bottom-edge midpoint, y=140) is
        # inside — foot point must win.
        config = ZoneConfig.from_dict(
            {
                "zones": [
                    {
                        "id": "floor",
                        "name": "Floor",
                        "polygon": [[0, 100], [200, 100], [200, 200], [0, 200]],
                    }
                ]
            }
        )
        det = _det((80.0, 40.0, 120.0, 140.0))  # center=(100,90), foot=(100,140)

        assert config.zone_for_detection(det) == "floor"

    def test_detection_outside_all_zones_is_none(self):
        config = _square_config()
        det = _det((500.0, 500.0, 540.0, 560.0))  # foot=(520,560), far outside

        assert config.zone_for_detection(det) is None


class TestMultipleZones:
    def test_returns_the_zone_that_contains_the_point(self):
        config = ZoneConfig.from_dict(
            {
                "zones": [
                    {"id": "left", "name": "L", "polygon": [[0, 0], [50, 0], [50, 100], [0, 100]]},
                    {
                        "id": "right",
                        "name": "R",
                        "polygon": [[50, 0], [100, 0], [100, 100], [50, 100]],
                    },
                ]
            }
        )

        assert config.zone_for_point(10.0, 50.0) == "left"
        assert config.zone_for_point(90.0, 50.0) == "right"


class TestConfigValidation:
    def test_missing_zones_key_raises_typed_error(self):
        with pytest.raises(ZoneConfigError):
            ZoneConfig.from_dict({"recording_start": "2026-07-16T06:00:00+02:00"})

    def test_zone_missing_id_raises_typed_error(self):
        with pytest.raises(ZoneConfigError):
            ZoneConfig.from_dict({"zones": [{"name": "x", "polygon": [[0, 0], [1, 0], [1, 1]]}]})

    def test_polygon_with_fewer_than_three_points_raises_typed_error(self):
        with pytest.raises(ZoneConfigError):
            ZoneConfig.from_dict({"zones": [{"id": "z", "name": "z", "polygon": [[0, 0], [1, 1]]}]})

    def test_malformed_polygon_point_raises_typed_error(self):
        with pytest.raises(ZoneConfigError):
            ZoneConfig.from_dict(
                {"zones": [{"id": "z", "name": "z", "polygon": [[0, 0], [1, 0], "nope"]}]}
            )

    def test_zone_config_error_is_a_value_error(self):
        # Callers that already catch ValueError keep working.
        assert issubclass(ZoneConfigError, ValueError)

    @pytest.mark.parametrize(
        ("raw_roi", "polygon", "reason"),
        [
            ([], [[0, 0], [100, 0], [100, 100], [0, 100]], "JSON object"),
            ({"margin_px": 10}, [[0, 0], [100, 0], [100, 100], [0, 100]], "zone_id"),
            (
                {"zone_id": "bending-1"},
                [[0, 0], [100, 0], [100, 100], [0, 100]],
                "margin_px",
            ),
            (
                {"zone_id": "missing", "margin_px": 10},
                [[0, 0], [100, 0], [100, 100], [0, 100]],
                "existing zone",
            ),
            (
                {"zone_id": "bending-1", "margin_px": "10"},
                [[0, 0], [100, 0], [100, 100], [0, 100]],
                "finite non-negative number",
            ),
            (
                {"zone_id": "bending-1", "margin_px": float("inf")},
                [[0, 0], [100, 0], [100, 100], [0, 100]],
                "finite non-negative number",
            ),
            (
                {"zone_id": "bending-1", "margin_px": -1},
                [[0, 0], [100, 0], [100, 100], [0, 100]],
                "finite non-negative number",
            ),
            (
                {"zone_id": "bending-1", "margin_px": 10},
                [[50, 0], [50, 50], [50, 100]],
                "non-zero area",
            ),
        ],
    )
    def test_inference_roi_is_fully_validated_at_config_load(self, raw_roi, polygon, reason):
        with pytest.raises(ZoneConfigError, match=reason):
            ZoneConfig.from_dict(
                {
                    "inference_roi": raw_roi,
                    "zones": [
                        {
                            "id": "bending-1",
                            "name": "Giętarka 1",
                            "polygon": polygon,
                        }
                    ],
                }
            )


class TestRestrictToZones:
    """Issue #96 — the platform's opt-in to masking the analysis to the polygons.

    ``serializeZonesConfig`` (gpu-exchange#162) always writes a concrete boolean,
    but a hand-written config may omit it entirely — absent must mean "off", so
    every pre-#96 config keeps its whole-frame totals.
    """

    def test_flag_defaults_to_false_when_the_key_is_absent(self):
        assert _square_config().restrict_to_zones is False

    def test_flag_is_read_from_the_config(self):
        config = ZoneConfig.from_dict(
            {
                "restrict_to_zones": True,
                "zones": [
                    {"id": "z", "name": "Z", "polygon": [[0, 0], [100, 0], [100, 100], [0, 100]]}
                ],
            }
        )

        assert config.restrict_to_zones is True

    def test_parses_the_bytes_the_platform_actually_emits(self, tmp_path):
        """Contract pin against the deployed platform (gpu-exchange#162).

        Verbatim output of ``serializeZonesConfig`` for a camera with the opt-in
        on — captured from the shipped platform code, not hand-written — so a
        change on either side of the boundary breaks a test rather than a
        production report.
        """
        emitted = (
            '{"recording_start":"2026-07-16T06:00:00+02:00","shift":{"timezone":"Europe/Warsaw",'
            '"windows":[["07:00","15:00"]],"breaks":[["11:00","11:20"]]},'
            '"zones":[{"id":"bending-1","name":"Giętarka 1",'
            '"polygon":[[1200,500],[2600,500],[2600,1900],[1200,1900]]}],'
            '"restrict_to_zones":true}'
        )
        path = tmp_path / "zones.json"
        path.write_text(emitted, encoding="utf-8")

        config = ZoneConfig.load(path)

        assert config.restrict_to_zones is True
        assert [z.id for z in config.zones] == ["bending-1"]
        assert config.shift_schedule is not None

    def test_platform_never_restricts_to_an_empty_zone_list(self, tmp_path):
        # The platform forces the flag false with no polygons; the engine ignores
        # it in that case anyway (belt and braces — see the aggregator tests).
        path = tmp_path / "zones.json"
        path.write_text('{"zones":[],"restrict_to_zones":false}', encoding="utf-8")

        config = ZoneConfig.load(path)

        assert config.restrict_to_zones is False
        assert config.zones == []


class TestConfigLoading:
    def test_load_reads_and_parses_a_json_file(self, tmp_path):
        path = tmp_path / "zones.json"
        path.write_text(
            json.dumps(
                {
                    "recording_start": "2026-07-16T06:00:00+02:00",
                    "shift": {"windows": [["07:00", "15:00"]], "breaks": [["11:00", "11:20"]]},
                    "zones": [
                        {
                            "id": "bending-1",
                            "name": "Giętarka 1",
                            "polygon": [[0, 0], [100, 0], [100, 100], [0, 100]],
                            "rules": {"type": "bending"},
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        config = ZoneConfig.load(path)

        assert [z.id for z in config.zones] == ["bending-1"]
        assert config.zones[0].name == "Giętarka 1"
        assert config.zone_for_point(50.0, 50.0) == "bending-1"

    def test_zone_exposes_id_name_polygon_and_rules(self):
        config = _square_config()

        zone = config.zones[0]
        assert isinstance(zone, Zone)
        assert zone.id == "bending-1"
        assert zone.name == "Giętarka 1"
        assert zone.polygon[0] == (0.0, 0.0)
        assert zone.rules == {"type": "bending"}


class TestInferenceROIBounds:
    def _config(self, polygon) -> ZoneConfig:
        return ZoneConfig.from_dict(
            {
                "inference_roi": {"zone_id": "bending-1", "margin_px": 25},
                "zones": [
                    {
                        "id": "bending-1",
                        "name": "Giętarka 1",
                        "polygon": polygon,
                    }
                ],
            }
        )

    def test_crop_and_margin_are_clipped_to_the_frame(self):
        config = self._config([[0, 0], [80, 0], [80, 80], [0, 80]])

        assert config.inference_bounds(frame_width=100, frame_height=100) == (0, 0, 100, 100)

    def test_absent_inference_roi_keeps_full_frame_mode(self):
        assert _square_config().inference_bounds(100, 100) is None

    def test_roi_with_no_pixels_inside_the_frame_fails_visibly(self):
        config = self._config([[200, 200], [300, 200], [300, 300], [200, 300]])

        with pytest.raises(ZoneConfigError, match="no pixels inside the frame"):
            config.inference_bounds(frame_width=100, frame_height=100)


class TestRuleTypeDispatch:
    """Issue #82 — per-zone rules are dispatched on ``rules.type``.

    ``bending`` is the only implemented ruleset (slices 2–3). An unimplemented
    type — the future ``welding`` station — must fail at config-load time with a
    typed error that names the supported types, not blow up untyped deep inside
    aggregation.
    """

    def _welding_config(self):
        return ZoneConfig.from_dict(
            {
                "zones": [
                    {
                        "id": "welding-1",
                        "name": "Spawalnia 1",
                        "polygon": [[0, 0], [100, 0], [100, 100], [0, 100]],
                        "rules": {"type": "welding"},
                    }
                ]
            }
        )

    def test_unknown_rule_type_raises_typed_error_naming_supported_types(self):
        with pytest.raises(UnsupportedRuleTypeError) as exc:
            self._welding_config()

        message = str(exc.value)
        assert "welding" in message  # the rejected type
        assert "bending" in message  # the supported type it should have been

    def test_unsupported_rule_type_error_is_a_zone_config_error(self):
        # A ZoneConfigError subclass, so callers already catching bad-config
        # errors keep catching an unknown rule type without special-casing.
        assert issubclass(UnsupportedRuleTypeError, ZoneConfigError)
        with pytest.raises(ZoneConfigError):
            self._welding_config()


def _bending_zone(rules: dict | None = None) -> Zone:
    """A bending zone; ``rules`` merges over the default ``{"type": "bending"}``."""
    return Zone(
        id="bending-1",
        name="Giętarka 1",
        polygon=[(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)],
        rules={"type": "bending", **(rules or {})},
    )


class TestBendingRulesetDispatch:
    """The dispatch turns a bending zone into a ruleset that derives both modes.

    ``build_zone_ruleset`` is the seam the aggregator now routes through: fed the
    same per-frame ``{track_id: foot_point}`` sightings, a bending ruleset must
    derive presence/absence + work (slice 2) and conversation (slice 3) — the
    mode-derivation logic extracted out of the aggregator with no behaviour change.
    """

    def test_bending_ruleset_derives_presence_and_conversation(self):
        ruleset = build_zone_ruleset(_bending_zone(), sampling_step_s=1.0)

        # Two tracks stand together, close and still, across three frames:
        # each anchors presence, and the pair reads as a conversation.
        for t in (0.0, 1.0, 2.0):
            ruleset.observe(t, {1: (10.0, 10.0), 2: (20.0, 10.0)})
        modes = ruleset.result()

        assert modes.presence.anchored_track_id in (1, 2)
        assert modes.presence.present_s > 0
        assert modes.conversation.conversation_s > 0

    def test_ruleset_honours_zone_conversation_rules(self):
        # proximity_px=5 is tighter than the 10px gap between the two tracks, so
        # the same sightings that conversed above now read as no conversation —
        # proving the zone's rules flow through the dispatch, not module defaults.
        ruleset = build_zone_ruleset(
            _bending_zone({"conversation": {"proximity_px": 5.0}}), sampling_step_s=1.0
        )

        for t in (0.0, 1.0, 2.0):
            ruleset.observe(t, {1: (10.0, 10.0), 2: (20.0, 10.0)})

        assert ruleset.result().conversation.conversation_s == 0

    def test_zone_without_explicit_type_defaults_to_bending(self):
        # A zone whose rules omit "type" must behave exactly like an explicit
        # bending zone — so no pre-#82 config silently changes meaning.
        explicit = build_zone_ruleset(_bending_zone(), sampling_step_s=1.0)
        implicit = build_zone_ruleset(
            Zone(
                id="bending-1",
                name="Giętarka 1",
                polygon=[(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)],
                rules={},
            ),
            sampling_step_s=1.0,
        )
        for t in (0.0, 1.0, 2.0):
            explicit.observe(t, {1: (10.0, 10.0), 2: (20.0, 10.0)})
            implicit.observe(t, {1: (10.0, 10.0), 2: (20.0, 10.0)})

        assert implicit.result() == explicit.result()
