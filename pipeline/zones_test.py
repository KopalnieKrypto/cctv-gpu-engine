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
from pipeline.zones import Zone, ZoneConfig, ZoneConfigError


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
