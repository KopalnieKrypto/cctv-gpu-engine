"""Behavior tests for issue #86's measured pose-mode selection.

Assumptions recorded before the benchmark's first RED:

* Inputs use full-frame ``xyxy`` boxes; annotations contain every person whose
  foot point is inside the semantic station zone.
* Detections outside that zone are excluded from station quality metrics.
* In-zone detections are matched in descending confidence order to the
  highest-IoU unmatched annotation. IoU >= 0.5 is a true positive; duplicates
  are false positives and unmatched annotations are false negatives.
* Outputs contain integer TP/FP/FN plus precision, recall, and F1. Zero-denominator
  metrics are explicit rather than NaN.
* This unit layer intentionally does not claim GPU throughput, VRAM, pilot-data
  completeness, or a production winner; those require measured artifacts.
"""

from __future__ import annotations

import io
import json
import threading

import numpy as np
import pytest

from pipeline.pose_benchmark import (
    DEFAULT_HEIGHT_BANDS,
    HEARTBEAT_INTERVAL_S,
    ArmMetrics,
    BenchmarkConfigError,
    BenchmarkFixture,
    BenchmarkFrame,
    BenchmarkHeartbeat,
    PeakProcessVramMonitor,
    arm_metrics_to_dict,
    build_results_artifact,
    evaluate_eligibility,
    load_benchmark_fixture,
    match_ground_truth,
    measure_end_to_end,
    measure_fixture_recall,
    query_process_gpu_vram_mb,
    recall_by_height,
    run_detection_arm,
    score_frame,
    select_winner,
    validate_fixture_manifest,
)
from pipeline.postprocessing import Detection, Keypoint
from pipeline.zones import Zone


def _detection(bbox, confidence) -> Detection:
    return Detection(
        bbox=list(bbox),
        confidence=confidence,
        keypoints=[Keypoint(0.0, 0.0, 0.0)] * 17,
    )


def test_score_frame_matches_once_filters_by_zone_and_counts_duplicates():
    zone = Zone(
        id="bending-1",
        name="Giętarka 1",
        polygon=[(0, 0), (100, 0), (100, 100), (0, 100)],
    )
    ground_truth = [
        [10, 10, 40, 90],
        [60, 20, 90, 100],  # foot point is on the zone boundary: included
    ]
    detections = [
        _detection([10, 10, 40, 90], 0.95),
        _detection([11, 11, 39, 89], 0.80),  # duplicate of the first person -> FP
        _detection([60, 20, 90, 100], 0.90),
        _detection([200, 200, 240, 280], 0.99),  # outside station -> ignored
    ]

    score = score_frame(detections, ground_truth, zone, iou_threshold=0.5)

    assert (score.tp, score.fp, score.fn) == (2, 1, 0)
    assert score.precision == pytest.approx(2 / 3)
    assert score.recall == pytest.approx(1.0)
    assert score.f1 == pytest.approx(0.8)


def test_match_ground_truth_pairs_each_person_once_in_confidence_order():
    # Highest-confidence detection claims its best GT first; a lower-confidence
    # duplicate of an already-claimed person matches nothing.
    detections = [
        _detection([0, 0, 10, 20], 0.9),
        _detection([0, 0, 10, 20], 0.5),  # duplicate of person 0
        _detection([100, 0, 110, 20], 0.8),
    ]
    ground_truth = [[0, 0, 10, 20], [100, 0, 110, 20]]

    matched = match_ground_truth(detections, ground_truth, iou_threshold=0.5)

    assert matched == {0, 1}


def test_match_ground_truth_ignores_detections_below_the_iou_threshold():
    detections = [_detection([0, 0, 10, 20], 0.9)]
    ground_truth = [[500, 500, 510, 520]]

    assert match_ground_truth(detections, ground_truth, iou_threshold=0.5) == set()


def test_recall_by_height_buckets_matched_people_into_native_size_bands():
    # The metric #101 made the headline: recall split by native person height,
    # because the 80-120 px band is the one tiling exists to recover. Two frames,
    # people at heights 100 (matched), 100 (missed), 300 (matched).
    bands = [(0.0, 120.0), (120.0, float("inf"))]
    frames = [
        (
            [_detection([0, 0, 30, 100], 0.9)],  # matches the first 100px person
            [[0, 0, 30, 100], [200, 0, 230, 100]],  # second 100px person is missed
        ),
        (
            [_detection([0, 0, 60, 300], 0.9)],  # matches the 300px person
            [[0, 0, 60, 300]],
        ),
    ]

    result = recall_by_height(frames, bands=bands, iou_threshold=0.5)

    assert result == [
        {"min_height": 0.0, "max_height": 120.0, "people": 2, "matched": 1, "recall": 0.5},
        {"min_height": 120.0, "max_height": None, "people": 1, "matched": 1, "recall": 1.0},
    ]


def test_default_height_bands_are_the_issue_101_boundaries():
    assert DEFAULT_HEIGHT_BANDS == [
        (0.0, 80.0),
        (80.0, 120.0),
        (120.0, 180.0),
        (180.0, 260.0),
        (260.0, float("inf")),
    ]


def test_score_frame_can_measure_whole_frame_regression_without_a_zone_filter():
    detection = _detection([200, 200, 240, 280], 0.9)

    score = score_frame([detection], [[200, 200, 240, 280]], zone=None)

    assert (score.tp, score.fp, score.fn) == (1, 0, 0)


def test_eligible_arm_records_every_numeric_bound_as_a_passing_check():
    baseline = ArmMetrics(
        name="baseline_640",
        tp=90,
        fp=4,
        fn=10,
        pose_wallclock_s=[0.1, 0.2],
        end_to_end_wallclock_s=100.0,
        measured_video_duration_s=300.0,
        peak_process_gpu_vram_mb=7000.0,
        film_recall={"film_1": 0.98, "film_2": 0.99},
    )
    candidate = ArmMetrics(
        name="focused_roi_640",
        tp=92,
        fp=4,
        fn=8,
        pose_wallclock_s=[0.08, 0.11],
        end_to_end_wallclock_s=105.0,
        measured_video_duration_s=300.0,
        peak_process_gpu_vram_mb=8000.0,
        film_recall={"film_1": 0.98, "film_2": 1.0},
    )

    result = evaluate_eligibility(candidate, baseline)

    assert result.eligible is True
    assert {check.name for check in result.checks} == {
        "pilot_recall",
        "pilot_precision",
        "throughput_regression",
        "one_hour_extrapolated_s",
        "peak_process_gpu_vram_mb",
        "film_1_recall_no_regression",
        "film_2_recall_no_regression",
    }
    assert all(check.passed for check in result.checks)
    assert candidate.one_hour_extrapolated_s == pytest.approx(1260.0)


def _arm(name: str, *, wallclock: float, tp: int = 95, fn: int = 5) -> ArmMetrics:
    return ArmMetrics(
        name=name,
        tp=tp,
        fp=4,
        fn=fn,
        pose_wallclock_s=[0.1],
        end_to_end_wallclock_s=wallclock,
        measured_video_duration_s=360.0,
        peak_process_gpu_vram_mb=7000.0,
        film_recall={"film_1": 0.98, "film_2": 0.99},
    )


def test_select_winner_chooses_fastest_eligible_arm():
    baseline = _arm("baseline_640", wallclock=100.0)
    larger = _arm("full_frame_1280", wallclock=105.0, tp=80, fn=20)  # recall fails
    focused = _arm("focused_roi_640", wallclock=90.0)

    selection = select_winner([baseline, larger, focused], baseline_name="baseline_640")

    assert selection.winner == "focused_roi_640"
    assert [result.arm for result in selection.eligibility] == [
        "baseline_640",
        "full_frame_1280",
        "focused_roi_640",
    ]


def test_select_winner_returns_none_when_no_arm_is_eligible():
    arms = [
        _arm("baseline_640", wallclock=100.0, tp=80, fn=20),
        _arm("full_frame_1280", wallclock=105.0, tp=80, fn=20),
        _arm("focused_roi_640", wallclock=90.0, tp=80, fn=20),
    ]

    assert select_winner(arms, baseline_name="baseline_640").winner is None


def test_fixture_manifest_requires_and_reports_sixty_frames_from_three_windows():
    manifest = {
        "schema_version": 1,
        "fixture_id": "bending-pilot-v1",
        "annotation_methodology": "docs/benchmarks/bending-pilot-v1.md",
        "frames": [
            {
                "id": f"frame-{index:03d}",
                "window_id": f"window-{index % 3 + 1}",
                "path": f"frames/frame-{index:03d}.jpg",
                "sha256": f"{index:064x}",
                "persons": [],
            }
            for index in range(60)
        ],
    }

    summary = validate_fixture_manifest(manifest)

    assert summary.fixture_id == "bending-pilot-v1"
    assert summary.frame_count == 60
    assert summary.window_ids == ("window-1", "window-2", "window-3")


@pytest.mark.parametrize(
    ("frame_count", "window_count", "reason"),
    [(59, 3, "at least 60"), (60, 2, "at least 3 distinct")],
)
def test_fixture_manifest_rejects_insufficient_frames_or_windows(frame_count, window_count, reason):
    manifest = {
        "schema_version": 1,
        "fixture_id": "too-small",
        "annotation_methodology": "docs/benchmarks/too-small.md",
        "frames": [
            {"id": str(index), "window_id": f"window-{index % window_count}"}
            for index in range(frame_count)
        ],
    }

    with pytest.raises(BenchmarkConfigError, match=reason):
        validate_fixture_manifest(manifest)


class _FlushTrackingStream(io.StringIO):
    def __init__(self):
        super().__init__()
        self.flush_count = 0

    def flush(self):
        self.flush_count += 1
        super().flush()


def test_long_run_heartbeat_flushes_and_refreshes_partial_results(tmp_path):
    stream = _FlushTrackingStream()
    partial_path = tmp_path / "partial.json"
    snapshot = {"arm": "baseline_640", "frames_completed": 7}

    with BenchmarkHeartbeat(
        label="baseline_640",
        partial_path=partial_path,
        snapshot=lambda: snapshot,
        stream=stream,
        interval_s=0.01,
    ):
        threading.Event().wait(0.035)

    assert HEARTBEAT_INTERVAL_S <= 60
    assert stream.getvalue().count("BENCHMARK_HEARTBEAT") >= 2
    assert stream.flush_count >= 2
    assert json.loads(partial_path.read_text(encoding="utf-8")) == snapshot


def test_arm_evidence_records_percentiles_and_explicit_extrapolation_method():
    metrics = ArmMetrics(
        name="baseline_640",
        tp=95,
        fp=4,
        fn=5,
        pose_wallclock_s=[1.0, 2.0, 3.0, 4.0, 5.0],
        end_to_end_wallclock_s=100.0,
        measured_video_duration_s=300.0,
        peak_process_gpu_vram_mb=7000.0,
        film_recall={"film_1": 0.98, "film_2": 0.99},
        frame_count=60,
    )

    evidence = arm_metrics_to_dict(metrics)

    assert evidence["pose_inference_wallclock_s"]["p50"] == pytest.approx(3.0)
    assert evidence["pose_inference_wallclock_s"]["p95"] == pytest.approx(4.8)
    assert evidence["pose_inference_wallclock_s"]["samples"] == [1, 2, 3, 4, 5]
    end_to_end = evidence["end_to_end"]
    assert end_to_end["measured_frame_count"] == 60
    assert end_to_end["measured_video_duration_s"] == 300.0
    assert end_to_end["measured_wallclock_s"] == 100.0
    assert end_to_end["one_hour_extrapolated_s"] == pytest.approx(1200.0)
    assert "measured_wallclock_s / measured_video_duration_s * 3600" in end_to_end["formula"]
    assert end_to_end["linearity_assumption"]


def test_gpu_vram_query_uses_nvidia_smi_compute_process_rows_for_one_pid():
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return type("Completed", (), {"stdout": "4242, 1024\n99, 500\n4242, 256\n"})()

    used_mb = query_process_gpu_vram_mb(pid=4242, run=fake_run)

    assert used_mb == 1280.0
    command, kwargs = calls[0]
    assert command == [
        "nvidia-smi",
        "--query-compute-apps=pid,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    assert kwargs["check"] is True
    assert kwargs["timeout"] == 30


def test_vram_monitor_keeps_the_peak_seen_during_a_run():
    samples = iter([100.0, 900.0, 400.0, 300.0])

    def sample(_pid):
        return next(samples, 300.0)

    with PeakProcessVramMonitor(pid=4242, sample=sample, interval_s=0.01) as monitor:
        threading.Event().wait(0.035)

    assert monitor.peak_mb == 900.0


def test_detection_arm_streams_frames_and_checkpoints_raw_results(tmp_path):
    zone = Zone(
        id="bending-1",
        name="Giętarka 1",
        polygon=[(0, 0), (100, 0), (100, 100), (0, 100)],
    )
    detection = _detection([10, 10, 40, 90], 0.95)

    class FakeDetector:
        def __init__(self):
            self.calls = 0

        def detect(self, _frame):
            self.calls += 1
            return [detection]

    frames = [
        BenchmarkFrame(
            id=f"frame-{index}",
            window_id="window-1",
            image=np.zeros((100, 100, 3), dtype=np.uint8),
            ground_truth=[[10, 10, 40, 90]],
        )
        for index in range(2)
    ]
    ticks = iter([0.0, 0.1, 1.0, 1.2])
    partial_path = tmp_path / "baseline.partial.json"
    detector = FakeDetector()

    result = run_detection_arm(
        name="baseline_640",
        detector=detector,
        frames=frames,
        zone=zone,
        partial_path=partial_path,
        clock=lambda: next(ticks),
        heartbeat_stream=io.StringIO(),
    )

    assert detector.calls == 2
    assert result.pose_wallclock_s == pytest.approx([0.1, 0.2])
    assert (result.tp, result.fp, result.fn) == (2, 0, 0)
    artifact = json.loads(partial_path.read_text(encoding="utf-8"))
    assert artifact["frames_completed"] == 2
    assert [frame["frame_id"] for frame in artifact["frames"]] == ["frame-0", "frame-1"]
    assert artifact["frames"][0]["detections"][0]["bbox"] == [10, 10, 40, 90]


def test_fixture_loader_verifies_hashes_and_in_zone_annotations(tmp_path):
    zone = Zone(
        id="bending-1",
        name="Giętarka 1",
        polygon=[(0, 0), (100, 0), (100, 100), (0, 100)],
    )
    frames = []
    for index in range(60):
        path = tmp_path / f"frame-{index:03d}.jpg"
        path.write_bytes(b"versioned-frame")
        frames.append(
            {
                "id": f"frame-{index:03d}",
                "window_id": f"window-{index % 3 + 1}",
                "path": path.name,
                "sha256": "a" * 64,
                "persons": [{"bbox": [10, 10, 40, 100]}],
            }
        )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "fixture_id": "bending-pilot-v1",
                "annotation_methodology": "methodology.md",
                "frames": frames,
            }
        ),
        encoding="utf-8",
    )

    fixture = load_benchmark_fixture(
        manifest_path,
        zone=zone,
        image_reader=lambda _path: np.zeros((100, 100, 3), dtype=np.uint8),
        sha256_file=lambda _path: "a" * 64,
    )

    assert fixture.summary.frame_count == 60
    assert len(fixture.frames) == 60
    assert callable(fixture.frames[0].image)
    assert fixture.frames[0].ground_truth == [[10.0, 10.0, 40.0, 100.0]]


def test_results_artifact_promotes_only_the_measured_fastest_eligible_arm():
    fixture = BenchmarkFixture(
        summary=type(
            "Summary",
            (),
            {
                "fixture_id": "pilot-v1",
                "frame_count": 60,
                "window_ids": ("w1", "w2", "w3"),
                "annotation_methodology": "docs/pilot-v1.md",
            },
        )(),
        frames=[
            BenchmarkFrame(
                id=f"frame-{index}",
                window_id=f"w{index % 3 + 1}",
                image=None,
                ground_truth=[],
            )
            for index in range(60)
        ],
    )
    metrics = [
        _arm("baseline_640", wallclock=100.0),
        _arm("full_frame_1280", wallclock=105.0, tp=80, fn=20),
        _arm("focused_roi_640", wallclock=90.0),
    ]
    raw_frames = {
        name: [{"frame_id": frame.id} for frame in fixture.frames]
        for name in ("baseline_640", "full_frame_1280", "focused_roi_640")
    }
    models = {
        "baseline_640": {"sha256": "a" * 64, "input_size": 640},
        "full_frame_1280": {"sha256": "b" * 64, "input_size": 1280},
        "focused_roi_640": {"sha256": "a" * 64, "input_size": 640},
    }

    artifact = build_results_artifact(
        fixture=fixture,
        metrics=metrics,
        raw_frames=raw_frames,
        model_evidence=models,
        reference_tiling={"full_frame_640": 1, "tiled_3x3_640": 97},
        production_default_changed=True,
    )

    assert artifact["decision"] == {
        "winner": "focused_roi_640",
        "production_default_changed": True,
        "follow_up_issue": None,
    }
    assert artifact["arms"]["focused_roi_640"]["eligibility_checks"]
    assert len(artifact["arms"]["baseline_640"]["frames"]) == 60


def test_end_to_end_measurement_heartbeats_and_records_calibration_inputs(tmp_path):
    ticks = iter([10.0, 12.5])

    def run(progress):
        progress(50)
        progress(100)

    measurement = measure_end_to_end(
        label="focused_roi_640",
        run=run,
        measured_video_duration_s=180.0,
        measured_frame_count=180,
        partial_path=tmp_path / "e2e.partial.json",
        clock=lambda: next(ticks),
        heartbeat_stream=io.StringIO(),
    )

    assert measurement.wallclock_s == 2.5
    assert measurement.measured_video_duration_s == 180.0
    assert measurement.measured_frame_count == 180
    partial = json.loads((tmp_path / "e2e.partial.json").read_text(encoding="utf-8"))
    assert partial["progress_pct"] == 100
    assert partial["state"] == "completed"


def test_regression_fixture_recall_aggregates_whole_frame_counts():
    fixture = BenchmarkFixture(
        summary=type("Summary", (), {})(),
        frames=[
            BenchmarkFrame("one", "film", None, [[0, 0, 10, 10]]),
            BenchmarkFrame("two", "film", None, [[0, 0, 10, 10]]),
        ],
    )

    class Detector:
        def __init__(self):
            self.calls = 0

        def detect(self, _image):
            self.calls += 1
            return [_detection([0, 0, 10, 10], 0.9)] if self.calls == 1 else []

    assert measure_fixture_recall(Detector(), fixture) == pytest.approx(0.5)


class TestArmInputSizes:
    """Issue #100 — arms declare the exact tensor they were measured at.

    An arm's identity is its input shape; scoring a 640×384 export while the
    harness believes it ran 640×640 would silently compare two different
    experiments. The mapping is explicit so a mismatched model fails at load
    rather than producing a plausible, wrong row in the gate table.
    """

    def test_the_issue_86_arms_keep_their_square_sizes(self):
        from pipeline.pose_benchmark import expected_input_size_for_arm

        assert expected_input_size_for_arm("baseline_640") == (640, 640)
        assert expected_input_size_for_arm("focused_roi_640") == (640, 640)
        assert expected_input_size_for_arm("full_frame_1280") == (1280, 1280)

    def test_the_non_square_arm_declares_640_by_384(self):
        from pipeline.pose_benchmark import expected_input_size_for_arm

        # Same width as baseline_640, so identical detection scale — the whole
        # claim the arm exists to test — for 0.60x the tensor.
        assert expected_input_size_for_arm("baseline_640x384") == (640, 384)

    def test_the_resolution_arm_declares_1280_by_736(self):
        from pipeline.pose_benchmark import expected_input_size_for_arm

        # Issue #101's arm: double baseline_640's width — so double the
        # detection scale — without full_frame_1280's square padding waste.
        assert expected_input_size_for_arm("full_frame_1280x736") == (1280, 736)

    def test_unknown_arm_is_a_config_error_not_a_silent_640(self):
        from pipeline.pose_benchmark import expected_input_size_for_arm

        with pytest.raises(BenchmarkConfigError, match="unknown benchmark arm"):
            expected_input_size_for_arm("baseline_512")

    def test_the_tiling_arms_declare_the_1280_by_736_tile_input(self):
        from pipeline.pose_benchmark import expected_input_size_for_arm

        # Both tiling arms run the same 1280x736 model per tile; they differ only
        # in whether the whole frame or just authored zones get tiled.
        assert expected_input_size_for_arm("tiled_1280x736") == (1280, 736)
        assert expected_input_size_for_arm("tiled_zones_1280x736") == (1280, 736)

    def test_run_arm_cli_accepts_the_non_square_arm(self):
        from pipeline.pose_benchmark import build_cli_parser

        args = build_cli_parser().parse_args(
            [
                "run-arm",
                "--arm",
                "baseline_640x384",
                "--fixture",
                "m.json",
                "--zones",
                "z.json",
                "--model",
                "m.onnx",
                "--throughput-clip",
                "c.mp4",
                "--film-1-fixture",
                "f1.json",
                "--film-2-fixture",
                "f2.json",
                "--output",
                "o.json",
            ]
        )

        assert args.arm == "baseline_640x384"


class TestTilingArmWiring:
    """Issue #110 — the tiling arms and their isolated-detector cost readout."""

    def test_pose_min_per_hour_extrapolates_per_frame_pose_cost(self):
        from pipeline.pose_benchmark import pose_min_per_hour

        # At 1 fps, a mean 0.5 s pose call per frame is 0.5 s x 3600 frames = 1800
        # pose-seconds an hour = 30 minutes. This is the detector's isolated cost,
        # not the full pipeline's — VLM/decode are read separately (#110).
        assert pose_min_per_hour([0.4, 0.6], fps=1) == pytest.approx(30.0)
        assert pose_min_per_hour([], fps=1) == 0.0

    def test_zone_bounding_boxes_reduces_each_zone_polygon_to_its_bbox(self):
        from pipeline.pose_benchmark import zone_bounding_boxes
        from pipeline.zones import ZoneConfig

        config = ZoneConfig.from_dict(
            {
                "zones": [
                    {"id": "z1", "name": "Z1", "polygon": [[10, 20], [110, 20], [60, 220]]},
                ]
            }
        )

        assert zone_bounding_boxes(config) == [(10.0, 20.0, 110.0, 220.0)]

    def test_run_tiling_arm_cli_parses_both_arms_and_the_roi_zones_flag(self):
        from pipeline.pose_benchmark import build_cli_parser

        args = build_cli_parser().parse_args(
            [
                "run-tiling-arm",
                "--arm",
                "tiled_zones_1280x736",
                "--fixture",
                "m.json",
                "--zones",
                "z.json",
                "--roi-zones",
                "roi.json",
                "--model",
                "m.onnx",
                "--overlap",
                "0.2",
                "--output",
                "o.json",
            ]
        )

        assert args.arm == "tiled_zones_1280x736"
        assert args.roi_zones == "roi.json"
        assert args.overlap == pytest.approx(0.2)

    def test_recall_by_height_from_evidence_rebuilds_detections_from_stored_bboxes(self):
        from pipeline.pose_benchmark import recall_by_height_from_evidence

        # The shape run_detection_arm persists: per-frame "detections" evidence
        # with bbox + confidence. One 100px person found, one 100px person missed.
        evidence_frames = [
            {"detections": [{"bbox": [0, 0, 30, 100], "confidence": 0.9}]},
        ]
        benchmark_frames = [
            BenchmarkFrame(
                id="f0",
                window_id="w1",
                image=None,
                ground_truth=[[0, 0, 30, 100], [200, 0, 230, 100]],
            )
        ]

        result = recall_by_height_from_evidence(
            evidence_frames,
            benchmark_frames,
            bands=[(0.0, 120.0)],
        )

        assert result == [
            {"min_height": 0.0, "max_height": 120.0, "people": 2, "matched": 1, "recall": 0.5}
        ]

    def test_run_tiling_arm_cli_rejects_a_full_frame_arm_name(self):
        from pipeline.pose_benchmark import build_cli_parser

        with pytest.raises(SystemExit):
            build_cli_parser().parse_args(
                [
                    "run-tiling-arm",
                    "--arm",
                    "baseline_640",
                    "--fixture",
                    "m.json",
                    "--zones",
                    "z.json",
                    "--model",
                    "m.onnx",
                    "--output",
                    "o.json",
                ]
            )
