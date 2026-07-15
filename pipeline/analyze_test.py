"""Tests for the analyze CLI — single-frame proof-of-concept entry point.

The detector and extractor are mocked at the boundary so we can verify the
CLI's argument parsing, orchestration, and JSON output schema without a GPU
or real video file.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import numpy as np
import pytest

from pipeline.analyze import main
from pipeline.postprocessing import Detection, Keypoint


def _detection(
    activity: str = "standing",
    confidence: float = 0.9,
    bbox: tuple[float, float, float, float] = (100.0, 200.0, 300.0, 600.0),
) -> Detection:
    kps = [Keypoint(x=float(i), y=float(i * 2), vis=0.95) for i in range(17)]
    det = Detection(
        bbox=list(bbox),
        confidence=confidence,
        keypoints=kps,
    )
    det.activity = activity
    return det


class _FakeEmbedder:
    """Stands in for the OSNet model — a GPU boundary, like the pose model.

    Gives every distinct bbox its own orthogonal appearance vector: the same
    box across frames reads as the same person, a different box reads as
    someone else. Enough for the tracker to behave realistically with no model
    and no GPU.
    """

    def __init__(self) -> None:
        self._identities: dict[tuple, int] = {}

    def embed(self, frame, detections: list[Detection]) -> list[np.ndarray]:
        vectors = []
        for det in detections:
            index = self._identities.setdefault(tuple(det.bbox), len(self._identities))
            vector = np.zeros(16, dtype=np.float32)
            vector[index] = 1.0
            vectors.append(vector)
        return vectors


@pytest.fixture(autouse=True)
def _fake_reid_model(mocker):
    """Keep the Re-ID model loader off the GPU for every test in this module."""
    mocker.patch("pipeline.analyze.load_reid_model", return_value=_FakeEmbedder())


class TestAnalyzeCLI:
    def test_outputs_json_with_person_count_and_per_detection_data(self, mocker, capsys):
        # Mock extractor → returns a dummy frame
        fake_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        mocker.patch(
            "pipeline.analyze.extract_frame_at",
            return_value=fake_frame,
        )

        # Mock detector loader → returns a fake detector that yields one person
        fake_detector = MagicMock()
        fake_detector.detect.return_value = [
            _detection(activity="walking", confidence=0.87),
        ]
        mocker.patch(
            "pipeline.analyze.load_pose_model",
            return_value=fake_detector,
        )

        exit_code = main(
            [
                "video.mp4",
                "--timestamp",
                "12.5",
                "--model",
                "models/yolo11n-pose.onnx",
            ]
        )

        assert exit_code == 0
        captured = capsys.readouterr()
        payload = json.loads(captured.out)

        # Top-level schema
        assert payload["video"] == "video.mp4"
        assert payload["timestamp_s"] == 12.5
        assert payload["person_count"] == 1
        assert isinstance(payload["persons"], list)

        person = payload["persons"][0]
        assert person["activity"] == "walking"
        assert person["confidence"] == pytest.approx(0.87, abs=1e-5)
        assert person["bbox"] == pytest.approx([100.0, 200.0, 300.0, 600.0])
        assert len(person["keypoints"]) == 17
        kp0 = person["keypoints"][0]
        assert set(kp0.keys()) == {"x", "y", "vis"}

        # Detector was loaded with the requested model path and called once
        from pipeline.analyze import load_pose_model as patched_loader

        patched_loader.assert_called_once_with("models/yolo11n-pose.onnx")
        fake_detector.detect.assert_called_once()

    def test_outputs_empty_persons_list_when_no_detections(self, mocker, capsys):
        mocker.patch(
            "pipeline.analyze.extract_frame_at",
            return_value=np.zeros((480, 640, 3), dtype=np.uint8),
        )
        fake_detector = MagicMock()
        fake_detector.detect.return_value = []
        mocker.patch(
            "pipeline.analyze.load_pose_model",
            return_value=fake_detector,
        )

        exit_code = main(
            [
                "video.mp4",
                "--timestamp",
                "0",
                "--model",
                "models/yolo11n-pose.onnx",
            ]
        )

        assert exit_code == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["person_count"] == 0
        assert payload["persons"] == []

    def test_propagates_runtime_error_with_nonzero_exit_code(self, mocker, capsys):
        # Simulate the CUDA-missing error from load_pose_model
        mocker.patch(
            "pipeline.analyze.extract_frame_at",
            return_value=np.zeros((480, 640, 3), dtype=np.uint8),
        )
        mocker.patch(
            "pipeline.analyze.load_pose_model",
            side_effect=RuntimeError("CUDAExecutionProvider not available"),
        )

        exit_code = main(
            [
                "video.mp4",
                "--timestamp",
                "5",
                "--model",
                "models/yolo11n-pose.onnx",
            ]
        )

        assert exit_code != 0
        captured = capsys.readouterr()
        assert "CUDAExecutionProvider" in captured.err


class TestAnalyzeFullVideoMode:
    def test_output_mode_writes_result_json_by_default(self, mocker, tmp_path):
        """Issue #72 — the canonical artifact is result.json, not HTML."""
        import json

        fake_frame = np.zeros((40, 60, 3), dtype=np.uint8)
        mocker.patch(
            "pipeline.analyze.iter_frames",
            return_value=iter([(0.0, fake_frame), (1.0, fake_frame)]),
        )
        fake_detector = MagicMock()
        fake_detector.detect.return_value = [_detection(activity="walking")]
        mocker.patch("pipeline.analyze.load_pose_model", return_value=fake_detector)

        out_path = tmp_path / "result.json"
        exit_code = main(
            ["video.mp4", "--output", str(out_path), "--model", "models/yolo11n-pose.onnx"]
        )

        assert exit_code == 0
        payload = json.loads(out_path.read_text(encoding="utf-8"))
        assert payload["schema_version"] == 1
        assert payload["total_frames"] == 2

    def test_output_mode_writes_standalone_html_with_summary(self, mocker, tmp_path):
        # Two frames: first frame has 2 walking persons, second frame has 1 standing
        fake_frame = np.zeros((40, 60, 3), dtype=np.uint8)
        mocker.patch(
            "pipeline.analyze.iter_frames",
            return_value=iter(
                [
                    (0.0, fake_frame),
                    (1.0, fake_frame),
                ]
            ),
        )

        fake_detector = MagicMock()
        fake_detector.detect.side_effect = [
            [_detection(activity="walking"), _detection(activity="walking")],
            [_detection(activity="standing")],
        ]
        mocker.patch(
            "pipeline.analyze.load_pose_model",
            return_value=fake_detector,
        )

        out_path = tmp_path / "report.html"
        exit_code = main(
            [
                "video.mp4",
                "--output",
                str(out_path),
                "--format",
                "html",
                "--model",
                "models/yolo11n-pose.onnx",
            ]
        )

        assert exit_code == 0
        assert out_path.exists()
        html = out_path.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in html or "<!doctype html>" in html.lower()
        # Detector ran for every frame
        assert fake_detector.detect.call_count == 2
        # Walking dominates → present in summary table
        assert "walking" in html.lower()

    def test_output_mode_handles_video_with_no_people(self, mocker, tmp_path):
        fake_frame = np.zeros((40, 60, 3), dtype=np.uint8)
        mocker.patch(
            "pipeline.analyze.iter_frames",
            return_value=iter([(0.0, fake_frame), (1.0, fake_frame)]),
        )

        fake_detector = MagicMock()
        fake_detector.detect.return_value = []  # never any persons
        mocker.patch(
            "pipeline.analyze.load_pose_model",
            return_value=fake_detector,
        )

        out_path = tmp_path / "empty.json"
        exit_code = main(
            ["video.mp4", "--output", str(out_path), "--model", "models/yolo11n-pose.onnx"]
        )

        assert exit_code == 0
        import json

        payload = json.loads(out_path.read_text(encoding="utf-8"))
        # No people → no keyframes, and dominant activity collapses to "none".
        assert payload["keyframes"] == []
        assert payload["dominant_activity"] == "none"
        assert payload["peak_persons"] == 0


class TestRunFullVideoToHtml:
    """Library function used by both the CLI and the gpu_service worker.

    Mocks live at the boundaries we control: the ffmpeg frame iterator and
    the pose-model loader. Aggregator + report renderer run for real so we're
    asserting end-to-end pipeline behaviour, not the shape of internal calls.
    """

    def test_single_chunk_returns_html_bytes(self, mocker):
        from pathlib import Path

        from pipeline.analyze import run_full_video_to_html

        fake_frame = np.zeros((40, 60, 3), dtype=np.uint8)
        mocker.patch(
            "pipeline.analyze.iter_frames",
            return_value=iter([(0.0, fake_frame), (1.0, fake_frame)]),
        )
        fake_detector = MagicMock()
        fake_detector.detect.return_value = [_detection(activity="walking")]
        mocker.patch(
            "pipeline.analyze.load_pose_model",
            return_value=fake_detector,
        )

        html = run_full_video_to_html([Path("chunk_001.mp4")])

        assert isinstance(html, bytes)
        text = html.decode("utf-8")
        assert "<!doctype html" in text.lower() or "<html" in text.lower()
        assert "walking" in text.lower()
        assert fake_detector.detect.call_count == 2

    def test_two_chunks_aggregate_into_one_report(self, mocker):
        from pathlib import Path

        from pipeline.analyze import run_full_video_to_html

        fake_frame = np.zeros((40, 60, 3), dtype=np.uint8)
        # Each chunk has its own iter_frames result; iter_frames is called
        # once per chunk so side_effect cycles through these.
        chunk_a_frames = [(float(i), fake_frame) for i in range(30)]
        chunk_b_frames = [(float(i), fake_frame) for i in range(30)]
        mocker.patch(
            "pipeline.analyze.iter_frames",
            side_effect=[iter(chunk_a_frames), iter(chunk_b_frames)],
        )
        fake_detector = MagicMock()
        fake_detector.detect.return_value = [_detection(activity="walking")]
        mocker.patch(
            "pipeline.analyze.load_pose_model",
            return_value=fake_detector,
        )

        html = run_full_video_to_html([Path("a.mp4"), Path("b.mp4")])

        # Both chunks were fed through the detector → 60 inferences total
        assert fake_detector.detect.call_count == 60
        assert isinstance(html, bytes) and len(html) > 0

    def test_progress_callback_invoked_at_least_once_per_chunk(self, mocker):
        from pathlib import Path

        from pipeline.analyze import run_full_video_to_html

        fake_frame = np.zeros((40, 60, 3), dtype=np.uint8)
        mocker.patch(
            "pipeline.analyze.iter_frames",
            side_effect=[
                iter([(0.0, fake_frame), (1.0, fake_frame)]),
                iter([(0.0, fake_frame), (1.0, fake_frame)]),
            ],
        )
        fake_detector = MagicMock()
        fake_detector.detect.return_value = []
        mocker.patch(
            "pipeline.analyze.load_pose_model",
            return_value=fake_detector,
        )

        seen: list[int] = []
        run_full_video_to_html(
            [Path("a.mp4"), Path("b.mp4")],
            progress=seen.append,
        )

        # At least one update per chunk, monotonically non-decreasing, ≤ 100,
        # final update at 100% so the worker can write status.json correctly.
        assert len(seen) >= 2
        assert seen == sorted(seen)
        assert all(0 <= p <= 100 for p in seen)
        assert seen[-1] == 100

    def test_progress_callback_fires_intra_chunk_for_long_chunks(self, mocker):
        """Long chunks fire ``progress`` every ``PROGRESS_FRAME_INTERVAL`` frames.

        Without this, gpu_service.metrics.MetricsAggregator only samples at
        chunk boundaries — for a single-chunk job that means *2 samples total*
        and ``gpu_util_peak`` is sampled outside the hot path. Telemetry
        cadence = progress cadence, so we assert the cadence here.
        """
        from pathlib import Path

        from pipeline.analyze import PROGRESS_FRAME_INTERVAL, run_full_video_to_html

        fake_frame = np.zeros((40, 60, 3), dtype=np.uint8)
        # 30 frames in a single chunk → expect chunk-start ticks at frames
        # 3, 6, 9, ... 30 (10 of them) plus the chunk-end tick at 100%.
        chunk_frames = [(float(i), fake_frame) for i in range(30)]
        mocker.patch(
            "pipeline.analyze.iter_frames",
            side_effect=[iter(chunk_frames)],
        )
        fake_detector = MagicMock()
        fake_detector.detect.return_value = []
        mocker.patch(
            "pipeline.analyze.load_pose_model",
            return_value=fake_detector,
        )

        seen: list[int] = []
        run_full_video_to_html(
            [Path("only.mp4")],
            progress=seen.append,
        )

        expected_intra = len(chunk_frames) // PROGRESS_FRAME_INTERVAL
        # >= because chunk-end may coincide with an intra tick — be lenient
        # on the exact count, strict on the cadence floor.
        assert len(seen) >= expected_intra + 1
        assert seen == sorted(seen)
        assert all(0 <= p <= 100 for p in seen)
        assert seen[-1] == 100

    def test_pipeline_runtime_error_propagates_uncaught(self, mocker):
        from pathlib import Path

        from pipeline.analyze import run_full_video_to_html

        mocker.patch(
            "pipeline.analyze.load_pose_model",
            side_effect=RuntimeError("CUDAExecutionProvider not available"),
        )

        with pytest.raises(RuntimeError, match="CUDAExecutionProvider"):
            run_full_video_to_html([Path("a.mp4")])


def _person_at(cx: float, cy: float, h: float = 400.0, w: float = 200.0) -> Detection:
    """A Detection whose bbox is centered on (cx, cy) with the given size.

    bbox height fixes the displacement normalization: with h=400 and
    DISPLACEMENT_WALK_THRESHOLD=0.05, a center move > 20px reads as walking.
    """
    x1, x2 = cx - w / 2, cx + w / 2
    y1, y2 = cy - h / 2, cy + h / 2
    kps = [Keypoint(x=cx, y=cy, vis=0.95) for _ in range(17)]
    det = Detection(bbox=[x1, y1, x2, y2], confidence=0.9, keypoints=kps)
    det.activity = "standing"
    return det


class TestVlmPerPersonDisplacement:
    """VLM-hybrid path must decide walking per person via nearest-center
    matching, not by comparing detections[0] to prev_centers[0] (issue #64).

    NMS sorts detections by confidence (postprocessing), which is not stable
    across frames — person A can be index 0 in frame N and index 1 in frame
    N+1. The old code paired ``detections[0]`` with ``prev_centers[0]`` and
    then applied the single frame-level outcome to *every* detection, so in a
    multi-person scene the walking/VLM branch flipped at random and everyone
    got one blanket label.

    These tests drive ``run_full_video_to_html`` with a fake detector + fake
    VLM. Because the pipeline mutates ``det.activity`` in place on the very
    objects the fake detector returns, we hold references to the frame-1
    detections and read their labels back after the run.
    """

    def _run_two_person_swap(self, mocker):
        """Frame 0 establishes centers; frame 1 keeps P1 stationary, moves P2
        far, and swaps detection order. Returns (p1_frame1, p2_frame1) with
        their post-run activity labels set by the pipeline."""
        from pathlib import Path

        from pipeline.analyze import run_full_video_to_html

        fake_frame = np.zeros((40, 60, 3), dtype=np.uint8)
        mocker.patch(
            "pipeline.analyze.iter_frames",
            return_value=iter([(0.0, fake_frame), (1.0, fake_frame)]),
        )

        # Frame 0: P1 @ (200,400), P2 @ (1000,400).
        p1_f0 = _person_at(200, 400)
        p2_f0 = _person_at(1000, 400)
        # Frame 1: P1 stays at (200,400); P2 moves to (1100,400) (100px > 20px
        # threshold → walking). Detection order is SWAPPED (P2 first) to expose
        # the index-based pairing bug.
        p1_f1 = _person_at(200, 400)
        p2_f1 = _person_at(1100, 400)

        fake_detector = MagicMock()
        fake_detector.detect.side_effect = [[p1_f0, p2_f0], [p2_f1, p1_f1]]
        mocker.patch("pipeline.analyze.load_pose_model", return_value=fake_detector)

        fake_vlm = MagicMock()
        fake_vlm.classify_frame.return_value = "sitting"
        mocker.patch("pipeline.vlm_classifier.VLMClassifier", return_value=fake_vlm)

        run_full_video_to_html([Path("v.mp4")], classifier="vlm")
        return p1_f1, p2_f1

    def test_vlm_displacement_uses_nearest_center_matching(self, mocker):
        """Stationary P1 must NOT be marked walking even though the detection
        order swapped and P2 moved — the moving check pairs by nearest center."""
        p1_f1, _p2_f1 = self._run_two_person_swap(mocker)
        assert p1_f1.activity != "walking", (
            "stationary P1 was mislabeled 'walking' — displacement pairing is "
            "still index-based (detections[0] vs prev_centers[0]) instead of "
            "nearest-center, so a swap in NMS order flips the branch."
        )
        # It should get the VLM's sitting/standing label instead.
        assert p1_f1.activity == "sitting"

    def test_vlm_per_person_labels_in_multi_person_frame(self, mocker):
        """A multi-person frame must not get one blanket label: the moving
        person is 'walking' while the stationary one keeps the VLM label."""
        p1_f1, p2_f1 = self._run_two_person_swap(mocker)
        assert p2_f1.activity == "walking"
        assert p1_f1.activity == "sitting"
        assert {p1_f1.activity, p2_f1.activity} == {"sitting", "walking"}, (
            "both people got the same blanket label — the frame-level outcome "
            "is still applied to every detection instead of per person."
        )

    def test_single_person_scene_behaviour_unchanged(self, mocker):
        """No regression for single-person scenes (the common case): a lone
        moving person is 'walking'; a lone stationary person gets the VLM
        label. With one detection, nearest-center == the old index-0 pairing,
        so this path is behaviourally identical to before the fix."""
        from pathlib import Path

        from pipeline.analyze import run_full_video_to_html

        fake_frame = np.zeros((40, 60, 3), dtype=np.uint8)

        # --- lone stationary person → VLM label ---
        mocker.patch(
            "pipeline.analyze.iter_frames",
            return_value=iter([(0.0, fake_frame), (1.0, fake_frame)]),
        )
        still_f0 = _person_at(500, 400)
        still_f1 = _person_at(500, 400)
        det_still = MagicMock()
        det_still.detect.side_effect = [[still_f0], [still_f1]]
        mocker.patch("pipeline.analyze.load_pose_model", return_value=det_still)
        fake_vlm = MagicMock()
        fake_vlm.classify_frame.return_value = "sitting"
        mocker.patch("pipeline.vlm_classifier.VLMClassifier", return_value=fake_vlm)

        run_full_video_to_html([Path("v.mp4")], classifier="vlm")
        assert still_f1.activity == "sitting"

        # --- lone moving person → walking (VLM not consulted for it) ---
        mocker.patch(
            "pipeline.analyze.iter_frames",
            return_value=iter([(0.0, fake_frame), (1.0, fake_frame)]),
        )
        move_f0 = _person_at(500, 400)
        move_f1 = _person_at(700, 400)  # 200px >> 20px threshold
        det_move = MagicMock()
        det_move.detect.side_effect = [[move_f0], [move_f1]]
        mocker.patch("pipeline.analyze.load_pose_model", return_value=det_move)

        run_full_video_to_html([Path("v.mp4")], classifier="vlm")
        assert move_f1.activity == "walking"


class TestDumpDetections:
    """Opt-in per-frame JSONL archive (issue #35).

    The aggregator only keeps a bounded keyframe buffer (#49), so the full
    per-frame detection stream is streamed to disk *during* the run, not
    reconstructed afterwards. Boundary mocks (ffmpeg iterator + model loader)
    match the rest of the suite; the dump-writing code path runs for real.
    """

    def test_run_full_video_to_json_writes_one_jsonl_line_per_frame(self, mocker, tmp_path):
        from pathlib import Path

        from pipeline.analyze import run_full_video_to_json

        fake_frame = np.zeros((40, 60, 3), dtype=np.uint8)
        mocker.patch(
            "pipeline.analyze.iter_frames",
            return_value=iter([(0.0, fake_frame), (1.0, fake_frame), (2.0, fake_frame)]),
        )
        fake_detector = MagicMock()
        fake_detector.detect.return_value = [_detection(activity="walking", confidence=0.87)]
        mocker.patch("pipeline.analyze.load_pose_model", return_value=fake_detector)

        dump_path = tmp_path / "detections.jsonl"
        run_full_video_to_json([Path("v.mp4")], dump_detections=dump_path)

        lines = dump_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 3  # one line per processed frame

        records = [json.loads(line) for line in lines]
        # frame_idx is 0-based and monotonic across the whole video.
        assert [r["frame_idx"] for r in records] == [0, 1, 2]
        # timestamps mirror the frames fed in.
        assert [r["timestamp_s"] for r in records] == [0.0, 1.0, 2.0]
        assert all(r["person_count"] == 1 for r in records)
        person = records[0]["persons"][0]
        assert person["confidence"] == pytest.approx(0.87, abs=1e-5)
        assert len(person["keypoints"]) == 17

    def test_dump_is_opt_in_no_file_written_by_default(self, mocker, tmp_path):
        """Default (no dump_detections) writes nothing — backward compatible."""
        from pathlib import Path

        from pipeline.analyze import run_full_video_to_json

        fake_frame = np.zeros((40, 60, 3), dtype=np.uint8)
        mocker.patch(
            "pipeline.analyze.iter_frames",
            return_value=iter([(0.0, fake_frame), (1.0, fake_frame)]),
        )
        fake_detector = MagicMock()
        fake_detector.detect.return_value = [_detection(activity="walking")]
        mocker.patch("pipeline.analyze.load_pose_model", return_value=fake_detector)

        raw = run_full_video_to_json([Path("v.mp4")])  # no dump_detections

        # Result artifact still produced; no stray files created in the cwd-adjacent tmp.
        assert isinstance(raw, bytes) and len(raw) > 0
        assert list(tmp_path.iterdir()) == []

    def test_cli_dump_detections_writes_jsonl_alongside_result(self, mocker, tmp_path):
        """`--dump-detections PATH` writes a well-formed JSONL beside result.json."""
        fake_frame = np.zeros((40, 60, 3), dtype=np.uint8)
        mocker.patch(
            "pipeline.analyze.iter_frames",
            return_value=iter([(0.0, fake_frame), (1.0, fake_frame)]),
        )
        fake_detector = MagicMock()
        fake_detector.detect.return_value = [_detection(activity="walking")]
        mocker.patch("pipeline.analyze.load_pose_model", return_value=fake_detector)

        result_path = tmp_path / "result.json"
        dump_path = tmp_path / "detections.jsonl"
        exit_code = main(
            [
                "video.mp4",
                "--output",
                str(result_path),
                "--dump-detections",
                str(dump_path),
                "--model",
                "models/yolo11n-pose.onnx",
            ]
        )

        assert exit_code == 0
        assert result_path.exists()
        assert dump_path.exists()
        lines = dump_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        records = [json.loads(line) for line in lines]
        assert [r["frame_idx"] for r in records] == [0, 1]
        assert all(r["person_count"] == 1 for r in records)

    def test_vlm_path_dumps_per_person_activities(self, mocker, tmp_path):
        """The VLM branch must dump too, capturing the per-person activity
        the pipeline assigns (walking vs the VLM's sitting/standing label)."""
        from pathlib import Path

        from pipeline.analyze import run_full_video_to_json

        fake_frame = np.zeros((40, 60, 3), dtype=np.uint8)
        mocker.patch(
            "pipeline.analyze.iter_frames",
            return_value=iter([(0.0, fake_frame), (1.0, fake_frame)]),
        )
        # Frame 0: lone person, no prior center → stationary → VLM label.
        # Frame 1: same person moved far → walking.
        det = MagicMock()
        det.detect.side_effect = [[_person_at(500, 400)], [_person_at(700, 400)]]
        mocker.patch("pipeline.analyze.load_pose_model", return_value=det)
        fake_vlm = MagicMock()
        fake_vlm.classify_frame.return_value = "sitting"
        mocker.patch("pipeline.vlm_classifier.VLMClassifier", return_value=fake_vlm)

        dump_path = tmp_path / "detections.jsonl"
        run_full_video_to_json([Path("v.mp4")], classifier="vlm", dump_detections=dump_path)

        records = [json.loads(line) for line in dump_path.read_text().splitlines()]
        assert len(records) == 2
        assert records[0]["persons"][0]["activity"] == "sitting"
        assert records[1]["persons"][0]["activity"] == "walking"


class TestDumpDetectionsIntegration:
    """End-to-end dump over a real 5-second fixture video (issue #35 AC).

    Only the GPU model is stubbed — ffmpeg frame extraction (``iter_frames``)
    runs for real, so this exercises the actual streaming loop that writes the
    JSONL, not a fully-mocked stand-in. Skipped where ffmpeg/ffprobe aren't on
    PATH (the gpu-service unit suite runs on hosts without them).
    """

    def _make_fixture(self, path, seconds: int = 5) -> None:
        import subprocess

        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"color=c=black:s=64x48:d={seconds}",
                "-pix_fmt",
                "yuv420p",
                str(path),
            ],
            check=True,
        )

    def test_five_second_fixture_produces_wellformed_jsonl(self, mocker, tmp_path):
        import shutil

        if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
            pytest.skip("ffmpeg/ffprobe not on PATH")

        from pipeline.analyze import run_full_video_to_json

        video = tmp_path / "fixture.mp4"
        self._make_fixture(video, seconds=5)

        # Real ffmpeg extraction; only the pose model is stubbed to yield one
        # person per frame so the dump has content to serialize.
        fake_detector = MagicMock()
        fake_detector.detect.return_value = [_detection(activity="standing")]
        mocker.patch("pipeline.analyze.load_pose_model", return_value=fake_detector)

        dump_path = tmp_path / "detections.jsonl"
        run_full_video_to_json([video], fps=1, dump_detections=dump_path)

        lines = dump_path.read_text(encoding="utf-8").splitlines()
        # A 5 s clip at 1 fps yields a handful of frames; be strict on the
        # invariants, lenient on the exact count (ffmpeg fps-filter edges).
        assert len(lines) >= 3
        records = [json.loads(line) for line in lines]  # every line is valid JSON
        assert [r["frame_idx"] for r in records] == list(range(len(records)))
        timestamps = [r["timestamp_s"] for r in records]
        assert timestamps == sorted(timestamps)
        for r in records:
            assert r["person_count"] == len(r["persons"])
            assert len(r["persons"][0]["keypoints"]) == 17
            assert set(r) == {"timestamp_s", "frame_idx", "person_count", "persons"}


class TestRunFullVideoToJson:
    """Canonical structured artifact (issue #72) — the bytes the gpu-agent uploads.

    Same boundary mocks as the HTML variant (ffmpeg iterator + model loader);
    the aggregator and JSON serializer run for real so we assert the actual
    ``result.json`` contract, not internal call shapes.
    """

    def test_single_chunk_returns_result_json_bytes(self, mocker):
        import json
        from pathlib import Path

        from pipeline.analyze import run_full_video_to_json

        fake_frame = np.zeros((40, 60, 3), dtype=np.uint8)
        # Three frames, not two: since issue #32 a person needs
        # MIN_TRACK_DETECTIONS sightings before they are counted, so a
        # two-frame fixture now (correctly) reports nobody and cannot exercise
        # the report schema.
        mocker.patch(
            "pipeline.analyze.iter_frames",
            return_value=iter([(float(t), fake_frame) for t in range(3)]),
        )
        fake_detector = MagicMock()
        fake_detector.detect.return_value = [_detection(activity="walking")]
        mocker.patch(
            "pipeline.analyze.load_pose_model",
            return_value=fake_detector,
        )

        raw = run_full_video_to_json([Path("chunk_001.mp4")])

        assert isinstance(raw, bytes)
        payload = json.loads(raw)
        assert payload["schema_version"] == 1
        assert payload["total_frames"] == 3
        assert payload["peak_persons"] == 1
        # The heuristic smoother reclassifies synthetic keypoints, so the exact
        # label isn't stable — assert it's one of the four canonical buckets.
        assert payload["dominant_activity"] in {"sitting", "standing", "walking", "running"}
        assert set(payload["person_minutes"]) == {"sitting", "standing", "walking", "running"}
        assert fake_detector.detect.call_count == 3

    def test_progress_callback_reaches_100_for_json_mode(self, mocker):
        from pathlib import Path

        from pipeline.analyze import run_full_video_to_json

        fake_frame = np.zeros((40, 60, 3), dtype=np.uint8)
        mocker.patch(
            "pipeline.analyze.iter_frames",
            side_effect=[iter([(0.0, fake_frame), (1.0, fake_frame)])],
        )
        fake_detector = MagicMock()
        fake_detector.detect.return_value = []
        mocker.patch(
            "pipeline.analyze.load_pose_model",
            return_value=fake_detector,
        )

        seen: list[int] = []
        run_full_video_to_json([Path("a.mp4")], progress=seen.append)

        assert seen[-1] == 100
        assert seen == sorted(seen)


class TestTrackingIntegration:
    """Issue #32 — the tracker sits between pose detection and aggregation.

    These drive the two accuracy complaints the client raised on 2026-07-15
    through the real pipeline: a cart read as a person, and people going
    uncounted. Only the GPU boundaries (frame iterator, model loaders) are
    faked; tracker, filter, aggregator and report renderer all run for real.
    """

    def test_sporadic_false_positive_never_reaches_the_report(self, mocker, tmp_path):
        # Film 1's bench-on-wheels: YOLO calls it a person for two frames of a
        # five-frame clip, and the report showed "peak 2 people" when there was
        # only ever one. Two frames is not a person.
        fake_frame = np.zeros((40, 60, 3), dtype=np.uint8)
        mocker.patch(
            "pipeline.analyze.iter_frames",
            return_value=iter([(float(t), fake_frame) for t in range(5)]),
        )

        # Sitting, not walking: the fixture's bboxes never move, and the
        # ActivitySmoother rightly rewrites a stationary "walking" label to
        # "standing". Sitting keeps this test about the filter.
        def person():
            return _detection(activity="sitting", bbox=(100.0, 200.0, 300.0, 600.0))

        def bench():
            return _detection(activity="sitting", bbox=(10.0, 10.0, 30.0, 40.0))

        fake_detector = MagicMock()
        fake_detector.detect.side_effect = [
            [person()],
            [person()],
            [person(), bench()],
            [person(), bench()],
            [person()],
        ]
        mocker.patch("pipeline.analyze.load_pose_model", return_value=fake_detector)

        out_path = tmp_path / "result.json"
        assert main(["video.mp4", "--output", str(out_path)]) == 0

        payload = json.loads(out_path.read_text(encoding="utf-8"))
        assert payload["peak_persons"] == 1
        # Five frames of one real person, and nothing at all from the bench.
        assert payload["person_minutes"]["sitting"] == pytest.approx(5 / 60)

    def test_person_present_throughout_is_counted_from_their_first_frame(self, mocker, tmp_path):
        # The filter withholds a track's first two frames until it proves
        # itself. Once proven, those frames must be released — otherwise every
        # person silently loses 2 s and we trade one bias for another.
        fake_frame = np.zeros((40, 60, 3), dtype=np.uint8)
        mocker.patch(
            "pipeline.analyze.iter_frames",
            return_value=iter([(float(t), fake_frame) for t in range(4)]),
        )
        fake_detector = MagicMock()
        fake_detector.detect.return_value = [_detection(activity="sitting")]
        mocker.patch("pipeline.analyze.load_pose_model", return_value=fake_detector)

        out_path = tmp_path / "result.json"
        assert main(["video.mp4", "--output", str(out_path)]) == 0

        payload = json.loads(out_path.read_text(encoding="utf-8"))
        assert payload["person_minutes"]["sitting"] == pytest.approx(4 / 60)
        assert payload["total_frames"] == 4

    def test_detections_dump_records_the_track_each_person_belongs_to(self, mocker, tmp_path):
        # The dump is the post-hoc audit trail (issue #35): without track_id
        # there is no way to reconstruct why a detection was counted or
        # filtered, which is exactly the bench argument we had to settle once.
        fake_frame = np.zeros((40, 60, 3), dtype=np.uint8)
        mocker.patch(
            "pipeline.analyze.iter_frames",
            return_value=iter([(float(t), fake_frame) for t in range(3)]),
        )
        fake_detector = MagicMock()
        fake_detector.detect.return_value = [_detection(activity="walking")]
        mocker.patch("pipeline.analyze.load_pose_model", return_value=fake_detector)

        dump_path = tmp_path / "detections.jsonl"
        main(
            [
                "video.mp4",
                "--output",
                str(tmp_path / "result.json"),
                "--dump-detections",
                str(dump_path),
            ]
        )

        lines = [json.loads(line) for line in dump_path.read_text().splitlines()]
        track_ids = [person["track_id"] for line in lines for person in line["persons"]]
        assert len(track_ids) == 3
        assert len(set(track_ids)) == 1  # one person, one identity, all three frames

    def test_no_tracker_flag_restores_the_untracked_behaviour(self, mocker, tmp_path):
        # The rollback lever, and how the re-baseline gets its "before" column:
        # with tracking off the two-frame bench counts as a person again, which
        # is exactly the pre-#32 behaviour the report needs to compare against.
        fake_frame = np.zeros((40, 60, 3), dtype=np.uint8)
        mocker.patch(
            "pipeline.analyze.iter_frames",
            return_value=iter([(float(t), fake_frame) for t in range(5)]),
        )
        fake_detector = MagicMock()
        fake_detector.detect.return_value = [
            _detection(activity="sitting", bbox=(100.0, 200.0, 300.0, 600.0)),
            _detection(activity="sitting", bbox=(10.0, 10.0, 30.0, 40.0)),
        ]
        mocker.patch("pipeline.analyze.load_pose_model", return_value=fake_detector)

        out_path = tmp_path / "result.json"
        assert main(["video.mp4", "--output", str(out_path), "--no-tracker"]) == 0

        payload = json.loads(out_path.read_text(encoding="utf-8"))
        assert payload["peak_persons"] == 2
        assert payload["person_minutes"]["sitting"] == pytest.approx(10 / 60)
