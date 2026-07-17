"""Tests for the gpu-agent task runner (issue #25).

The runner is the in-process pipeline behind ``POST /analyze``: download
chunks → ffmpeg concat → pose+VLM inference → upload HTML. Every
collaborator (HTTP client, concat function, pipeline function) is injected
so the suite runs on macOS without ffmpeg, CUDA, or boto3 reachable.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from gpu_service.rest_api import TaskRegistry
from gpu_service.task_runner import DEFAULT_ZONES_CONFIG_PATH, run_task
from pipeline.zones import ZoneConfig

VALID_TASK_ID = "11111111-2222-3333-4444-555555555555"
RESULT_URL = f"https://r2.example.com/tenants/acme/results/{VALID_TASK_ID}/report.html?sig=abc"


def test_default_zones_config_path_matches_gpu_agent_mount_contract() -> None:
    assert DEFAULT_ZONES_CONFIG_PATH == Path("/config/zones.json")


def _payload(input_urls: list[str] | None = None) -> dict:
    return {
        "task_id": VALID_TASK_ID,
        "input_presigned_urls": input_urls or ["https://r2.example.com/get/chunk_001.mp4"],
        "result_presigned_url": RESULT_URL,
        "params": {},
    }


class TestRunTaskHappyPath:
    # Issue #87 contract assumptions before the first RED:
    # - input is an optional server-owned JSON file path, never a request path;
    # - a present valid file becomes one ZoneConfig passed to the pipeline;
    # - a missing file preserves the existing un-gated call and result bytes;
    # - malformed input is a task failure, not a silent zones=None fallback;
    # - the legacy R2 worker and real-camera detection quality are out of scope.
    def test_valid_server_zones_config_reaches_pipeline(self, tmp_path) -> None:
        zones_path = tmp_path / "zones.json"
        zones_path.write_text(
            json.dumps(
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
        )
        observed: list[ZoneConfig] = []

        def pipeline(chunks, progress, *, zones=None):
            observed.append(zones)
            return b'{"zones":[]}'

        registry = TaskRegistry()
        http = MagicMock()
        http.download.side_effect = lambda url, dest: dest.write_bytes(b"FAKE_MP4")

        run_task(
            payload=_payload(),
            registry=registry,
            workdir=tmp_path / "work",
            http=http,
            concat=MagicMock(),
            pipeline=pipeline,
            zones_config_path=zones_path,
        )

        assert registry.get(VALID_TASK_ID) == {"state": "completed"}
        assert len(observed) == 1
        assert isinstance(observed[0], ZoneConfig)
        assert [zone.id for zone in observed[0].zones] == ["bending-1"]

    def test_environment_overrides_default_zones_config_path(self, tmp_path, monkeypatch) -> None:
        zones_path = tmp_path / "platform-mounted-zones.json"
        zones_path.write_text(
            json.dumps(
                {
                    "zones": [
                        {
                            "id": "bending-1",
                            "name": "Giętarka 1",
                            "polygon": [[0, 0], [100, 0], [100, 100], [0, 100]],
                        }
                    ]
                }
            )
        )
        monkeypatch.setenv("ZONES_CONFIG_PATH", str(zones_path))
        observed: list[ZoneConfig | None] = []

        def pipeline(chunks, progress, *, zones=None):
            observed.append(zones)
            return b"{}"

        http = MagicMock()
        http.download.side_effect = lambda url, dest: dest.write_bytes(b"FAKE_MP4")
        registry = TaskRegistry()

        run_task(
            payload=_payload(),
            registry=registry,
            workdir=tmp_path / "work",
            http=http,
            concat=MagicMock(),
            pipeline=pipeline,
        )

        assert registry.get(VALID_TASK_ID) == {"state": "completed"}
        assert isinstance(observed[0], ZoneConfig)
        assert [zone.id for zone in observed[0].zones] == ["bending-1"]

    def test_missing_zones_config_preserves_un_gated_call_and_result_bytes(self, tmp_path) -> None:
        expected = b'{"schema_version":5,"zones":[]}'
        pipeline = MagicMock(return_value=expected)
        registry = TaskRegistry()
        http = MagicMock()
        http.download.side_effect = lambda url, dest: dest.write_bytes(b"FAKE_MP4")

        run_task(
            payload=_payload(),
            registry=registry,
            workdir=tmp_path / "work",
            http=http,
            concat=MagicMock(),
            pipeline=pipeline,
            zones_config_path=tmp_path / "not-mounted.json",
        )

        assert registry.get(VALID_TASK_ID) == {"state": "completed"}
        http.upload.assert_called_once_with(RESULT_URL, expected)
        assert len(pipeline.call_args.args) == 2
        assert pipeline.call_args.kwargs == {}

    def test_client_supplied_zones_path_is_ignored(self, tmp_path, monkeypatch) -> None:
        client_path = tmp_path / "client-selected.json"
        client_path.write_text(
            json.dumps(
                {
                    "zones": [
                        {
                            "id": "attacker-selected-zone",
                            "name": "Must not load",
                            "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
                        }
                    ]
                }
            )
        )
        payload = _payload()
        payload["params"] = {"zones_config_path": str(client_path)}
        monkeypatch.delenv("ZONES_CONFIG_PATH", raising=False)
        pipeline = MagicMock(return_value=b"{}")
        http = MagicMock()
        http.download.side_effect = lambda url, dest: dest.write_bytes(b"FAKE_MP4")

        run_task(
            payload=payload,
            registry=TaskRegistry(),
            workdir=tmp_path / "work",
            http=http,
            concat=MagicMock(),
            pipeline=pipeline,
            zones_config_path=tmp_path / "server-owned-path-is-absent.json",
        )

        assert len(pipeline.call_args.args) == 2
        assert pipeline.call_args.kwargs == {}

    def test_single_chunk_runs_pipeline_uploads_html_marks_completed(self, tmp_path) -> None:
        registry = TaskRegistry()
        http = MagicMock()
        http.download.side_effect = lambda url, dest: dest.write_bytes(b"FAKE_MP4")
        http.upload = MagicMock()
        # Single-chunk → concat is skipped, ffmpeg untouched.
        concat = MagicMock()
        pipeline = MagicMock(return_value=b"<html>ok</html>")

        run_task(
            payload=_payload(),
            registry=registry,
            workdir=tmp_path,
            http=http,
            concat=concat,
            pipeline=pipeline,
        )

        # State sequence: running → completed
        final = registry.get(VALID_TASK_ID)
        assert final == {"state": "completed"}

        # HTTP download happened once with the only input URL.
        assert http.download.call_count == 1
        download_args = http.download.call_args
        assert download_args.args[0] == "https://r2.example.com/get/chunk_001.mp4"

        # No concat call when there's a single chunk.
        concat.assert_not_called()

        # Pipeline received the downloaded chunk as the only input.
        pipeline_args = pipeline.call_args
        chunks = pipeline_args.args[0]
        assert len(chunks) == 1
        assert chunks[0].exists()
        assert chunks[0].read_bytes() == b"FAKE_MP4"

        # Upload received the bytes returned by the pipeline + the result URL.
        http.upload.assert_called_once_with(RESULT_URL, b"<html>ok</html>")

    def test_marks_state_running_before_completed(self, tmp_path) -> None:
        registry = TaskRegistry()
        observed: list[str] = []

        def fake_pipeline(chunks, progress):
            observed.append(registry.get(VALID_TASK_ID)["state"])
            return b"<html></html>"

        http = MagicMock()
        http.download.side_effect = lambda url, dest: dest.write_bytes(b"x")

        run_task(
            payload=_payload(),
            registry=registry,
            workdir=tmp_path,
            http=http,
            concat=MagicMock(),
            pipeline=fake_pipeline,
        )

        # State was running while the pipeline ran, then flipped to completed.
        assert observed == ["running"]
        assert registry.get(VALID_TASK_ID) == {"state": "completed"}

    def test_pipeline_progress_callback_updates_registry(self, tmp_path) -> None:
        registry = TaskRegistry()

        def fake_pipeline(chunks, progress):
            progress(25)
            progress(75)
            return b"<html></html>"

        http = MagicMock()
        http.download.side_effect = lambda url, dest: dest.write_bytes(b"x")

        run_task(
            payload=_payload(),
            registry=registry,
            workdir=tmp_path,
            http=http,
            concat=MagicMock(),
            pipeline=fake_pipeline,
        )

        # Progress flows through to the registry while running; the final
        # completed state replaces it (no progress field in completed).
        assert registry.get(VALID_TASK_ID) == {"state": "completed"}

    def test_downloads_each_input_url_in_order(self, tmp_path) -> None:
        registry = TaskRegistry()
        http = MagicMock()

        def fake_download(url, dest):
            dest.write_bytes(url.encode())

        http.download.side_effect = fake_download
        # 2 chunks → concat is exercised, but mocked; pipeline receives the
        # concat output path (which the concat mock will create below).
        concat_output = tmp_path / "concat.mp4"

        def fake_concat(inputs: list[Path], output: Path) -> None:
            output.write_bytes(b"CONCATENATED")
            assert output == concat_output

        run_task(
            payload=_payload(
                input_urls=[
                    "https://r2.example.com/get/chunk_001.mp4",
                    "https://r2.example.com/get/chunk_002.mp4",
                ]
            ),
            registry=registry,
            workdir=tmp_path,
            http=http,
            concat=fake_concat,
            pipeline=MagicMock(return_value=b"<html></html>"),
        )

        # Both downloads happened, in the order given.
        urls_called = [c.args[0] for c in http.download.call_args_list]
        assert urls_called == [
            "https://r2.example.com/get/chunk_001.mp4",
            "https://r2.example.com/get/chunk_002.mp4",
        ]


class TestRunTaskZonesIntegration:
    def test_valid_config_produces_shift_gated_per_zone_result_json(self, tmp_path, mocker) -> None:
        import numpy as np

        from pipeline.analyze import run_full_video_to_json
        from pipeline.postprocessing import Detection, Keypoint

        zones_path = tmp_path / "zones.json"
        zones_path.write_text(
            json.dumps(
                {
                    "recording_start": "2026-07-16T06:00:00+02:00",
                    "shift": {"windows": [["07:00", "15:00"]], "breaks": []},
                    "zones": [
                        {
                            "id": "bending-1",
                            "name": "Giętarka 1",
                            "polygon": [[0, 0], [100, 0], [100, 100], [0, 100]],
                            "rules": {"type": "bending"},
                        }
                    ],
                }
            )
        )

        class FakeDetector:
            def __init__(self, zones):
                self.zones = zones

            def detect(self, _frame):
                detection = Detection(
                    bbox=(10.0, 10.0, 40.0, 80.0),
                    confidence=0.9,
                    keypoints=[Keypoint(25.0, 40.0, 0.9) for _ in range(17)],
                    activity="standing",
                    track_id=1,
                )
                detection.zone_id = self.zones.zone_for_detection(detection)
                return [detection]

        mocker.patch(
            "pipeline.analyze.load_pose_model",
            side_effect=lambda _model_path, zones=None: FakeDetector(zones),
        )
        frame = np.zeros((120, 120, 3), dtype=np.uint8)
        mocker.patch(
            "pipeline.analyze.iter_frames",
            side_effect=lambda _path, fps: iter([(0.0, frame), (3600.0, frame)]),
        )

        def pipeline(chunks, progress, *, zones=None):
            return run_full_video_to_json(
                chunks,
                progress=progress,
                classifier="heuristic",
                track_persons=False,
                zones=zones,
            )

        registry = TaskRegistry()
        http = MagicMock()
        http.download.side_effect = lambda url, dest: dest.write_bytes(b"FAKE_MP4")

        run_task(
            payload=_payload(),
            registry=registry,
            workdir=tmp_path / "work",
            http=http,
            concat=MagicMock(),
            pipeline=pipeline,
            zones_config_path=zones_path,
        )

        assert registry.get(VALID_TASK_ID) == {"state": "completed"}
        uploaded = json.loads(http.upload.call_args.args[1])
        assert uploaded["shift"] == {
            "windows": [["07:00", "15:00"]],
            "breaks": [],
            "excluded_duration_s": 1.0,
        }
        assert len(uploaded["zones"]) == 1
        zone = uploaded["zones"][0]
        assert zone["zone_id"] == "bending-1"
        assert zone["person_minutes"]["standing"] == 1 / 60


class TestRunTaskFailurePaths:
    def test_malformed_zones_config_fails_before_download_or_pipeline(self, tmp_path) -> None:
        zones_path = tmp_path / "zones.json"
        zones_path.write_text('{"zones": [}')
        registry = TaskRegistry()
        http = MagicMock()
        pipeline = MagicMock()

        run_task(
            payload=_payload(),
            registry=registry,
            workdir=tmp_path / "work",
            http=http,
            concat=MagicMock(),
            pipeline=pipeline,
            zones_config_path=zones_path,
        )

        final = registry.get(VALID_TASK_ID)
        assert final["state"] == "failed"
        assert "ZoneConfigError" in final["error"]
        http.download.assert_not_called()
        pipeline.assert_not_called()
        http.upload.assert_not_called()

    def test_download_failure_marks_failed_with_error(self, tmp_path) -> None:
        registry = TaskRegistry()
        http = MagicMock()
        http.download.side_effect = ConnectionError("R2 unreachable")
        pipeline = MagicMock()

        run_task(
            payload=_payload(),
            registry=registry,
            workdir=tmp_path,
            http=http,
            concat=MagicMock(),
            pipeline=pipeline,
        )

        final = registry.get(VALID_TASK_ID)
        assert final["state"] == "failed"
        assert "R2 unreachable" in final["error"]
        # Pipeline never ran (download was the blocker).
        pipeline.assert_not_called()
        # Upload never ran either.
        http.upload.assert_not_called()

    def test_pipeline_failure_marks_failed_and_skips_upload(self, tmp_path) -> None:
        registry = TaskRegistry()
        http = MagicMock()
        http.download.side_effect = lambda url, dest: dest.write_bytes(b"x")
        pipeline = MagicMock(side_effect=RuntimeError("CUDA OOM"))

        run_task(
            payload=_payload(),
            registry=registry,
            workdir=tmp_path,
            http=http,
            concat=MagicMock(),
            pipeline=pipeline,
        )

        final = registry.get(VALID_TASK_ID)
        assert final["state"] == "failed"
        assert "CUDA OOM" in final["error"]
        http.upload.assert_not_called()

    def test_upload_failure_marks_failed(self, tmp_path) -> None:
        registry = TaskRegistry()
        http = MagicMock()
        http.download.side_effect = lambda url, dest: dest.write_bytes(b"x")
        http.upload.side_effect = ConnectionError("R2 PUT failed")

        run_task(
            payload=_payload(),
            registry=registry,
            workdir=tmp_path,
            http=http,
            concat=MagicMock(),
            pipeline=MagicMock(return_value=b"<html></html>"),
        )

        final = registry.get(VALID_TASK_ID)
        assert final["state"] == "failed"
        assert "R2 PUT failed" in final["error"]

    def test_concat_failure_marks_failed(self, tmp_path) -> None:
        registry = TaskRegistry()
        http = MagicMock()
        http.download.side_effect = lambda url, dest: dest.write_bytes(b"x")
        concat = MagicMock(side_effect=RuntimeError("ffmpeg exited 1"))

        run_task(
            payload=_payload(
                input_urls=[
                    "https://r2.example.com/get/chunk_001.mp4",
                    "https://r2.example.com/get/chunk_002.mp4",
                ]
            ),
            registry=registry,
            workdir=tmp_path,
            http=http,
            concat=concat,
            pipeline=MagicMock(),
        )

        final = registry.get(VALID_TASK_ID)
        assert final["state"] == "failed"
        assert "ffmpeg" in final["error"]
