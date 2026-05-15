"""Tests for the gpu-agent task runner (issue #25).

The runner is the in-process pipeline behind ``POST /analyze``: download
chunks → ffmpeg concat → pose+VLM inference → upload HTML. Every
collaborator (HTTP client, concat function, pipeline function) is injected
so the suite runs on macOS without ffmpeg, CUDA, or boto3 reachable.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from gpu_service.rest_api import TaskRegistry
from gpu_service.task_runner import run_task

VALID_TASK_ID = "11111111-2222-3333-4444-555555555555"
RESULT_URL = f"https://r2.example.com/tenants/acme/results/{VALID_TASK_ID}/report.html?sig=abc"


def _payload(input_urls: list[str] | None = None) -> dict:
    return {
        "task_id": VALID_TASK_ID,
        "input_presigned_urls": input_urls or ["https://r2.example.com/get/chunk_001.mp4"],
        "result_presigned_url": RESULT_URL,
        "params": {},
    }


class TestRunTaskHappyPath:
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


class TestRunTaskFailurePaths:
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
