"""Worker orchestration tests for issue #5 (R2 job coordination).

These tests exercise :func:`gpu_service.worker.process_job` and
:func:`gpu_service.worker.worker_loop` against an in-memory R2 fake. The fake
is a faithful implementation of the small R2 surface the worker actually uses
(list/get/put status, download chunks, upload report) — it is *not* a mock of
internal collaborators. Mocking lives only at the system boundary, which here
means R2 itself.

The pipeline is injected as a plain callable so we never need a real GPU,
real ffmpeg, or a real ONNX session to test orchestration: claim race
protection, status transitions, progress updates, and failure handling.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from gpu_service.metrics import MetricsSample
from gpu_service.worker import ProgressCallback, process_job, worker_loop


class FakeMetricsCollector:
    """Returns a pre-seeded sequence of MetricsSamples in order.

    Mirrors the boundary-driven pattern used by InMemoryR2: a tiny in-memory
    stand-in for the production collector so we never touch real psutil/NVML
    in unit tests. When the seeded list is exhausted we keep returning the
    last sample — process_job can call ``sample()`` more times than we
    pre-baked (one extra final sample at completion).
    """

    def __init__(self, samples: list[MetricsSample]) -> None:
        assert samples, "FakeMetricsCollector needs at least one seeded sample"
        self._samples = list(samples)
        self.call_count = 0

    def sample(self) -> MetricsSample:
        self.call_count += 1
        idx = min(self.call_count - 1, len(self._samples) - 1)
        return self._samples[idx]


class InMemoryR2:
    """Fake R2 that stores objects in a dict, keyed by full R2 key.

    Methods mirror the duck-typed interface that
    :func:`gpu_service.worker.process_job` consumes. Test setup helpers
    (``put_chunk``, ``get_object``) are deliberately separate from the
    production methods so tests can be explicit about what they're seeding.
    """

    def __init__(self) -> None:
        self._objects: dict[str, bytes] = {}

    # ----- production interface (consumed by worker) -----

    def list_pending_job_ids(self) -> list[str]:
        result: list[str] = []
        for key, raw in self._objects.items():
            if not key.endswith("/status.json"):
                continue
            try:
                status = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                continue
            if status.get("status") == "pending":
                # surveillance-jobs/{job_id}/status.json
                parts = key.split("/")
                if len(parts) >= 3:
                    result.append(parts[1])
        return sorted(result)

    def get_status(self, job_id: str) -> dict[str, Any] | None:
        raw = self._objects.get(self._status_key(job_id))
        if raw is None:
            return None
        return json.loads(raw)

    def put_status(self, job_id: str, status: dict[str, Any]) -> None:
        self._objects[self._status_key(job_id)] = json.dumps(status).encode()

    def download_chunks(self, job_id: str, dest: Path) -> list[Path]:
        prefix = f"surveillance-jobs/{job_id}/input/"
        downloaded: list[Path] = []
        for key in sorted(self._objects):
            if not key.startswith(prefix):
                continue
            name = key[len(prefix) :]
            path = dest / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(self._objects[key])
            downloaded.append(path)
        return downloaded

    def upload_report(self, job_id: str, body: bytes) -> str:
        key = f"surveillance-jobs/{job_id}/output/result.json"
        self._objects[key] = body
        return key

    # ----- test setup helpers -----

    def put_chunk(self, job_id: str, chunk_name: str, data: bytes) -> None:
        self._objects[f"surveillance-jobs/{job_id}/input/{chunk_name}"] = data

    def get_object(self, key: str) -> bytes | None:
        return self._objects.get(key)

    @staticmethod
    def _status_key(job_id: str) -> str:
        return f"surveillance-jobs/{job_id}/status.json"


def _seed_pending_job(r2: InMemoryR2, job_id: str, chunks: list[str]) -> None:
    r2.put_status(
        job_id,
        {
            "job_id": job_id,
            "status": "pending",
            "created_at": "2026-04-07T10:00:00Z",
            "updated_at": "2026-04-07T10:00:00Z",
            "input_chunks": [f"input/{c}" for c in chunks],
            "worker_id": None,
            "progress_pct": 0,
            "error": None,
            "duration_s": None,
            "report_key": None,
        },
    )
    for chunk in chunks:
        r2.put_chunk(job_id, chunk, b"fake mp4 bytes for " + chunk.encode())


def test_processes_pending_job_end_to_end(tmp_path: Path) -> None:
    """Tracer bullet — single pending job goes pending → completed in one call.

    Verifies the entire happy-path lifecycle through the public ``process_job``
    interface: claim, download, run pipeline (injected stub), upload report,
    finalize status. Pipeline is a plain function so the test doesn't need
    a GPU, ffmpeg, or any real model.
    """
    r2 = InMemoryR2()
    _seed_pending_job(r2, "job-abc", ["chunk_001.mp4"])

    received_chunks: list[Path] = []

    def fake_pipeline(chunks: list[Path], progress: ProgressCallback) -> bytes:
        received_chunks.extend(chunks)
        return b'{"schema_version": 1, "fake": "result"}'

    result = process_job(
        client=r2,
        job_id="job-abc",
        worker_id="worker-1",
        pipeline=fake_pipeline,
        workdir=tmp_path,
        now=lambda: "2026-04-07T10:00:05Z",
    )

    assert result == "completed"
    final = r2.get_status("job-abc")
    assert final is not None
    assert final["status"] == "completed"
    assert final["worker_id"] == "worker-1"
    assert final["progress_pct"] == 100
    assert final["report_key"] == "surveillance-jobs/job-abc/output/result.json"
    assert final["error"] is None
    assert final["updated_at"] == "2026-04-07T10:00:05Z"

    # Pipeline must have received the actual downloaded chunk paths, and the
    # uploaded report bytes must be exactly what the pipeline produced.
    assert len(received_chunks) == 1
    assert received_chunks[0].name == "chunk_001.mp4"
    assert received_chunks[0].read_bytes() == b"fake mp4 bytes for chunk_001.mp4"
    assert r2.get_object(final["report_key"]) == b'{"schema_version": 1, "fake": "result"}'


def test_two_workers_one_job_only_one_processes(tmp_path: Path) -> None:
    """Claim race — second worker observes ``processing`` and skips.

    SPEC §6.5 mandates last-writer-wins claim semantics: a worker reads
    ``status.json``, sees ``pending``, then writes ``processing`` + its own
    ``worker_id``. A second worker arriving later must see the new status and
    decline to do any work — it must NOT re-download chunks or re-run the
    pipeline. The pipeline stub for worker B asserts it is never called.
    """
    r2 = InMemoryR2()
    _seed_pending_job(r2, "job-shared", ["chunk_001.mp4"])

    pipeline_a_called = []
    pipeline_b_called = []

    def pipeline_a(chunks: list[Path], progress: ProgressCallback) -> bytes:
        pipeline_a_called.append(True)
        return b"<html>A</html>"

    def pipeline_b(chunks: list[Path], progress: ProgressCallback) -> bytes:
        pipeline_b_called.append(True)
        raise AssertionError("worker B must not run the pipeline on a claimed job")

    result_a = process_job(
        client=r2,
        job_id="job-shared",
        worker_id="worker-A",
        pipeline=pipeline_a,
        workdir=tmp_path / "a",
        now=lambda: "2026-04-07T10:00:01Z",
    )
    result_b = process_job(
        client=r2,
        job_id="job-shared",
        worker_id="worker-B",
        pipeline=pipeline_b,
        workdir=tmp_path / "b",
        now=lambda: "2026-04-07T10:00:02Z",
    )

    assert result_a == "completed"
    assert result_b == "skipped"
    assert pipeline_a_called == [True]
    assert pipeline_b_called == []  # never invoked

    final = r2.get_status("job-shared")
    assert final is not None
    assert final["worker_id"] == "worker-A"  # claim stuck with the first writer
    assert final["status"] == "completed"


def test_pipeline_exception_marks_job_failed_and_keeps_worker_alive(tmp_path: Path) -> None:
    """Pipeline crash → status: failed with error message; no exception escapes.

    The worker must absorb pipeline errors and record them in ``status.json``
    so the polling loop survives a single bad job. SPEC §8.2 — "Pipeline crash
    → set status: failed + error message". The exception class and stringified
    message are preserved so an operator can diagnose without log access.
    """
    r2 = InMemoryR2()
    _seed_pending_job(r2, "job-broken", ["chunk_001.mp4"])

    def exploding_pipeline(chunks: list[Path], progress: ProgressCallback) -> bytes:
        raise RuntimeError("CUDA out of memory")

    result = process_job(
        client=r2,
        job_id="job-broken",
        worker_id="worker-1",
        pipeline=exploding_pipeline,
        workdir=tmp_path,
        now=lambda: "2026-04-07T10:00:09Z",
    )

    assert result == "failed"
    final = r2.get_status("job-broken")
    assert final is not None
    assert final["status"] == "failed"
    assert final["worker_id"] == "worker-1"
    assert final["report_key"] is None
    assert final["error"] is not None
    assert "CUDA out of memory" in final["error"]
    assert final["updated_at"] == "2026-04-07T10:00:09Z"


def test_pipeline_progress_callback_persists_progress_pct(tmp_path: Path) -> None:
    """Pipeline-reported progress is written through to status.json.

    SPEC §6.6: ``status.json`` updated periodically during processing with
    ``progress_pct``. Client-agent polls every 15s. We capture an "observed"
    snapshot of progress_pct after each callback so we know each value was
    actually persisted (not just the last one). Final value after upload must
    be 100, regardless of what the pipeline reported last.
    """
    r2 = InMemoryR2()
    _seed_pending_job(r2, "job-progress", ["chunk_001.mp4"])

    observed_progress: list[int] = []

    def progressing_pipeline(chunks: list[Path], progress: ProgressCallback) -> bytes:
        for pct in (10, 50, 90):
            progress(pct)
            observed_progress.append(r2.get_status("job-progress")["progress_pct"])
        return b"<html>done</html>"

    process_job(
        client=r2,
        job_id="job-progress",
        worker_id="worker-1",
        pipeline=progressing_pipeline,
        workdir=tmp_path,
        now=lambda: "2026-04-07T10:00:42Z",
    )

    assert observed_progress == [10, 50, 90]
    final = r2.get_status("job-progress")
    assert final["progress_pct"] == 100
    assert final["status"] == "completed"


def test_worker_loop_returns_quietly_when_no_pending_jobs(tmp_path: Path) -> None:
    """Empty bucket → loop completes one iteration with no calls and no errors.

    SPEC §8.1 worker loop: when there are no pending jobs the loop should
    sleep and continue. We bound it with ``max_iterations=1`` so the test
    terminates, and we capture sleep calls to assert the loop yields between
    polls instead of busy-waiting.
    """
    r2 = InMemoryR2()
    sleep_calls: list[float] = []

    def fake_pipeline(chunks: list[Path], progress: ProgressCallback) -> bytes:
        raise AssertionError("pipeline must not be called when no pending jobs exist")

    worker_loop(
        client=r2,
        worker_id="worker-1",
        pipeline=fake_pipeline,
        workdir=tmp_path,
        poll_interval_s=10.0,
        sleep=sleep_calls.append,
        max_iterations=1,
    )

    assert sleep_calls == [10.0]


def test_worker_loop_skips_status_with_missing_required_fields(tmp_path: Path) -> None:
    """Malformed status.json (missing ``status`` key) → skip with no crash.

    Issue #5 testing focus: "Malformed status.json: missing fields → logs
    warning, skips job". The polling loop must remain alive and proceed to
    the next iteration.
    """
    r2 = InMemoryR2()
    # Seed a status.json that lacks the `status` field — list_pending_job_ids
    # filters those out, so the loop should never even attempt to process it.
    r2.put_status("job-malformed", {"job_id": "job-malformed", "input_chunks": []})

    def fake_pipeline(chunks: list[Path], progress: ProgressCallback) -> bytes:
        raise AssertionError("pipeline must not run for a malformed status.json")

    worker_loop(
        client=r2,
        worker_id="worker-1",
        pipeline=fake_pipeline,
        workdir=tmp_path,
        poll_interval_s=5.0,
        sleep=lambda _s: None,
        max_iterations=1,
    )

    # Status untouched (no claim attempted), worker still alive.
    leftover = r2.get_status("job-malformed")
    assert leftover is not None
    assert "status" not in leftover


def test_missing_chunks_marks_job_failed(tmp_path: Path) -> None:
    """No downloadable chunks → job fails with a clear error.

    Issue #5 testing focus: "Download failure: missing video → worker sets
    failed, stays alive". A job whose status.json points at chunks that no
    longer exist in R2 must end up in ``failed`` with a useful error string,
    and the pipeline must not be invoked with an empty chunk list.
    """
    r2 = InMemoryR2()
    # Pending status but NO chunk objects in the bucket.
    r2.put_status(
        "job-noinput",
        {
            "job_id": "job-noinput",
            "status": "pending",
            "input_chunks": ["input/chunk_001.mp4"],
            "worker_id": None,
            "progress_pct": 0,
            "error": None,
            "report_key": None,
        },
    )

    pipeline_calls: list[Any] = []

    def fake_pipeline(chunks: list[Path], progress: ProgressCallback) -> bytes:
        pipeline_calls.append(chunks)
        return b""

    result = process_job(
        client=r2,
        job_id="job-noinput",
        worker_id="worker-1",
        pipeline=fake_pipeline,
        workdir=tmp_path,
        now=lambda: "2026-04-07T10:00:11Z",
    )

    assert result == "failed"
    assert pipeline_calls == []  # pipeline never invoked
    final = r2.get_status("job-noinput")
    assert final["status"] == "failed"
    assert final["error"] is not None
    assert "no input chunks" in final["error"].lower()


def test_worker_loop_processes_pending_then_sleeps(tmp_path: Path) -> None:
    """worker_loop wires list_pending → process_job → sleep correctly.

    Two pending jobs + one iteration → both completed, sleep called once.
    This is the integration test for the polling loop's plumbing.
    """
    r2 = InMemoryR2()
    _seed_pending_job(r2, "job-1", ["chunk_001.mp4"])
    _seed_pending_job(r2, "job-2", ["chunk_001.mp4"])
    sleep_calls: list[float] = []

    def fake_pipeline(chunks: list[Path], progress: ProgressCallback) -> bytes:
        return b"<html>ok</html>"

    worker_loop(
        client=r2,
        worker_id="worker-1",
        pipeline=fake_pipeline,
        workdir=tmp_path,
        poll_interval_s=7.5,
        sleep=sleep_calls.append,
        max_iterations=1,
        now=lambda: "2026-04-07T10:00:99Z",
    )

    assert r2.get_status("job-1")["status"] == "completed"
    assert r2.get_status("job-2")["status"] == "completed"
    assert sleep_calls == [7.5]


def test_process_job_records_metrics_when_collector_provided(tmp_path: Path) -> None:
    """Telemetry hook — passing a MetricsCollector populates status['metrics'].

    The collector is sampled on every progress callback plus once more after
    the pipeline finishes (so the recorded peak reflects the full job).
    Status payload exposes the aggregator's summary keys (peak/avg) so the
    dashboard can render GPU/CPU/RAM/disk columns next to each job.
    """
    r2 = InMemoryR2()
    _seed_pending_job(r2, "job-metrics", ["chunk_001.mp4"])

    collector = FakeMetricsCollector(
        [
            MetricsSample(
                cpu_util_pct=10,
                ram_used_pct=20,
                disk_used_pct=30,
                gpu_util_pct=40,
                gpu_temp_c=55,
                gpu_mem_used_mb=600,
            ),
            MetricsSample(
                cpu_util_pct=85,
                ram_used_pct=42,
                disk_used_pct=31,
                gpu_util_pct=78,
                gpu_temp_c=72,
                gpu_mem_used_mb=900,
            ),
        ]
    )

    def progressing_pipeline(chunks: list[Path], progress: ProgressCallback) -> bytes:
        progress(50)
        return b"<html>ok</html>"

    result = process_job(
        client=r2,
        job_id="job-metrics",
        worker_id="worker-1",
        pipeline=progressing_pipeline,
        workdir=tmp_path,
        now=lambda: "2026-04-29T07:00:00Z",
        metrics_collector=collector,
    )

    assert result == "completed"
    # Two samples taken: one on progress(50), one final after upload.
    assert collector.call_count == 2

    final = r2.get_status("job-metrics")
    assert final is not None
    metrics = final["metrics"]
    assert metrics["samples_count"] == 2
    assert metrics["cpu_util_peak_pct"] == 85
    assert metrics["gpu_util_peak_pct"] == 78
    assert metrics["gpu_temp_peak_c"] == 72
    assert metrics["ram_used_peak_pct"] == 42
    # disk_used_pct = last sample
    assert metrics["disk_used_pct"] == 31


def test_process_job_works_without_collector(tmp_path: Path) -> None:
    """Backward compat — omitting metrics_collector keeps status.json metrics-free.

    Existing call sites and the worker_loop default path pass no collector;
    status.json must not gain a stray ``metrics`` key in that case (the
    dashboard treats its absence as 'no telemetry recorded').
    """
    r2 = InMemoryR2()
    _seed_pending_job(r2, "job-no-metrics", ["chunk_001.mp4"])

    def progressing_pipeline(chunks: list[Path], progress: ProgressCallback) -> bytes:
        progress(33)
        return b"<html>ok</html>"

    result = process_job(
        client=r2,
        job_id="job-no-metrics",
        worker_id="worker-1",
        pipeline=progressing_pipeline,
        workdir=tmp_path,
        now=lambda: "2026-04-29T07:00:00Z",
    )

    assert result == "completed"
    final = r2.get_status("job-no-metrics")
    assert final is not None
    assert "metrics" not in final


class TestMainCli:
    """``python -m gpu_service.worker`` entry point.

    Mocks live at the boundaries we control: ``R2Client`` (boto3) and
    ``worker_loop`` itself. The pipeline closure passed to ``worker_loop``
    is exercised lightly — we just verify it's callable with the expected
    shape so the Dockerfile can rely on it.
    """

    def _full_env(self) -> dict[str, str]:
        return {
            "R2_ENDPOINT": "https://example.r2.cloudflarestorage.com",
            "R2_ACCESS_KEY_ID": "AKIA-fake",
            "R2_SECRET_ACCESS_KEY": "secret-fake",
            "R2_BUCKET": "test-bucket",
            "WORKER_ID": "test-worker",
            "WORKDIR": "/tmp/test-jobs",
            "POLL_INTERVAL_S": "3",
        }

    def test_main_constructs_r2_client_and_starts_worker_loop(self, mocker, monkeypatch):
        from gpu_service.worker import main

        for k in (
            "R2_ENDPOINT",
            "R2_ACCESS_KEY_ID",
            "R2_SECRET_ACCESS_KEY",
            "R2_BUCKET",
            "WORKER_ID",
            "WORKDIR",
            "POLL_INTERVAL_S",
        ):
            monkeypatch.delenv(k, raising=False)
        for k, v in self._full_env().items():
            monkeypatch.setenv(k, v)

        fake_r2_instance = mocker.MagicMock(name="R2Client-instance")
        r2_ctor = mocker.patch(
            "gpu_service.r2_client.R2Client",
            return_value=fake_r2_instance,
        )
        loop_mock = mocker.patch("gpu_service.worker.worker_loop")
        # Issue #6: dashboard runs in a daemon thread alongside the worker.
        # Stub it so the test doesn't try to bind a real socket.
        mocker.patch("gpu_service.dashboard.serve")
        # Issue #43: preflight calls torch.cuda by default — stub it for the
        # CPU dev box. The dedicated wiring tests above verify it's invoked.
        mocker.patch("gpu_service.worker.preflight_or_exit")

        exit_code = main([])

        assert exit_code == 0
        r2_ctor.assert_called_once_with(
            endpoint="https://example.r2.cloudflarestorage.com",
            access_key="AKIA-fake",
            secret_key="secret-fake",
            bucket="test-bucket",
        )
        loop_mock.assert_called_once()
        kwargs = loop_mock.call_args.kwargs
        assert kwargs["client"] is fake_r2_instance
        assert kwargs["worker_id"] == "test-worker"
        assert kwargs["workdir"] == Path("/tmp/test-jobs")
        assert kwargs["poll_interval_s"] == 3.0
        assert callable(kwargs["pipeline"])

    def test_main_runs_vram_preflight_with_classifier_and_env_override(self, mocker, monkeypatch):
        # Issue #43 AC: container entrypoint queries free VRAM with the right
        # budget *before* loading any model. We assert the preflight is
        # invoked and the worker actually proceeds past it on success.
        from gpu_service.worker import main

        for k in (
            "R2_ENDPOINT",
            "R2_ACCESS_KEY_ID",
            "R2_SECRET_ACCESS_KEY",
            "R2_BUCKET",
            "WORKER_ID",
            "WORKDIR",
            "POLL_INTERVAL_S",
            "CLASSIFIER",
            "VRAM_BUDGET_MB",
        ):
            monkeypatch.delenv(k, raising=False)
        for k, v in self._full_env().items():
            monkeypatch.setenv(k, v)
        monkeypatch.setenv("CLASSIFIER", "vlm")
        monkeypatch.setenv("VRAM_BUDGET_MB", "512")

        preflight_mock = mocker.patch("gpu_service.worker.preflight_or_exit")
        mocker.patch("gpu_service.r2_client.R2Client")
        mocker.patch("gpu_service.worker.worker_loop")
        mocker.patch("gpu_service.dashboard.serve")

        exit_code = main([])

        assert exit_code == 0
        preflight_mock.assert_called_once()
        kwargs = preflight_mock.call_args.kwargs
        assert kwargs["classifier"] == "vlm"
        assert kwargs["env_override"] == "512"

    def test_main_aborts_before_constructing_r2_client_when_preflight_fails(
        self, mocker, monkeypatch
    ):
        # Source incident: PyTorch OOM trace mid-load. After this fix, the
        # operator should see the preflight line and the worker must never
        # reach R2Client / worker_loop / dashboard.
        import sys as _sys

        from gpu_service.worker import main

        for k in (
            "R2_ENDPOINT",
            "R2_ACCESS_KEY_ID",
            "R2_SECRET_ACCESS_KEY",
            "R2_BUCKET",
            "WORKER_ID",
            "WORKDIR",
            "POLL_INTERVAL_S",
            "CLASSIFIER",
            "VRAM_BUDGET_MB",
        ):
            monkeypatch.delenv(k, raising=False)
        for k, v in self._full_env().items():
            monkeypatch.setenv(k, v)

        def fake_preflight(**kwargs):
            print("VRAM_PREFLIGHT_FAIL test-injected-failure", file=_sys.stderr, flush=True)
            _sys.exit(2)

        mocker.patch("gpu_service.worker.preflight_or_exit", side_effect=fake_preflight)
        r2_ctor = mocker.patch("gpu_service.r2_client.R2Client")
        loop_mock = mocker.patch("gpu_service.worker.worker_loop")
        dashboard_mock = mocker.patch("gpu_service.dashboard.serve")

        with pytest.raises(SystemExit) as exc_info:
            main([])

        assert exc_info.value.code == 2
        r2_ctor.assert_not_called()
        loop_mock.assert_not_called()
        dashboard_mock.assert_not_called()

    def test_main_fails_fast_when_required_env_var_missing(self, mocker, monkeypatch, capsys):
        from gpu_service.worker import main

        env = self._full_env()
        del env["R2_ACCESS_KEY_ID"]
        for k in (
            "R2_ENDPOINT",
            "R2_ACCESS_KEY_ID",
            "R2_SECRET_ACCESS_KEY",
            "R2_BUCKET",
            "WORKER_ID",
            "WORKDIR",
            "POLL_INTERVAL_S",
        ):
            monkeypatch.delenv(k, raising=False)
        for k, v in env.items():
            monkeypatch.setenv(k, v)

        loop_mock = mocker.patch("gpu_service.worker.worker_loop")
        r2_ctor = mocker.patch("gpu_service.r2_client.R2Client")

        exit_code = main([])

        assert exit_code != 0
        err = capsys.readouterr().err
        assert "R2_ACCESS_KEY_ID" in err
        # Fail-fast: never reached the loop or constructed a client
        loop_mock.assert_not_called()
        r2_ctor.assert_not_called()
