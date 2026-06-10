"""Tests for the dispatcher factory that wires REST → task_runner.

``main()`` itself (Flask serve + model warmup + SIGTERM handler install)
is integration territory; here we cover the small glue function that
binds workdir/http/concat/pipeline into a single ``dispatch(payload)``
closure for ``create_app``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from gpu_service.rest_api import TaskRegistry
from gpu_service.rest_server import make_dispatcher


class TestMakeDispatcher:
    def test_returns_callable_that_runs_task_through_runner(self, tmp_path) -> None:
        registry = TaskRegistry()
        http = MagicMock()
        http.download.side_effect = lambda url, dest: dest.write_bytes(b"x")
        concat = MagicMock()
        pipeline = MagicMock(return_value=b"<html></html>")
        # Inline executor — run on the calling thread so the assertion below
        # sees a completed state without sleeps.
        executor = MagicMock()
        executor.submit.side_effect = lambda fn, *a, **kw: fn(*a, **kw)

        dispatch = make_dispatcher(
            registry=registry,
            workdir_root=tmp_path,
            http=http,
            concat=concat,
            pipeline=pipeline,
            executor=executor,
        )

        payload = {
            "task_id": "task-xyz",
            "input_presigned_urls": ["https://r2.example.com/chunk_001.mp4"],
            "result_presigned_url": "https://r2.example.com/put/r.html",
            "params": {},
        }
        dispatch(payload)

        # Submitted to the executor (production = ThreadPoolExecutor).
        executor.submit.assert_called_once()
        # The whole pipeline ran end-to-end through run_task.
        assert registry.get("task-xyz") == {"state": "completed"}
        # Per-task workdir was created under workdir_root, named by task_id.
        assert (tmp_path / "task-xyz").exists()

    def test_per_task_workdir_isolates_two_tasks(self, tmp_path) -> None:
        registry = TaskRegistry()
        http = MagicMock()
        http.download.side_effect = lambda url, dest: dest.write_bytes(b"x")

        executor = MagicMock()
        executor.submit.side_effect = lambda fn, *a, **kw: fn(*a, **kw)

        dispatch = make_dispatcher(
            registry=registry,
            workdir_root=tmp_path,
            http=http,
            concat=MagicMock(),
            pipeline=MagicMock(return_value=b"<html></html>"),
            executor=executor,
        )

        dispatch(
            {
                "task_id": "a",
                "input_presigned_urls": ["u1"],
                "result_presigned_url": "u_result",
                "params": {},
            }
        )
        dispatch(
            {
                "task_id": "b",
                "input_presigned_urls": ["u2"],
                "result_presigned_url": "u_result",
                "params": {},
            }
        )

        assert (tmp_path / "a").exists()
        assert (tmp_path / "b").exists()
        assert registry.get("a") == {"state": "completed"}
        assert registry.get("b") == {"state": "completed"}


class TestMainPreflight:
    """``gpu_service.rest_server.main`` VRAM preflight (issue #43).

    ``main`` is otherwise integration territory (Flask + warmup + SIGTERM), so
    we stub everything heavy and only assert the ordering invariant:
    ``preflight_or_exit`` must be called *before* ``_warm_up_pipeline``, and
    a failing preflight must prevent any model load.
    """

    def _stub_heavy_deps(self, mocker, tmp_path, monkeypatch):
        monkeypatch.setenv("WORKDIR", str(tmp_path))
        monkeypatch.setenv("CLASSIFIER", "vlm")
        # Stub everything that would otherwise touch GPU / network / signals.
        warmup = mocker.patch("gpu_service.rest_server._warm_up_pipeline")
        mocker.patch("gpu_service.rest_server.signal.signal")
        fake_app = MagicMock()
        mocker.patch("gpu_service.rest_server.create_app", return_value=fake_app)
        mocker.patch("gpu_service.rest_server.PresignedHttpClient")
        mocker.patch("gpu_service.rest_server.ThreadPoolExecutor")
        return warmup, fake_app

    def test_main_runs_preflight_before_warm_up_pipeline(self, mocker, tmp_path, monkeypatch):
        from gpu_service.rest_server import main

        warmup, fake_app = self._stub_heavy_deps(mocker, tmp_path, monkeypatch)
        monkeypatch.setenv("VRAM_BUDGET_MB", "4096")

        call_order: list[str] = []
        mocker.patch(
            "gpu_service.rest_server.preflight_or_exit",
            side_effect=lambda **kw: call_order.append("preflight"),
        )
        warmup.side_effect = lambda *a, **kw: call_order.append("warmup") or MagicMock()

        main()

        # Preflight first, then warmup. Anything else is a regression.
        assert call_order == ["preflight", "warmup"]
        fake_app.run.assert_called_once()

    def test_main_passes_classifier_and_env_override_to_preflight(
        self, mocker, tmp_path, monkeypatch
    ):
        from gpu_service.rest_server import main

        self._stub_heavy_deps(mocker, tmp_path, monkeypatch)
        monkeypatch.setenv("VRAM_BUDGET_MB", "4096")
        preflight = mocker.patch("gpu_service.rest_server.preflight_or_exit")

        main()

        preflight.assert_called_once()
        kwargs = preflight.call_args.kwargs
        assert kwargs["classifier"] == "vlm"
        assert kwargs["env_override"] == "4096"

    def test_main_aborts_before_warm_up_when_preflight_fails(self, mocker, tmp_path, monkeypatch):
        # Source incident reproduction: insufficient VRAM → exit(2) before
        # the VLM weights are even touched.
        import sys as _sys

        from gpu_service.rest_server import main

        warmup, fake_app = self._stub_heavy_deps(mocker, tmp_path, monkeypatch)

        def fake_preflight(**kwargs):
            print("VRAM_PREFLIGHT_FAIL test-injected", file=_sys.stderr, flush=True)
            _sys.exit(2)

        mocker.patch("gpu_service.rest_server.preflight_or_exit", side_effect=fake_preflight)

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 2
        warmup.assert_not_called()
        fake_app.run.assert_not_called()
