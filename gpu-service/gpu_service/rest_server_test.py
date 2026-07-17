"""Tests for the dispatcher factory that wires REST → task_runner.

``main()`` itself (Flask serve + model warmup + SIGTERM handler install)
is integration territory; here we cover the small glue function that
binds workdir/http/concat/pipeline into a single ``dispatch(payload)``
closure for ``create_app``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gpu_service.rest_api import Readiness, TaskRegistry, create_app
from gpu_service.rest_server import _warm_up_pipeline, make_dispatcher
from pipeline.zones import ZoneConfig


class TestWarmPipeline:
    def test_forwards_server_loaded_zones_to_json_pipeline(self, mocker) -> None:
        run_full_video_to_json = mocker.patch(
            "pipeline.analyze.run_full_video_to_json", return_value=b'{"zones":[]}'
        )
        load_pose_model = mocker.patch("pipeline.pose_detector.load_pose_model")
        zones = ZoneConfig.from_dict(
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

        pipeline = _warm_up_pipeline("model.onnx", "vlm")
        result = pipeline([Path("chunk.mp4")], lambda _pct: None, zones=zones)

        assert result == b'{"zones":[]}'
        load_pose_model.assert_called_once_with("model.onnx")
        assert run_full_video_to_json.call_args.kwargs["zones"] is zones


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


class TestRestZonesIntegration:
    def test_post_analyze_uses_server_config_and_ignores_client_path(
        self, tmp_path, monkeypatch
    ) -> None:
        server_config = tmp_path / "server-zones.json"
        server_config.write_text(
            """{
              "zones": [{
                "id": "bending-1",
                "name": "Giętarka 1",
                "polygon": [[0,0], [100,0], [100,100], [0,100]]
              }]
            }"""
        )
        monkeypatch.setenv("ZONES_CONFIG_PATH", str(server_config))
        observed_zone_ids: list[list[str]] = []

        def pipeline(chunks, progress, *, zones=None):
            observed_zone_ids.append([zone.id for zone in zones.zones])
            return b'{"zones":[]}'

        registry = TaskRegistry()
        readiness = Readiness()
        readiness.mark_ready()
        http = MagicMock()
        http.download.side_effect = lambda url, dest: dest.write_bytes(b"FAKE_MP4")
        executor = MagicMock()
        executor.submit.side_effect = lambda fn, *args, **kwargs: fn(*args, **kwargs)
        dispatch = make_dispatcher(
            registry=registry,
            workdir_root=tmp_path / "work",
            http=http,
            concat=MagicMock(),
            pipeline=pipeline,
            executor=executor,
        )
        app = create_app(readiness=readiness, registry=registry, dispatch=dispatch)
        task_id = "11111111-2222-3333-4444-555555555555"

        response = app.test_client().post(
            "/analyze",
            json={
                "task_id": task_id,
                "input_presigned_urls": ["https://r2.example.com/chunk.mp4"],
                "result_presigned_url": (
                    f"https://r2.example.com/tenants/acme/results/{task_id}/result.json?sig=x"
                ),
                "params": {"zones_config_path": "/client/chosen.json"},
            },
        )

        assert response.status_code == 202
        assert registry.get(task_id) == {"state": "completed"}
        assert observed_zone_ids == [["bending-1"]]


class TestMainServing:
    """``main`` must serve the gpu-agent REST contract through waitress, not
    Flask's Werkzeug dev server (issue #63).

    Werkzeug's ``app.run`` is single-process, has weaker slow-client / timeout
    handling, and prints a "do not use in production" banner — yet it was the
    server on the one port a production gpu-agent depends on. waitress is
    already a dependency (``client_agent.appliance`` serves through it), so the
    fix is wiring, not a new import.
    """

    def _stub_heavy_deps(self, mocker, tmp_path, monkeypatch):
        monkeypatch.setenv("WORKDIR", str(tmp_path))
        monkeypatch.setenv("CLASSIFIER", "vlm")
        monkeypatch.setenv("VRAM_BUDGET_MB", "4096")
        mocker.patch("gpu_service.rest_server.preflight_or_exit")
        mocker.patch("gpu_service.rest_server._warm_up_pipeline")
        mocker.patch("gpu_service.rest_server.signal.signal")
        fake_app = MagicMock()
        mocker.patch("gpu_service.rest_server.create_app", return_value=fake_app)
        mocker.patch("gpu_service.rest_server.PresignedHttpClient")
        mocker.patch("gpu_service.rest_server.ThreadPoolExecutor")
        fake_serve = mocker.patch("gpu_service.rest_server.serve")
        return fake_app, fake_serve

    def test_main_serves_via_waitress(self, mocker, tmp_path, monkeypatch):
        from gpu_service.rest_server import main

        fake_app, fake_serve = self._stub_heavy_deps(mocker, tmp_path, monkeypatch)
        monkeypatch.setenv("REST_PORT", "5003")

        main()

        # Served through waitress with the built app on the contract port.
        fake_serve.assert_called_once()
        args, kwargs = fake_serve.call_args
        served_app = kwargs.get("app", args[0] if args else None)
        assert served_app is fake_app
        assert kwargs.get("host") == "0.0.0.0"
        assert kwargs.get("port") == 5003
        # And the Werkzeug dev server is never touched.
        fake_app.run.assert_not_called()

    def test_sigterm_still_flips_running_tasks(self, mocker, tmp_path, monkeypatch):
        """The SIGTERM contract must survive the switch to waitress: the
        handler main() installs still flips in-flight tasks to failed and
        exits 0 (the signal arrives on the main thread, so sys.exit unwinds
        waitress.serve exactly as it unwound app.run)."""
        import signal as _signal

        from gpu_service.rest_api import TaskRegistry
        from gpu_service.rest_server import main

        self._stub_heavy_deps(mocker, tmp_path, monkeypatch)

        # Capture the handlers main() installs instead of clobbering pytest's.
        handlers: dict[int, object] = {}
        mocker.patch(
            "gpu_service.rest_server.signal.signal",
            side_effect=lambda sig, h: handlers.__setitem__(sig, h),
        )
        # Real registry so we can assert the state transition, captured here.
        real_registry = TaskRegistry()
        mocker.patch("gpu_service.rest_server.TaskRegistry", return_value=real_registry)

        main()

        # An in-flight task exists when docker-stop delivers SIGTERM.
        real_registry.set_running("job-1", progress=0.5)
        sigterm_handler = handlers[_signal.SIGTERM]

        with pytest.raises(SystemExit) as exc_info:
            sigterm_handler(_signal.SIGTERM, None)

        assert exc_info.value.code == 0
        assert real_registry.get("job-1") == {"state": "failed", "error": "terminated"}


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
        # Stub everything that would otherwise touch GPU / network / signals /
        # bind a port. waitress.serve is stubbed so main() returns instead of
        # blocking in the WSGI loop (issue #63).
        warmup = mocker.patch("gpu_service.rest_server._warm_up_pipeline")
        mocker.patch("gpu_service.rest_server.signal.signal")
        fake_app = MagicMock()
        mocker.patch("gpu_service.rest_server.create_app", return_value=fake_app)
        mocker.patch("gpu_service.rest_server.PresignedHttpClient")
        mocker.patch("gpu_service.rest_server.ThreadPoolExecutor")
        fake_serve = mocker.patch("gpu_service.rest_server.serve")
        return warmup, fake_app, fake_serve

    def test_main_runs_preflight_before_warm_up_pipeline(self, mocker, tmp_path, monkeypatch):
        from gpu_service.rest_server import main

        warmup, fake_app, fake_serve = self._stub_heavy_deps(mocker, tmp_path, monkeypatch)
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
        fake_serve.assert_called_once()
        fake_app.run.assert_not_called()

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

        warmup, fake_app, fake_serve = self._stub_heavy_deps(mocker, tmp_path, monkeypatch)

        def fake_preflight(**kwargs):
            print("VRAM_PREFLIGHT_FAIL test-injected", file=_sys.stderr, flush=True)
            _sys.exit(2)

        mocker.patch("gpu_service.rest_server.preflight_or_exit", side_effect=fake_preflight)

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 2
        warmup.assert_not_called()
        fake_serve.assert_not_called()
        fake_app.run.assert_not_called()
