"""Tests for the dispatcher factory that wires REST → task_runner.

``main()`` itself (Flask serve + model warmup + SIGTERM handler install)
is integration territory; here we cover the small glue function that
binds workdir/http/concat/pipeline into a single ``dispatch(payload)``
closure for ``create_app``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

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
