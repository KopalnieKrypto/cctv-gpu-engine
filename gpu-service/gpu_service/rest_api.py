"""GPU-agent REST contract for the gpu-exchange integration (issue #25).

Three routes on :5003:

* ``POST /analyze``    accept a multi-chunk task, return 202 immediately
* ``GET  /healthz``    readiness gate (model + CUDA loaded)
* ``GET  /status/:id`` task state machine — no result payload, by design

The HTTP layer is intentionally thin: collaborators (readiness probe, task
registry, dispatcher) are injected so the suite runs on macOS without
ffmpeg/CUDA/boto3.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from flask import Flask, jsonify, request

from gpu_service.tenant_url import TenantPrefixError, extract_tenant_id

Payload = dict[str, Any]
Dispatcher = Callable[[Payload], None]


class Readiness:
    """Thread-safe readiness flag flipped once the GPU stack is warm.

    Worker startup loads the YOLO ONNX model and probes CUDA before calling
    :meth:`mark_ready`. Until then ``/healthz`` returns 503 so the gpu-agent
    keeps polling instead of dispatching a task to a half-booted container.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ready = False

    def mark_ready(self) -> None:
        with self._lock:
            self._ready = True

    def is_ready(self) -> bool:
        with self._lock:
            return self._ready


class TaskRegistry:
    """In-memory task state store.

    One container instance per task is the deployment model — gpu-agent
    spins us up, posts /analyze, polls /status/:id, then docker-stops us.
    So a process-local dict is enough; no persistence required.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._states: dict[str, dict[str, Any]] = {}

    def get(self, task_id: str) -> dict[str, Any] | None:
        with self._lock:
            entry = self._states.get(task_id)
            return dict(entry) if entry is not None else None

    def set_queued(self, task_id: str) -> None:
        with self._lock:
            self._states[task_id] = {"state": "queued"}

    def set_running(self, task_id: str, progress: float = 0.0) -> None:
        with self._lock:
            self._states[task_id] = {"state": "running", "progress": progress}

    def set_progress(self, task_id: str, progress: float) -> None:
        with self._lock:
            entry = self._states.setdefault(task_id, {"state": "running"})
            entry["state"] = "running"
            entry["progress"] = progress

    def set_completed(self, task_id: str) -> None:
        with self._lock:
            self._states[task_id] = {"state": "completed"}

    def set_failed(self, task_id: str, error: str) -> None:
        with self._lock:
            self._states[task_id] = {"state": "failed", "error": error}

    def running_task_ids(self) -> list[str]:
        with self._lock:
            return [tid for tid, st in self._states.items() if st.get("state") == "running"]

    def active_task_ids(self) -> list[str]:
        """Tasks that have no terminal state yet — queued *or* running.

        A ``queued`` task (accepted but not yet picked up by the single
        executor) is just as much in-flight as a running one; a SIGTERM
        sweep must fail both so nothing is left without a terminal state.
        """
        with self._lock:
            return [
                tid for tid, st in self._states.items() if st.get("state") in ("queued", "running")
            ]


def terminate_running_tasks(registry: TaskRegistry) -> None:
    """Flip every in-flight task to ``state: failed, error: terminated``.

    Invoked from a SIGTERM handler when the gpu-agent docker-stops the
    container. Both queued and running tasks are swept; tasks that
    completed cleanly stay completed; tasks that already failed keep their
    original error.
    """
    for task_id in registry.active_task_ids():
        registry.set_failed(task_id, error="terminated")


class _ValidationError(ValueError):
    """Raised by the body parser to be converted into a 400 by the route."""


def _validate_analyze_payload(body: Any) -> Payload:
    if not isinstance(body, dict):
        raise _ValidationError("body must be a JSON object")

    task_id = body.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        raise _ValidationError("task_id must be a non-empty string")

    urls = body.get("input_presigned_urls")
    if not isinstance(urls, list) or not urls:
        raise _ValidationError("input_presigned_urls must be a non-empty list")
    for u in urls:
        if not isinstance(u, str) or not u:
            raise _ValidationError("input_presigned_urls entries must be non-empty strings")

    result_url = body.get("result_presigned_url")
    if not isinstance(result_url, str) or not result_url:
        raise _ValidationError("result_presigned_url must be a non-empty string")

    params = body.get("params", {})
    if not isinstance(params, dict):
        raise _ValidationError("params must be an object")

    return {
        "task_id": task_id,
        "input_presigned_urls": urls,
        "result_presigned_url": result_url,
        "params": params,
    }


def create_app(
    readiness: Readiness,
    registry: TaskRegistry | None = None,
    dispatch: Dispatcher | None = None,
) -> Flask:
    """Build the Flask app wiring routes to the injected collaborators."""
    app = Flask(__name__)
    registry = registry or TaskRegistry()
    dispatch = dispatch or (lambda _payload: None)

    @app.get("/healthz")
    def healthz():  # type: ignore[unused-ignore]
        if readiness.is_ready():
            return jsonify({"ok": True}), 200
        return jsonify({"ok": False}), 503

    @app.post("/analyze")
    def analyze():  # type: ignore[unused-ignore]
        body = request.get_json(silent=True)
        try:
            payload = _validate_analyze_payload(body)
        except _ValidationError as e:
            return jsonify({"error": str(e)}), 400

        try:
            extract_tenant_id(payload["result_presigned_url"], payload["task_id"])
        except TenantPrefixError as e:
            return jsonify({"error": f"tenant prefix check failed: {e}"}), 400

        task_id = payload["task_id"]
        # Idempotency: a gpu-agent retry (or warm-container reuse) may POST
        # the same task_id while it is still queued/running. Accept it again
        # (202) but do not re-dispatch — otherwise a long VLM job runs twice.
        existing = registry.get(task_id)
        if existing is not None and existing.get("state") in ("queued", "running"):
            return jsonify({"accepted": True, "task_id": task_id}), 202

        registry.set_queued(task_id)
        dispatch(payload)
        return jsonify({"accepted": True, "task_id": task_id}), 202

    @app.get("/status/<task_id>")
    def status(task_id: str):  # type: ignore[unused-ignore]
        entry = registry.get(task_id)
        if entry is None:
            return jsonify({"error": "task not found"}), 404
        return jsonify(entry), 200

    return app
