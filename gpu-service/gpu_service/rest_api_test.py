"""Tests for the GPU-agent REST contract (issue #25).

The container exposes :5003 to the gpu-agent in the gpu-exchange repo with
three routes:

* ``POST /analyze``    — accept a task, run async, return 202
* ``GET  /healthz``    — readiness gate (model + CUDA loaded)
* ``GET  /status/:id`` — task state machine, no result payload

All tests inject collaborators (readiness probe, task runner, registry) so
the suite runs on macOS without ffmpeg, CUDA, or boto3 reachable.
"""

from __future__ import annotations

from gpu_service.rest_api import Readiness, TaskRegistry, create_app

VALID_TASK_ID = "11111111-2222-3333-4444-555555555555"
VALID_RESULT_URL = (
    f"https://r2.example.com/tenants/acme/results/{VALID_TASK_ID}/report.html?sig=abc"
)


def _valid_payload() -> dict:
    return {
        "task_id": VALID_TASK_ID,
        "input_presigned_urls": ["https://r2.example.com/get/chunk_001.mp4"],
        "result_presigned_url": VALID_RESULT_URL,
        "params": {},
    }


def _make_client(*, ready: bool = True, dispatcher=None):
    readiness = Readiness()
    if ready:
        readiness.mark_ready()
    registry = TaskRegistry()
    app = create_app(
        readiness=readiness,
        registry=registry,
        dispatch=dispatcher or (lambda payload: None),
    )
    return app.test_client(), registry


class TestHealthz:
    def test_returns_503_while_not_ready(self) -> None:
        readiness = Readiness()
        app = create_app(readiness=readiness)
        client = app.test_client()

        response = client.get("/healthz")

        assert response.status_code == 503
        assert response.get_json() == {"ok": False}

    def test_returns_200_once_marked_ready(self) -> None:
        readiness = Readiness()
        app = create_app(readiness=readiness)
        client = app.test_client()

        readiness.mark_ready()
        response = client.get("/healthz")

        assert response.status_code == 200
        assert response.get_json() == {"ok": True}


class TestAnalyzeValidation:
    def test_rejects_missing_task_id(self) -> None:
        client, _ = _make_client()
        payload = _valid_payload()
        del payload["task_id"]

        response = client.post("/analyze", json=payload)

        assert response.status_code == 400
        assert "task_id" in response.get_json()["error"]

    def test_rejects_empty_input_urls(self) -> None:
        client, _ = _make_client()
        payload = _valid_payload()
        payload["input_presigned_urls"] = []

        response = client.post("/analyze", json=payload)

        assert response.status_code == 400
        assert "input_presigned_urls" in response.get_json()["error"]

    def test_rejects_non_list_input_urls(self) -> None:
        client, _ = _make_client()
        payload = _valid_payload()
        payload["input_presigned_urls"] = "https://not-a-list"

        response = client.post("/analyze", json=payload)

        assert response.status_code == 400

    def test_rejects_missing_result_url(self) -> None:
        client, _ = _make_client()
        payload = _valid_payload()
        del payload["result_presigned_url"]

        response = client.post("/analyze", json=payload)

        assert response.status_code == 400

    def test_rejects_non_json_body(self) -> None:
        client, _ = _make_client()

        response = client.post(
            "/analyze",
            data="not json",
            content_type="application/json",
        )

        assert response.status_code == 400


class TestAnalyzeTenantGuard:
    def test_rejects_cross_tenant_result_url(self) -> None:
        client, _ = _make_client()
        payload = _valid_payload()
        # Path lacks the required tenants/<tid>/results/<task_id>/ shape.
        payload["result_presigned_url"] = "https://r2.example.com/uploads/anywhere.html"

        response = client.post("/analyze", json=payload)

        assert response.status_code == 400
        assert "tenant" in response.get_json()["error"].lower()

    def test_rejects_task_id_mismatch_in_result_url(self) -> None:
        client, _ = _make_client()
        payload = _valid_payload()
        # URL claims a different task slot than the payload — block.
        payload["result_presigned_url"] = (
            "https://r2.example.com/tenants/acme/results/other-task/output.html"
        )

        response = client.post("/analyze", json=payload)

        assert response.status_code == 400


class TestStatus:
    def test_returns_404_for_unknown_task(self) -> None:
        client, _ = _make_client()

        response = client.get("/status/never-seen-this-one")

        assert response.status_code == 404

    def test_returns_running_with_progress(self) -> None:
        client, registry = _make_client()
        registry.set_running("t-1", progress=0.42)

        response = client.get("/status/t-1")

        assert response.status_code == 200
        body = response.get_json()
        assert body == {"state": "running", "progress": 0.42}

    def test_returns_completed_without_result_payload(self) -> None:
        # AC #5: completed responses MUST NOT include result_html. The
        # gpu-agent learned the result key when it issued the presigned
        # URL — it trusts that URL has content once state=completed.
        client, registry = _make_client()
        registry.set_completed("t-2")

        response = client.get("/status/t-2")

        assert response.status_code == 200
        body = response.get_json()
        assert body == {"state": "completed"}
        assert "result_html" not in body
        assert "report" not in body

    def test_returns_failed_with_error_string(self) -> None:
        client, registry = _make_client()
        registry.set_failed("t-3", error="ffmpeg concat exit 1")

        response = client.get("/status/t-3")

        assert response.status_code == 200
        body = response.get_json()
        assert body == {"state": "failed", "error": "ffmpeg concat exit 1"}


class TestAnalyzeAsyncDispatch:
    def test_returns_202_accepted_with_task_id(self) -> None:
        dispatched: list = []
        client, _ = _make_client(dispatcher=lambda payload: dispatched.append(payload))
        payload = _valid_payload()

        response = client.post("/analyze", json=payload)

        assert response.status_code == 202
        body = response.get_json()
        assert body == {"accepted": True, "task_id": VALID_TASK_ID}

    def test_dispatches_payload_to_injected_executor(self) -> None:
        dispatched: list = []
        client, _ = _make_client(dispatcher=lambda payload: dispatched.append(payload))
        payload = _valid_payload()

        client.post("/analyze", json=payload)

        assert len(dispatched) == 1
        assert dispatched[0]["task_id"] == VALID_TASK_ID
        assert dispatched[0]["input_presigned_urls"] == payload["input_presigned_urls"]
        assert dispatched[0]["result_presigned_url"] == payload["result_presigned_url"]

    def test_does_not_dispatch_when_validation_fails(self) -> None:
        dispatched: list = []
        client, _ = _make_client(dispatcher=lambda payload: dispatched.append(payload))
        payload = _valid_payload()
        del payload["task_id"]

        response = client.post("/analyze", json=payload)

        assert response.status_code == 400
        assert dispatched == []

    def test_does_not_dispatch_when_tenant_guard_rejects(self) -> None:
        dispatched: list = []
        client, _ = _make_client(dispatcher=lambda payload: dispatched.append(payload))
        payload = _valid_payload()
        payload["result_presigned_url"] = "https://r2.example.com/no-tenant.html"

        response = client.post("/analyze", json=payload)

        assert response.status_code == 400
        assert dispatched == []


class TestQueuedState:
    """Issue #59 — an accepted task must be observable as ``queued`` the
    instant ``/analyze`` returns 202, not only once the single-worker
    executor picks it up. Otherwise a task queued behind a 20-minute VLM
    job 404s for the whole duration and the gpu-agent re-dispatches it.
    """

    def test_status_is_queued_immediately_after_accept(self) -> None:
        # Dispatcher records the payload but never runs it — simulates the
        # executor queue backed up behind a long-running task.
        dispatched: list = []
        client, _ = _make_client(dispatcher=lambda payload: dispatched.append(payload))

        client.post("/analyze", json=_valid_payload())
        response = client.get(f"/status/{VALID_TASK_ID}")

        assert response.status_code == 200
        assert response.get_json() == {"state": "queued"}

    def test_queued_transitions_to_running(self) -> None:
        # Inline "executor": the dispatcher runs the task immediately, so
        # run_task's first action (set_running) overwrites the queued entry.
        readiness = Readiness()
        readiness.mark_ready()
        registry = TaskRegistry()
        app = create_app(
            readiness=readiness,
            registry=registry,
            dispatch=lambda payload: registry.set_running(payload["task_id"], progress=0.25),
        )
        client = app.test_client()

        client.post("/analyze", json=_valid_payload())
        response = client.get(f"/status/{VALID_TASK_ID}")

        assert response.status_code == 200
        assert response.get_json() == {"state": "running", "progress": 0.25}

    def test_duplicate_task_id_is_idempotent(self) -> None:
        # A gpu-agent retry (or warm-container reuse) may POST the same
        # task_id twice. The second POST must not enqueue a second
        # run_task — otherwise a 20-minute VLM job runs twice.
        dispatched: list = []
        client, _ = _make_client(dispatcher=lambda payload: dispatched.append(payload))
        payload = _valid_payload()

        first = client.post("/analyze", json=payload)
        second = client.post("/analyze", json=payload)

        assert first.status_code == 202
        assert second.status_code == 202
        assert second.get_json() == {"accepted": True, "task_id": VALID_TASK_ID}
        # Enqueued exactly once despite two accepted POSTs.
        assert len(dispatched) == 1


class TestSigtermCleanup:
    def test_running_tasks_flipped_to_failed_with_terminated_error(self) -> None:
        # gpu-agent calls `docker stop` (SIGTERM) when the job completes
        # or times out. Any task that was still mid-flight in this
        # container instance has to land in `failed` so the gpu-agent
        # surfaces a real signal instead of forever-running.
        from gpu_service.rest_api import terminate_running_tasks

        registry = TaskRegistry()
        registry.set_running("running-1", progress=0.4)
        registry.set_running("running-2", progress=0.9)
        registry.set_completed("done-1")
        registry.set_failed("already-failed", error="boom")

        terminate_running_tasks(registry)

        assert registry.get("running-1") == {"state": "failed", "error": "terminated"}
        assert registry.get("running-2") == {"state": "failed", "error": "terminated"}
        # Already-completed and already-failed are untouched.
        assert registry.get("done-1") == {"state": "completed"}
        assert registry.get("already-failed") == {"state": "failed", "error": "boom"}

    def test_sigterm_fails_queued_tasks_too(self) -> None:
        # Issue #59: a task still sitting in the executor queue (never
        # started) must also flip to failed on docker-stop — otherwise the
        # gpu-agent sees a queued task vanish with no terminal state.
        from gpu_service.rest_api import terminate_running_tasks

        registry = TaskRegistry()
        registry.set_queued("queued-1")
        registry.set_running("running-1", progress=0.5)

        terminate_running_tasks(registry)

        assert registry.get("queued-1") == {"state": "failed", "error": "terminated"}
        assert registry.get("running-1") == {"state": "failed", "error": "terminated"}
