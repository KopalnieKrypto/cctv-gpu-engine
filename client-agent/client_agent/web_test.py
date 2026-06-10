"""Tests for the client-agent Flask web UI (issue #7).

The web app exposes the operator-facing surface of the client-agent: an MP4
upload form, a job list with status badges, and a viewer/download for the
generated report. These tests pin behaviour through the public Flask
``test_client`` only — no real sockets, no real R2.

The R2 client is faked with an in-memory stand-in (same pattern as the
gpu-side dashboard tests). Mocks live only at system boundaries; boto3
itself is exercised in ``client_agent/r2_client_test.py``.
"""

from __future__ import annotations

from typing import Any

from client_agent.web import create_app


class FakeR2:
    """In-memory stand-in for the client-agent's R2 surface.

    Mirrors the duck-typed contract :class:`ClientR2Like` exposes:

    * ``upload_input_chunk(job_id, fileobj)`` — captures the uploaded
      file-like so tests can assert it was streamed (not read into bytes).
    * ``put_status(job_id, status_dict)``
    * ``list_all_job_statuses() -> list[tuple[str, dict]]``
    * ``get_status(job_id)``
    * ``get_report(job_id) -> bytes``
    """

    def __init__(
        self,
        jobs: list[tuple[str, dict[str, Any]]] | None = None,
        reports: dict[str, bytes] | None = None,
    ) -> None:
        self._jobs: list[tuple[str, dict[str, Any]]] = list(jobs or [])
        self._reports: dict[str, bytes] = dict(reports or {})
        self.uploaded: list[tuple[str, Any]] = []

    def upload_input_chunk(self, job_id: str, fileobj: Any) -> str:
        self.uploaded.append((job_id, fileobj))
        return f"surveillance-jobs/{job_id}/input/chunk_001.mp4"

    def put_status(self, job_id: str, status: dict[str, Any]) -> None:
        # Replace if exists, append otherwise — mirrors R2 put semantics.
        self._jobs = [(jid, s) for jid, s in self._jobs if jid != job_id]
        self._jobs.append((job_id, status))

    def list_all_job_statuses(self) -> list[tuple[str, dict[str, Any]]]:
        return list(self._jobs)

    def get_status(self, job_id: str) -> dict[str, Any] | None:
        for jid, status in self._jobs:
            if jid == job_id:
                return status
        return None

    def get_report(self, job_id: str) -> bytes:
        return self._reports[job_id]


def _make_app(
    fake: FakeR2 | None = None,
    job_id: str = "job-test123",
):
    """Build a Flask app wired to a FakeR2 with a deterministic job_id factory."""
    fake = fake or FakeR2()
    app = create_app(fake, job_id_factory=lambda: job_id)
    app.config["TESTING"] = True
    return app, fake


# ----- 1. GET / shows upload form -----


def test_get_root_returns_upload_form() -> None:
    """`GET /` renders an HTML form that accepts a file named ``mp4`` via
    multipart/form-data — the operator-facing entry point for issue #7."""
    app, _ = _make_app()
    client = app.test_client()

    resp = client.get("/")

    assert resp.status_code == 200
    assert resp.mimetype == "text/html"
    body = resp.get_data(as_text=True)
    assert "<form" in body
    assert 'enctype="multipart/form-data"' in body
    assert 'name="mp4"' in body
    assert 'type="file"' in body


# ----- 2. POST /upload streams file-like to R2 and redirects to /jobs -----


def test_post_upload_streams_file_like_to_r2_and_redirects() -> None:
    """The uploaded MP4 must reach R2 as a *file-like object*, not as bytes
    materialised in process memory.

    Acceptance criterion (#7 testing focus): "2GB+ MP4 completes without
    timeout (multipart)". Reading a 2 GB request body into RAM would either
    OOM the container or trip Werkzeug's ``MAX_CONTENT_LENGTH``. We assert
    the contract by checking the object passed to ``upload_input_chunk``
    has a ``read`` method (so boto3's multipart streaming can drain it
    chunk-by-chunk) instead of being a ``bytes`` instance.
    """
    from io import BytesIO

    app, fake = _make_app(job_id="job-fixed-1")
    client = app.test_client()

    resp = client.post(
        "/upload",
        data={"mp4": (BytesIO(b"fake mp4 bytes"), "camera-01.mp4", "video/mp4")},
        content_type="multipart/form-data",
    )

    # Redirect to job list — POST/redirect/GET so refresh doesn't re-upload.
    assert resp.status_code == 302
    assert resp.headers["Location"].endswith("/jobs")

    # Exactly one upload happened, against the deterministic job_id.
    assert len(fake.uploaded) == 1
    job_id, fileobj = fake.uploaded[0]
    assert job_id == "job-fixed-1"

    # The crucial part: a *file-like* arrived, not bytes. boto3's
    # upload_fileobj iterates .read(chunk_size); a bytes instance would force
    # the whole upload into memory.
    assert hasattr(fileobj, "read"), f"expected a file-like, got {type(fileobj)!r}"
    assert not isinstance(fileobj, (bytes, bytearray))


# ----- 3. POST /upload writes status.json with status=pending -----


def test_post_upload_writes_status_pending() -> None:
    """After streaming the MP4 to R2, the agent must write a ``status.json``
    with ``status: pending`` so the gpu-service worker picks the job up.

    SPEC §6.2 / §7.4: status.json is the job-coordination handshake. Without
    it the worker's ``list_pending_job_ids`` walk would never see the new job.
    """
    from io import BytesIO

    app, fake = _make_app(job_id="job-fixed-3")
    client = app.test_client()

    client.post(
        "/upload",
        data={"mp4": (BytesIO(b"x"), "cam.mp4", "video/mp4")},
        content_type="multipart/form-data",
    )

    status = fake.get_status("job-fixed-3")
    assert status is not None, "no status.json was written"
    assert status["status"] == "pending"
    assert status["job_id"] == "job-fixed-3"


# ----- 4. POST /upload validates input -----


def test_post_upload_without_file_returns_400() -> None:
    """No ``mp4`` field in the multipart body → 400 with a readable message,
    and *no* upload to R2.

    The form's ``required`` attribute prevents this from a real browser, but
    a curl/automation client could still POST without a file. We refuse
    early instead of letting Werkzeug raise a generic 400.
    """
    app, fake = _make_app()
    client = app.test_client()

    resp = client.post("/upload", data={}, content_type="multipart/form-data")

    assert resp.status_code == 400
    assert b"mp4" in resp.data.lower()  # error mentions the missing field
    assert fake.uploaded == []
    assert fake.list_all_job_statuses() == []


def test_post_upload_with_non_mp4_filename_returns_400() -> None:
    """Reject obviously-wrong file extensions before touching R2.

    The pipeline only handles MP4. Letting an .exe or .jpg through would
    result in a job that the gpu-service worker picks up, fails on, and
    surfaces in the dashboard as a noisy ``failed`` row. Cheaper to refuse
    at the edge.
    """
    from io import BytesIO

    app, fake = _make_app()
    client = app.test_client()

    resp = client.post(
        "/upload",
        data={"mp4": (BytesIO(b"not a video"), "evil.exe", "application/octet-stream")},
        content_type="multipart/form-data",
    )

    assert resp.status_code == 400
    assert fake.uploaded == []
    assert fake.list_all_job_statuses() == []


# ----- 5. POST /upload surfaces R2 errors as a 500 with a clear message -----


def test_post_upload_returns_500_when_r2_raises() -> None:
    """When the R2 client blows up (bad creds, network) the UI must return
    a 500 with the error string visible — *not* a Flask traceback page or a
    bare 502/500 with no body.

    Acceptance criterion (#7 testing focus): "Invalid R2 credentials → UI
    shows clear error". The exception type doesn't matter; what matters is
    that the operator sees *something readable* in the response body so they
    can correlate with their R2 dashboard.
    """
    from io import BytesIO

    class ExplodingR2(FakeR2):
        def upload_input_chunk(self, job_id: str, fileobj: Any) -> str:
            raise RuntimeError(
                "InvalidAccessKeyId: the AWS Access Key Id you provided does not exist"
            )

    fake = ExplodingR2()
    app = create_app(fake, job_id_factory=lambda: "job-doomed")
    app.config["TESTING"] = True
    # Disable Flask's debug-mode passthrough so the framework actually
    # returns the 500 response we're testing instead of re-raising.
    app.config["PROPAGATE_EXCEPTIONS"] = False
    client = app.test_client()

    resp = client.post(
        "/upload",
        data={"mp4": (BytesIO(b"x"), "cam.mp4", "video/mp4")},
        content_type="multipart/form-data",
    )

    assert resp.status_code == 500
    body = resp.get_data(as_text=True)
    assert "InvalidAccessKeyId" in body
    # No status.json should be written when the upload itself failed —
    # otherwise the worker would pick up a job whose chunk doesn't exist.
    assert fake.list_all_job_statuses() == []


# ----- 6. GET /jobs empty state -----


def test_get_jobs_empty_bucket_shows_placeholder() -> None:
    """First-launch state — bucket is empty, page must say so explicitly
    instead of rendering an empty table that looks like a broken UI."""
    app, _ = _make_app()
    client = app.test_client()

    resp = client.get("/jobs")

    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "no jobs yet" in body
    assert "<tbody>" not in body


# ----- 7. GET /jobs renders all four status badges -----


def test_get_jobs_renders_all_status_badges() -> None:
    """Every status from SPEC §6.2 must render with a CSS class the
    operator can spot at a glance: pending, processing, completed, failed.

    The test asserts on substrings rather than DOM structure so future
    template tweaks (column reordering, prettier badges) don't break it.
    """
    fake = FakeR2(
        [
            ("job-pend", {"status": "pending", "updated_at": "2026-04-08T08:00:00Z"}),
            ("job-proc", {"status": "processing", "updated_at": "2026-04-08T09:00:00Z"}),
            ("job-done", {"status": "completed", "updated_at": "2026-04-08T10:00:00Z"}),
            ("job-fail", {"status": "failed", "updated_at": "2026-04-08T11:00:00Z"}),
        ]
    )
    app, _ = _make_app(fake=fake)
    client = app.test_client()

    body = client.get("/jobs").get_data(as_text=True)

    for jid in ("job-pend", "job-proc", "job-done", "job-fail"):
        assert jid in body, f"missing job_id {jid} in /jobs render"
    for css in (
        "status-pending",
        "status-processing",
        "status-completed",
        "status-failed",
    ):
        assert css in body, f"missing CSS class {css} in /jobs render"


def test_get_jobs_sorted_by_updated_at_descending() -> None:
    """Newest jobs first — operator wants the latest at the top."""
    fake = FakeR2(
        [
            ("oldest", {"status": "completed", "updated_at": "2026-04-08T08:00:00Z"}),
            ("newest", {"status": "completed", "updated_at": "2026-04-08T12:00:00Z"}),
            ("middle", {"status": "completed", "updated_at": "2026-04-08T10:00:00Z"}),
        ]
    )
    app, _ = _make_app(fake=fake)
    body = app.test_client().get("/jobs").get_data(as_text=True)

    # Order is enforced by checking that 'newest' appears before 'middle'
    # which appears before 'oldest' in the rendered HTML.
    assert body.index("newest") < body.index("middle") < body.index("oldest")


# ----- 8. GET /jobs auto-refresh every 10 seconds -----


def test_get_jobs_has_meta_refresh_10s() -> None:
    """Acceptance criterion: "Auto-refresh every 10 seconds". We use a
    plain ``<meta http-equiv="refresh" content="10">`` so the page stays
    JS-free and works in any browser including locked-down kiosks."""
    app, _ = _make_app()
    body = app.test_client().get("/jobs").get_data(as_text=True).lower()

    assert 'http-equiv="refresh"' in body
    assert 'content="10"' in body


# ----- 9. Report links only render for completed jobs -----


def test_get_jobs_report_links_only_for_completed_jobs() -> None:
    """A pending/processing/failed job has no report yet — clicking through
    would 404 and confuse the operator. Only completed rows expose the
    view + download links."""
    fake = FakeR2(
        [
            ("job-pend", {"status": "pending", "updated_at": "2026-04-08T08:00:00Z"}),
            ("job-done", {"status": "completed", "updated_at": "2026-04-08T10:00:00Z"}),
            ("job-fail", {"status": "failed", "updated_at": "2026-04-08T11:00:00Z"}),
        ]
    )
    app, _ = _make_app(fake=fake)
    body = app.test_client().get("/jobs").get_data(as_text=True)

    # Completed: both links are present.
    assert "/jobs/job-done/report" in body
    assert "/jobs/job-done/report/download" in body
    # Non-completed: no links pointing at their report endpoints.
    assert "/jobs/job-pend/report" not in body
    assert "/jobs/job-fail/report" not in body


# ----- 10. GET /jobs/<id>/report — inline HTML proxy -----


def test_get_report_returns_html_inline() -> None:
    """`GET /jobs/<id>/report` proxies the HTML report from R2 verbatim
    with ``text/html`` Content-Type so the browser renders it inline.

    The report is a *standalone* HTML file (vendored Chart.js, base64
    images — see CLAUDE.md "Reports" rule) so we never need to rewrite asset
    URLs or set up a static path."""
    report_html = b"<!doctype html><html><body><h1>Report job-x</h1></body></html>"
    fake = FakeR2(
        jobs=[("job-x", {"status": "completed", "updated_at": "2026-04-08T10:00:00Z"})],
        reports={"job-x": report_html},
    )
    app, _ = _make_app(fake=fake)

    resp = app.test_client().get("/jobs/job-x/report")

    assert resp.status_code == 200
    assert resp.mimetype == "text/html"
    assert resp.data == report_html
    # Inline view: no attachment disposition.
    assert "attachment" not in resp.headers.get("Content-Disposition", "")


# ----- 11. GET /jobs/<id>/report/download -----


def test_get_report_download_sets_attachment_disposition() -> None:
    """`GET /jobs/<id>/report/download` returns the same body as the inline
    endpoint but with ``Content-Disposition: attachment`` so the browser
    saves it instead of rendering. Filename includes the job_id so the
    operator can match downloaded files back to jobs without renaming."""
    report_html = b"<!doctype html><html><body>Report</body></html>"
    fake = FakeR2(
        jobs=[("job-y", {"status": "completed", "updated_at": "2026-04-08T10:00:00Z"})],
        reports={"job-y": report_html},
    )
    app, _ = _make_app(fake=fake)

    resp = app.test_client().get("/jobs/job-y/report/download")

    assert resp.status_code == 200
    assert resp.mimetype == "text/html"
    assert resp.data == report_html
    disposition = resp.headers.get("Content-Disposition", "")
    assert "attachment" in disposition
    assert "job-y" in disposition


# ----- 12. Report endpoints return 404 when the report doesn't exist -----


def test_get_report_returns_404_when_missing() -> None:
    """Job exists but its report.html isn't in R2 yet (worker still
    processing, or upload failed). The viewer must 404 cleanly instead of
    raising a 500 — the operator opens the link from /jobs and a 500 looks
    like a bug in the agent itself."""
    fake = FakeR2(
        jobs=[("job-z", {"status": "completed", "updated_at": "2026-04-08T10:00:00Z"})],
        reports={},  # no report uploaded
    )
    app, _ = _make_app(fake=fake)

    inline = app.test_client().get("/jobs/job-z/report")
    download = app.test_client().get("/jobs/job-z/report/download")

    assert inline.status_code == 404
    assert download.status_code == 404


# ----- 13. Two POST /upload requests create distinct jobs -----


def test_two_uploads_create_distinct_job_ids() -> None:
    """Acceptance criterion: "Concurrent uploads: two at same time → separate
    jobs created". Each request must allocate a fresh job_id from the factory
    so the two MP4s land at different R2 keys and don't clobber each other.

    We exercise the *default* (uuid4-based) factory here — the rest of the
    suite pins it for determinism, but this test is the one that verifies
    we never bake a constant in by accident."""
    from io import BytesIO

    fake = FakeR2()
    # No job_id_factory override → uses the production uuid4-based default.
    app = create_app(fake)
    app.config["TESTING"] = True
    test_client = app.test_client()

    test_client.post(
        "/upload",
        data={"mp4": (BytesIO(b"a"), "cam-a.mp4", "video/mp4")},
        content_type="multipart/form-data",
    )
    test_client.post(
        "/upload",
        data={"mp4": (BytesIO(b"b"), "cam-b.mp4", "video/mp4")},
        content_type="multipart/form-data",
    )

    job_ids = [jid for jid, _ in fake.uploaded]
    assert len(job_ids) == 2
    assert job_ids[0] != job_ids[1], "two uploads collapsed onto the same job_id"
    # And both made it into status.json so the worker will see both.
    statuses = {jid for jid, _ in fake.list_all_job_statuses()}
    assert statuses == set(job_ids)


# ===== RTSP recording surface (issue #8) =====


class FakeRecorder:
    """In-memory stand-in for :class:`client_agent.recorder.Recorder`.

    Captures every ``start``/``stop`` call so the route tests can assert
    on the exact arguments. ``status_state`` lets a test put the fake
    "in flight" before the request to exercise the busy-rejection path,
    and ``probe_result`` controls /test-connection's outcome.
    """

    def __init__(
        self,
        *,
        status_state: str = "idle",
        probe_ok: bool = True,
        probe_message: str = "",
    ) -> None:
        from client_agent.recorder import RecorderStatus

        self._snapshot = RecorderStatus(state=status_state)
        self.probe_ok = probe_ok
        self.probe_message = probe_message
        self.starts: list[dict[str, Any]] = []
        self.stops: int = 0
        self.probes: list[str] = []

    def start(self, *, url: str, duration_s: int) -> str:
        from client_agent.recorder import RecorderBusy

        if self._snapshot.state in ("recording", "uploading"):
            raise RecorderBusy("busy")
        self.starts.append({"url": url, "duration_s": duration_s})
        return "job-rec-fake"

    def stop(self) -> None:
        self.stops += 1

    def status(self):
        return self._snapshot

    def probe(self, url: str, *, timeout: float):
        from client_agent.recorder import ProbeResult

        self.probes.append(url)
        return ProbeResult(ok=self.probe_ok, message=self.probe_message)


def _make_app_with_recorder(
    fake: FakeR2 | None = None,
    recorder: FakeRecorder | None = None,
):
    """Same as ``_make_app`` but also wires a fake recorder."""
    fake = fake or FakeR2()
    recorder = recorder or FakeRecorder()
    app = create_app(fake, job_id_factory=lambda: "job-test123", recorder=recorder)
    app.config["TESTING"] = True
    return app, fake, recorder


# ----- 14. GET / shows RTSP recording form alongside the upload form -----


def test_get_root_includes_rtsp_recording_form() -> None:
    """The same landing page must offer both flows: MP4 upload (already
    tested) *and* RTSP recording. The recording form needs an URL field,
    a unified ``duration_s`` selector covering both sub-hour smoke windows
    (5/15/30/45m) and the original hourly slots (1/2/4/8h), and a submit
    that posts to /start. We assert on substrings so future template
    tweaks don't break this test."""
    app, _, _ = _make_app_with_recorder()
    body = app.test_client().get("/").get_data(as_text=True)

    # MP4 upload form still present.
    assert 'name="mp4"' in body
    # RTSP form additions:
    assert 'action="/start"' in body
    assert 'name="rtsp_url"' in body
    assert 'name="duration_s"' in body
    # Test-connection button triggers JS fetch, not a separate form.
    assert "testConnection()" in body
    for seconds in ("300", "900", "1800", "2700", "3600", "7200", "14400", "28800"):
        assert f'value="{seconds}"' in body


# ----- 15. POST /test-connection — happy path returns ok=true -----


def test_post_test_connection_returns_ok_true_when_probe_succeeds() -> None:
    """`POST /test-connection` is a no-side-effects probe — the operator
    presses "test" before committing to a recording, expects a synchronous
    yes/no within the ffmpeg timeout. We return JSON so the form can be
    upgraded with JS later without changing the contract."""
    fake_recorder = FakeRecorder(probe_ok=True)
    app, _, _ = _make_app_with_recorder(recorder=fake_recorder)

    resp = app.test_client().post(
        "/test-connection", data={"rtsp_url": "rtsp://camera.local/stream"}
    )

    assert resp.status_code == 200
    assert resp.is_json
    body = resp.get_json()
    assert body["ok"] is True
    assert fake_recorder.probes == ["rtsp://camera.local/stream"]


def test_post_test_connection_returns_failure_with_message() -> None:
    """A failed probe still returns 200 (the request itself succeeded —
    it's the *probe* that failed) so the form's JS can read the JSON
    body without treating it as an HTTP error. The message must surface
    in the body so the operator can tell *why* the connection failed."""
    fake_recorder = FakeRecorder(probe_ok=False, probe_message="Connection refused")
    app, _, _ = _make_app_with_recorder(recorder=fake_recorder)

    resp = app.test_client().post("/test-connection", data={"rtsp_url": "rtsp://nope.local/stream"})

    assert resp.status_code == 200
    body = resp.get_json()
    assert body["ok"] is False
    assert "Connection refused" in body["message"]


# ----- 16. POST /start — happy path triggers recorder and redirects to /jobs -----


def test_post_start_invokes_recorder_and_redirects_to_jobs() -> None:
    """Acceptance criterion (#8): "Recording: ffmpeg stream copy from
    RTSP URL for selected duration". The route's job is to translate the
    form fields into a recorder.start call and follow the same
    POST/redirect/GET pattern the upload form uses, so a refresh doesn't
    re-launch the recording."""
    fake_recorder = FakeRecorder()
    app, _, _ = _make_app_with_recorder(recorder=fake_recorder)

    resp = app.test_client().post(
        "/start",
        data={"rtsp_url": "rtsp://camera.local/stream", "duration_s": str(4 * 3600)},
    )

    assert resp.status_code == 302
    assert resp.headers["Location"].endswith("/jobs")
    assert fake_recorder.starts == [{"url": "rtsp://camera.local/stream", "duration_s": 4 * 3600}]


def test_post_start_returns_409_when_recorder_busy() -> None:
    """Acceptance criterion (#8): "second recording while one active →
    rejected". 409 Conflict is the right HTTP shape for "the resource is
    in a state that prevents this operation". The body must be readable
    so the operator sees what's wrong without opening devtools."""
    fake_recorder = FakeRecorder(status_state="recording")
    app, _, _ = _make_app_with_recorder(recorder=fake_recorder)

    resp = app.test_client().post(
        "/start",
        data={"rtsp_url": "rtsp://camera.local/stream", "duration_s": "3600"},
    )

    assert resp.status_code == 409
    assert b"busy" in resp.data.lower()
    assert fake_recorder.starts == []


def test_post_start_rejects_unsupported_duration() -> None:
    """#8 + sub-hour extension: supported durations are the presets
    {5,15,30,45 min, 1,2,4,8 h} expressed in seconds. Anything else is a
    misconfigured client (or a curl probe) and must be refused *before*
    spawning ffmpeg — otherwise an honest typo could pin the recorder for
    an unintended duration. We also guard against the pre-extension
    ``duration_h`` field so a stale client fails loudly instead of
    silently starting a 1-second recording."""
    fake_recorder = FakeRecorder()
    app, _, _ = _make_app_with_recorder(recorder=fake_recorder)
    test_client = app.test_client()

    # Raw-value rejections: zero, negative, non-numeric, unsupported ints,
    # and legacy hourly values that happen to also be invalid as seconds
    # (e.g. "1" — the old duration_h=1, which would now mean "1 second").
    for bad in ("0", "-1", "abc", "1", "4", "8", "600", "3601", "99999"):
        resp = test_client.post(
            "/start",
            data={"rtsp_url": "rtsp://camera.local/stream", "duration_s": bad},
        )
        assert resp.status_code == 400, f"duration_s={bad!r} should be rejected"

    # Legacy field name must not be honored — a stale client sending
    # ``duration_h=1`` would otherwise be interpreted as "missing field".
    resp = test_client.post(
        "/start",
        data={"rtsp_url": "rtsp://camera.local/stream", "duration_h": "1"},
    )
    assert resp.status_code == 400

    assert fake_recorder.starts == []


def test_post_start_accepts_all_sub_hour_and_hourly_presets() -> None:
    """Positive coverage for the extended preset list: every supported
    value must reach the recorder with the exact ``duration_s`` the form
    submitted, so the HTTP allowlist and the recorder's length-agnostic
    core stay in lockstep. We reset the fake recorder's busy state
    between calls by instantiating a fresh app per preset — cheap, and
    it also proves the 409-busy branch isn't leaking between requests."""
    presets = (300, 900, 1800, 2700, 3600, 7200, 14400, 28800)
    for seconds in presets:
        fake_recorder = FakeRecorder()
        app, _, _ = _make_app_with_recorder(recorder=fake_recorder)

        resp = app.test_client().post(
            "/start",
            data={"rtsp_url": "rtsp://camera.local/stream", "duration_s": str(seconds)},
        )

        assert resp.status_code == 302, f"preset {seconds}s should be accepted"
        assert fake_recorder.starts == [
            {"url": "rtsp://camera.local/stream", "duration_s": seconds}
        ]


def test_post_stop_invokes_recorder_stop() -> None:
    """`POST /stop` is the operator's escape hatch — wired to whatever
    cancellation primitive the production Recorder has. The route just
    delegates and returns 200 so the form's submit button works without
    JavaScript."""
    fake_recorder = FakeRecorder()
    app, _, _ = _make_app_with_recorder(recorder=fake_recorder)

    resp = app.test_client().post("/stop")

    assert resp.status_code == 200
    assert fake_recorder.stops == 1


# ===== ONVIF discovery surface (issue #21) =====


def test_get_cameras_discover_returns_json_with_cameras_and_scanned_at() -> None:
    """`GET /cameras/discover` runs the injected discovery function and
    returns JSON ``{cameras: [...], scanned_at: <iso>, error: null}``.

    Acceptance criterion (#21): endpoint shape exactly as specified. The
    discovery function is duck-typed and injected via ``create_app`` so
    these tests never run real WS-Discovery — same pattern as the
    recorder/R2 client. Cameras serialize as plain dicts so the JS in the
    UI can render without a translation layer."""
    from client_agent.discovery import DiscoveredCamera

    cams = [
        DiscoveredCamera(
            ip="192.168.1.10",
            port=80,
            vendor="Hikvision",
            model="DS-2CD2042",
            rtsp_url="rtsp://192.168.1.10:554/Streaming/Channels/101",
            snapshot_url="http://192.168.1.10/Streaming/Channels/101/picture",
        ),
        DiscoveredCamera(
            ip="192.168.1.11",
            port=80,
            vendor="Dahua",
            model="IPC-HFW1230S",
            rtsp_url="rtsp://192.168.1.11:554/cam/realmonitor",
            snapshot_url=None,
        ),
    ]

    fake_r2 = FakeR2()
    app = create_app(fake_r2, discover_fn=lambda: cams)
    app.config["TESTING"] = True

    resp = app.test_client().get("/cameras/discover")

    assert resp.status_code == 200
    assert resp.is_json
    body = resp.get_json()
    assert body["error"] is None
    assert isinstance(body["scanned_at"], str) and body["scanned_at"].endswith("Z")
    assert body["cameras"] == [
        {
            "ip": "192.168.1.10",
            "port": 80,
            "vendor": "Hikvision",
            "model": "DS-2CD2042",
            "rtsp_url": "rtsp://192.168.1.10:554/Streaming/Channels/101",
            "snapshot_url": "http://192.168.1.10/Streaming/Channels/101/picture",
            "discovery_method": "onvif",
            "needs_manual_url": False,
        },
        {
            "ip": "192.168.1.11",
            "port": 80,
            "vendor": "Dahua",
            "model": "IPC-HFW1230S",
            "rtsp_url": "rtsp://192.168.1.11:554/cam/realmonitor",
            "snapshot_url": None,
            "discovery_method": "onvif",
            "needs_manual_url": False,
        },
    ]


def test_get_cameras_discover_without_discover_fn_returns_404() -> None:
    """Same shape as `/start` when no recorder is wired: a clear 404 with a
    readable message. Tests that don't care about discovery don't have to
    inject a fake."""
    fake_r2 = FakeR2()
    app = create_app(fake_r2)  # no discover_fn
    app.config["TESTING"] = True

    resp = app.test_client().get("/cameras/discover")

    assert resp.status_code == 404
    assert b"discovery" in resp.data.lower()


def test_get_cameras_discover_surfaces_errors_as_json_error_field() -> None:
    """When discovery raises (multicast blocked, ONVIF library failure,
    network blip) the endpoint must still return 200 + JSON with the
    error in the ``error`` field — never a 500 traceback.

    The UI does ``fetch().then(r => r.json())`` and renders ``error`` as a
    banner; a 500 with HTML body would break that contract. Issue #21:
    "error: null|string"."""

    def boom() -> list:
        raise RuntimeError("multicast not reachable: blocked by docker bridge")

    fake_r2 = FakeR2()
    app = create_app(fake_r2, discover_fn=boom)
    app.config["TESTING"] = True
    # Make sure Flask doesn't re-raise in TESTING mode and skip our handler.
    app.config["PROPAGATE_EXCEPTIONS"] = False

    resp = app.test_client().get("/cameras/discover")

    assert resp.status_code == 200
    assert resp.is_json
    body = resp.get_json()
    assert body["cameras"] == []
    assert isinstance(body["scanned_at"], str)
    assert "multicast not reachable" in body["error"]


def test_get_root_includes_camera_discovery_button_and_results_container() -> None:
    """The same landing page that hosts the upload + RTSP forms must also
    expose ONVIF discovery: a "Wykryj kamery" button, a results container,
    and JS that (a) fetches /cameras/discover, (b) renders each camera with
    a snapshot thumbnail (issue #21 design tweak — operator wybiera kamerę
    po obrazie), and (c) on click pastes the rtsp_url into the existing
    rtsp_url field. Substring assertions only — future template tweaks
    don't have to fight a brittle DOM test."""
    app, _, _ = _make_app_with_recorder()
    body = app.test_client().get("/").get_data(as_text=True)

    # Button + container are present and labeled in Polish (operator UX).
    assert "Wykryj kamery" in body
    # JS hits the new endpoint and uses snapshot_url for the thumbnail.
    assert "/cameras/discover" in body
    assert "snapshot_url" in body
    # Click on a result must paste the rtsp_url into the existing input —
    # otherwise the operator still has to copy/paste manually.
    assert "rtsp_url" in body
    # Vendor + model + IP must end up in the rendered list so the operator
    # can disambiguate between identical-snapshot cameras.
    for field in ("vendor", "model", "ip"):
        assert field in body, f"missing field reference {field!r} in UI"


def test_get_root_form_has_camera_ip_field_for_discovered_camera_path() -> None:
    """Issue #22 acceptance: ``UI: po kliknięciu wykrytej kamery formularz
    dostaje camera_ip zamiast pełnego URL; hasło nigdy nie pojawia się w
    DOM``. The recording form must expose a ``camera_ip`` field
    (typically a hidden input populated by JS) so the click-from-list
    flow can submit IP-only and let the server re-attach creds."""
    app, _, _ = _make_app_with_recorder()
    body = app.test_client().get("/").get_data(as_text=True)

    assert 'name="camera_ip"' in body


def test_get_root_js_click_sets_camera_ip_not_full_url() -> None:
    """Click on a discovered camera assigns the IP to the camera_ip field;
    it must NOT paste the full RTSP URL anywhere visible. The JS must
    also clear the rtsp_url field so a previous manual entry doesn't
    accidentally piggy-back onto the camera_ip submission (the server
    treats camera_ip as the priority signal — but a stale rtsp_url
    visible in the DOM violates the spirit of "hasło nie w DOM")."""
    app, _, _ = _make_app_with_recorder()
    body = app.test_client().get("/").get_data(as_text=True)

    # JS references camera_ip and assigns from cam.ip.
    assert "camera_ip" in body
    assert "cam.ip" in body
    # The previous behavior — pasting cam.rtsp_url into the rtsp_url
    # input on click — must be gone. We can't assert exact code shape,
    # so we assert the click handler clears the rtsp_url input.
    # ``rtsp_url'' clearing pattern: set value to '' or similar.
    # Substring assertion: look for either ``rtsp_url').value = ''`` or
    # ``rtsp_url').value=''``.
    assert any(
        snippet in body.replace(" ", "")
        for snippet in (
            "rtsp_url').value=''",
            'rtsp_url").value=""',
            'rtsp_url\').value=""',
        )
    ), "JS does not clear rtsp_url field on camera click — stale URL stays in DOM"


def test_get_root_renders_discovery_method_badge_in_js() -> None:
    """The JS that renders camera rows must show *which* discovery channel
    found each camera (ONVIF Stage 1 vs RTSP-scan Stage 2). Operators need
    this for two reasons: ONVIF is more trustworthy (vendor/model came from
    the device), and RTSP-scan rows often need creds populated in
    ``.env.client``. Substring assertions: we look for the
    ``discovery_method`` reference and the human-facing labels."""
    app, _, _ = _make_app_with_recorder()
    body = app.test_client().get("/").get_data(as_text=True)

    # JS reads cam.discovery_method from the response.
    assert "discovery_method" in body
    # Human labels for both channels.
    assert "ONVIF" in body
    assert "RTSP" in body


def test_get_cameras_discover_strips_credentials_from_rtsp_url() -> None:
    """Issue #22 acceptance: ``hasło nigdy nie pojawia się w DOM``. Even if
    the discovery layer baked creds into the URL (``_real_rtsp_scan``
    template, or some ONVIF firmwares returning ``user:pass@`` in
    GetStreamUri), the JSON shipped to the browser must be credential-free.
    The UI pairs the cred-less URL with the camera_ip and lets ``/start``
    re-attach creds server-side."""
    from client_agent.discovery import DiscoveredCamera

    cams = [
        DiscoveredCamera(
            ip="192.168.50.2",
            port=554,
            vendor="Hikvision",
            model="",
            rtsp_url="rtsp://admin:Secret1!@192.168.50.2:554/Streaming/Channels/101",
            snapshot_url=None,
            discovery_method="rtsp-scan",
        ),
    ]
    fake_r2 = FakeR2()
    app = create_app(fake_r2, discover_fn=lambda: cams)
    app.config["TESTING"] = True

    body = app.test_client().get("/cameras/discover").get_json()

    cam = body["cameras"][0]
    assert "Secret1" not in cam["rtsp_url"]
    assert "admin@" not in cam["rtsp_url"]
    assert "@" not in cam["rtsp_url"].split("/", 3)[2], (
        f"userinfo not stripped: {cam['rtsp_url']!r}"
    )
    assert cam["rtsp_url"] == "rtsp://192.168.50.2:554/Streaming/Channels/101"


def test_get_cameras_discover_passes_through_credless_urls_unchanged() -> None:
    """Already-cred-less URLs (the ONVIF happy path on most cameras) must
    flow through untouched — the strip helper is idempotent so a re-render
    or a debug pass through the response doesn't accidentally mutate the
    host/port/path."""
    from client_agent.discovery import DiscoveredCamera

    cams = [
        DiscoveredCamera(
            ip="192.168.1.10",
            port=80,
            vendor="Hikvision",
            model="DS-2CD2042",
            rtsp_url="rtsp://192.168.1.10:554/Streaming/Channels/101",
            snapshot_url=None,
        ),
    ]
    fake_r2 = FakeR2()
    app = create_app(fake_r2, discover_fn=lambda: cams)
    app.config["TESTING"] = True

    body = app.test_client().get("/cameras/discover").get_json()
    assert body["cameras"][0]["rtsp_url"] == "rtsp://192.168.1.10:554/Streaming/Channels/101"


# ===== needs_manual_url UI hint (issue #37) =====


def test_get_root_renders_manual_url_hint_for_unknown_vendor() -> None:
    """Issue #37 UI acceptance: the discovery results JS must branch on
    ``needs_manual_url`` and render a hint pointing the operator at the
    vendor app instead of the bogus ``rtsp://...:554/`` URL.

    Substring assertions only — the exact JS shape is free to evolve as long
    as (a) the JS references ``needs_manual_url`` so it can branch, and
    (b) the hint text mentions the vendor app so the operator knows where
    to find the real URL."""
    app, _, _ = _make_app_with_recorder()
    body = app.test_client().get("/").get_data(as_text=True)

    assert "needs_manual_url" in body, "JS does not branch on needs_manual_url"
    # Hint must point the operator at the vendor app — that's where the
    # per-device URI actually lives. Accept Polish or English wording so a
    # future i18n pass doesn't break the test.
    body_lower = body.lower()
    assert ("vendor app" in body_lower) or ("aplikacj" in body_lower), (
        "manual-URL hint does not mention the vendor app"
    )


# ===== Tuya-local UI badge + hint (issue #38) =====


def test_get_root_js_branches_on_tuya_local_discovery_method() -> None:
    """Issue #38 UI acceptance: the discovery results JS must distinguish
    Stage-3 Tuya broadcast rows from Stage-1/2 rows. Substring assertions
    only — the JS may use any rendering shape (badge, label, separate
    section) as long as it (a) references ``"tuya-local"`` so a branch
    exists, and (b) surfaces some operator-readable signal that the row
    came from Tuya so the manual-URL hint mentions enabling RTSP in the
    Setti+ / Tuya Smart app."""
    app, _, _ = _make_app_with_recorder()
    body = app.test_client().get("/").get_data(as_text=True)

    assert "tuya-local" in body, "JS does not branch on discovery_method='tuya-local'"
    # The hint text on Tuya rows must mention the operator-facing app name(s).
    # Accept either "Setti+" or "Tuya Smart" since both are common in PL.
    body_lower = body.lower()
    assert ("setti" in body_lower) or ("tuya smart" in body_lower), (
        "Tuya-local hint does not mention the Setti+ / Tuya Smart vendor app"
    )


# ===== needs_manual_url surfaced in JSON (issue #37) =====


def test_get_cameras_discover_includes_needs_manual_url_for_unknown_vendor() -> None:
    """Issue #37: when Stage-2 emits a camera with ``needs_manual_url=True``
    (Unknown-vendor host like AnyKa/Tuya/Setti+), the discovery JSON must
    carry the flag so the SPA can branch — render an editable input + hint
    instead of the bogus ``rtsp://...:554/`` URL.

    The cred-less ``rtsp_url=""`` must also flow through untouched —
    ``strip_credentials_from_url`` on an empty string mustn't accidentally
    rewrite it into something the SPA's truthiness check would mistake for
    a real URL."""
    from client_agent.discovery import DiscoveredCamera

    cams = [
        DiscoveredCamera(
            ip="192.168.1.198",
            port=554,
            vendor="Unknown (nginx-RTSP / per-device URI)",
            model="",
            rtsp_url="",
            snapshot_url=None,
            discovery_method="rtsp-scan",
            needs_manual_url=True,
        ),
    ]
    fake_r2 = FakeR2()
    app = create_app(fake_r2, discover_fn=lambda: cams)
    app.config["TESTING"] = True

    body = app.test_client().get("/cameras/discover").get_json()
    cam = body["cameras"][0]
    assert cam["needs_manual_url"] is True
    assert cam["rtsp_url"] == ""
    assert "Unknown" in cam["vendor"]


def test_get_cameras_discover_defaults_needs_manual_url_false_for_onvif() -> None:
    """Backward-compat: ONVIF-enriched rows (default ``needs_manual_url=False``)
    keep flowing through as ``false`` so the SPA's existing branches don't
    have to special-case the field being absent."""
    from client_agent.discovery import DiscoveredCamera

    cams = [
        DiscoveredCamera(
            ip="192.168.1.10",
            port=80,
            vendor="Hikvision",
            model="DS-2CD2042",
            rtsp_url="rtsp://192.168.1.10:554/Streaming/Channels/101",
            snapshot_url=None,
        ),
    ]
    fake_r2 = FakeR2()
    app = create_app(fake_r2, discover_fn=lambda: cams)
    app.config["TESTING"] = True

    body = app.test_client().get("/cameras/discover").get_json()
    assert body["cameras"][0]["needs_manual_url"] is False


# ===== Stage 3 Tuya rows surfaced in JSON (issue #38) =====


def test_get_cameras_discover_includes_tuya_local_rows_with_manual_url_flag() -> None:
    """Issue #38: Stage 3 (Tuya local broadcast) rows must flow through
    ``/cameras/discover`` JSON as-is — ``discovery_method='tuya-local'``,
    ``needs_manual_url=True``, ``rtsp_url=''``, ``vendor`` mentioning Tuya.
    The SPA's truthiness check on ``cam.rtsp_url`` plus the new
    ``discovery_method`` branch is what lets the UI render a "open vendor
    app to enable RTSP" hint instead of a bogus URL."""
    from client_agent.discovery import DiscoveredCamera

    cams = [
        DiscoveredCamera(
            ip="192.168.1.24",
            port=6668,
            vendor="Tuya (Setti+/Tapo/Vstarcam/…)",
            model="2qpika50turuwci4",
            rtsp_url="",
            snapshot_url=None,
            discovery_method="tuya-local",
            needs_manual_url=True,
        ),
    ]
    fake_r2 = FakeR2()
    app = create_app(fake_r2, discover_fn=lambda: cams)
    app.config["TESTING"] = True

    body = app.test_client().get("/cameras/discover").get_json()
    cam = body["cameras"][0]
    assert cam["discovery_method"] == "tuya-local"
    assert cam["needs_manual_url"] is True
    assert cam["rtsp_url"] == ""
    assert "Tuya" in cam["vendor"]
    assert cam["model"] == "2qpika50turuwci4"
    assert cam["port"] == 6668


# ===== /start with camera_ip (issue #22) =====


def test_post_start_with_camera_ip_resolves_creds_and_starts_recording() -> None:
    """Issue #22 happy path: operator clicks a discovered camera. The UI
    posts ``{camera_ip, duration_s}`` (no rtsp_url, no creds in DOM). The
    route looks up the camera in the last-discovery cache, calls the
    injected ``credentials_resolver`` to get user/pass from env, builds
    a vendor-specific RTSP URL, and hands the **full** URL to the
    recorder. The recorder's URL must contain the resolved creds — that
    is what ffmpeg connects with."""
    from client_agent.discovery import DiscoveredCamera

    cams = [
        DiscoveredCamera(
            ip="192.168.50.2",
            port=554,
            vendor="Hikvision",
            model="DS-2CD2042",
            rtsp_url="rtsp://192.168.50.2:554/Streaming/Channels/101",
            snapshot_url=None,
            discovery_method="rtsp-scan",
        ),
    ]
    fake_r2 = FakeR2()
    fake_recorder = FakeRecorder()
    resolver_calls: list[str] = []

    def resolver(ip: str) -> tuple[str, str] | None:
        resolver_calls.append(ip)
        return ("admin", "Secret1!")

    app = create_app(
        fake_r2,
        recorder=fake_recorder,
        discover_fn=lambda: cams,
        credentials_resolver=resolver,
    )
    app.config["TESTING"] = True
    test_client = app.test_client()

    # Discovery first — populates the cache the /start handler reads.
    test_client.get("/cameras/discover")

    resp = test_client.post(
        "/start",
        data={"camera_ip": "192.168.50.2", "duration_s": "300"},
    )

    assert resp.status_code == 302
    assert resp.headers["Location"].endswith("/jobs")
    assert resolver_calls == ["192.168.50.2"]
    assert len(fake_recorder.starts) == 1
    started = fake_recorder.starts[0]
    assert started["duration_s"] == 300
    # The URL the recorder receives MUST carry creds — that's the whole
    # point of resolving them server-side. Hikvision template + URL-encoded
    # password (``!`` is preserved by quote(safe="")).
    assert started["url"] == ("rtsp://admin:Secret1%21@192.168.50.2:554/Streaming/Channels/101")


def test_post_start_with_camera_ip_not_in_cache_returns_400() -> None:
    """No prior /cameras/discover, or the IP wasn't in the last scan →
    we don't have vendor/port to build a URL. Refuse early with a
    readable message so the operator knows to re-scan, instead of
    fabricating a URL that ffmpeg would fail to connect to."""
    fake_r2 = FakeR2()
    fake_recorder = FakeRecorder()
    app = create_app(
        fake_r2,
        recorder=fake_recorder,
        discover_fn=lambda: [],
        credentials_resolver=lambda _ip: ("admin", "Secret"),
    )
    app.config["TESTING"] = True

    resp = app.test_client().post(
        "/start",
        data={"camera_ip": "10.0.0.99", "duration_s": "300"},
    )

    assert resp.status_code == 400
    assert b"10.0.0.99" in resp.data
    assert b"discovery" in resp.data.lower() or b"scan" in resp.data.lower()
    assert fake_recorder.starts == []


def test_post_start_with_camera_ip_but_no_resolved_creds_returns_400() -> None:
    """Issue #22 acceptance: ``→ błąd 400 z czytelnym komunikatem``. When
    the resolver returns ``None`` (no per-IP override and no
    ``RTSP_DEFAULT_USER/PASS`` set), refuse the recording. Building a
    no-creds URL would let ffmpeg attempt anonymous auth which most IP
    cameras reject — better to surface the misconfiguration to the
    operator with a clear message naming the env vars to set."""
    from client_agent.discovery import DiscoveredCamera

    cams = [
        DiscoveredCamera(
            ip="192.168.50.2",
            port=554,
            vendor="Hikvision",
            model="",
            rtsp_url="rtsp://192.168.50.2:554/Streaming/Channels/101",
            snapshot_url=None,
        ),
    ]
    fake_r2 = FakeR2()
    fake_recorder = FakeRecorder()
    app = create_app(
        fake_r2,
        recorder=fake_recorder,
        discover_fn=lambda: cams,
        credentials_resolver=lambda _ip: None,  # nothing in env
    )
    app.config["TESTING"] = True
    test_client = app.test_client()
    test_client.get("/cameras/discover")  # populate cache

    resp = test_client.post(
        "/start",
        data={"camera_ip": "192.168.50.2", "duration_s": "300"},
    )

    assert resp.status_code == 400
    body = resp.data.decode()
    # Message must name the env vars so the operator can fix it without
    # opening the source.
    assert "RTSP_DEFAULT" in body
    assert "RTSP_CAM_" in body
    assert fake_recorder.starts == []


def test_post_start_with_neither_rtsp_url_nor_camera_ip_returns_400() -> None:
    """The body must carry one of the two — anything else is a
    misconfigured client (curl probe, stale UI). Pre-#22 the rejection
    message was ``missing rtsp_url``; updated to mention both fields so
    the operator sees what's allowed."""
    fake_r2 = FakeR2()
    fake_recorder = FakeRecorder()
    app = create_app(
        fake_r2,
        recorder=fake_recorder,
        discover_fn=lambda: [],
        credentials_resolver=lambda _ip: None,
    )
    app.config["TESTING"] = True

    resp = app.test_client().post("/start", data={"duration_s": "300"})

    assert resp.status_code == 400
    body = resp.data.lower()
    assert b"rtsp_url" in body
    assert b"camera_ip" in body
    assert fake_recorder.starts == []


def test_post_start_with_camera_ip_does_not_log_password(caplog) -> None:
    """Issue #22 acceptance: ``Logi Flask nie zawierają hasła w żadnej
    formie``. The /start route resolves creds and builds the full RTSP
    URL — that URL must reach the recorder but must NOT show up in
    werkzeug request logs, app logs, or Flask error logs.

    We instrument with caplog at DEBUG level so even verbose third-party
    output is captured. The assertion grep looks for the literal password
    in any record across any logger; a match means the URL leaked
    somewhere we can't easily contain (e.g. werkzeug echoing the form
    body — currently it doesn't, but a future Flask upgrade could)."""
    import logging

    from client_agent.discovery import DiscoveredCamera

    cams = [
        DiscoveredCamera(
            ip="192.168.50.2",
            port=554,
            vendor="Hikvision",
            model="",
            rtsp_url="rtsp://192.168.50.2:554/Streaming/Channels/101",
            snapshot_url=None,
        ),
    ]
    fake_r2 = FakeR2()
    fake_recorder = FakeRecorder()
    secret_password = "TopSecret-Pa$$word-9876"
    app = create_app(
        fake_r2,
        recorder=fake_recorder,
        discover_fn=lambda: cams,
        credentials_resolver=lambda _ip: ("admin", secret_password),
    )
    app.config["TESTING"] = True
    test_client = app.test_client()
    test_client.get("/cameras/discover")

    with caplog.at_level(logging.DEBUG):
        resp = test_client.post(
            "/start",
            data={"camera_ip": "192.168.50.2", "duration_s": "300"},
        )

    assert resp.status_code == 302
    # Recorder DID receive the URL with the (URL-encoded) password — the
    # whole point of resolving server-side. Sanity check the bottom-up.
    started_url = fake_recorder.starts[0]["url"]
    # quote(safe="") encodes the special chars, so the literal password
    # *substring* "TopSecret-Pa" survives URL-encoding (only $ : @ change).
    # We assert that *no* substring of the password appears verbatim or
    # URL-encoded in any captured log message.
    encoded_password = "TopSecret-Pa%24%24word-9876"
    assert encoded_password in started_url

    # Iterate every captured log record across every logger.
    for record in caplog.records:
        msg = record.getMessage()
        assert secret_password not in msg, f"raw password leaked into log {record.name!r}: {msg!r}"
        assert encoded_password not in msg, (
            f"url-encoded password leaked into log {record.name!r}: {msg!r}"
        )


# ===== Per-camera snapshot endpoint (issue #41) =====


def test_get_camera_snapshot_returns_jpeg_bytes_with_image_content_type() -> None:
    """Issue #41 tracer-bullet: ``GET /cameras/<camera_id>/snapshot`` runs
    the injected snapshot grabber against the camera's RTSP URL and returns
    the resulting JPEG bytes verbatim with ``image/jpeg`` Content-Type.

    The grabber is injected so unit tests never open a real socket; the
    production grabber (OpenCV's ``cv2.VideoCapture`` + ``cv2.imencode``)
    is wired in ``agent.py`` / ``appliance.py``. The resolver is also
    injected so the route's camera_id → RTSP URL lookup is testable in
    isolation."""
    from client_agent.web import CameraSnapshotSource

    fake_jpeg = b"\xff\xd8\xff\xe0FAKE-JPEG-PAYLOAD\xff\xd9"
    grab_calls: list[tuple[str, float]] = []

    def fake_grabber(url: str, timeout_s: float) -> bytes:
        grab_calls.append((url, timeout_s))
        return fake_jpeg

    def resolver(camera_id: str) -> CameraSnapshotSource | None:
        if camera_id == "cam-abc":
            return CameraSnapshotSource(rtsp_url="rtsp://camera.local/stream")
        return None

    fake_r2 = FakeR2()
    fake_recorder = FakeRecorder()
    app = create_app(
        fake_r2,
        recorder=fake_recorder,
        camera_resolver=resolver,
        snapshot_grabber=fake_grabber,
    )
    app.config["TESTING"] = True

    resp = app.test_client().get("/cameras/cam-abc/snapshot")

    assert resp.status_code == 200
    assert resp.mimetype == "image/jpeg"
    assert resp.data == fake_jpeg
    assert grab_calls == [("rtsp://camera.local/stream", 5.0)]


def test_get_camera_snapshot_without_recorder_returns_404() -> None:
    """Issue #41 AC: ``Route registered when recorder is not None
    (consistent with /start / /stop gating)``. We register the route
    unconditionally for simplicity but return 404 when ``recorder=None``
    so the operator sees the same "not configured" shape as the other
    recorder-dependent endpoints."""
    fake_r2 = FakeR2()
    # recorder intentionally omitted — the upload-only surface.
    app = create_app(fake_r2, snapshot_grabber=lambda _u, _t: b"\xff\xd8\xff\xd9")
    app.config["TESTING"] = True

    resp = app.test_client().get("/cameras/cam-abc/snapshot")

    assert resp.status_code == 404


def test_get_camera_snapshot_returns_404_when_camera_id_unknown() -> None:
    """Issue #41 AC: ``Returns 404 for unknown camera_id``. Resolver
    returns ``None`` and last_discovery is empty → the route refuses
    instead of fabricating a URL ffmpeg would fail to open. The body
    names the camera_id so the operator can correlate with their
    discovery scan."""
    from client_agent.web import CameraSnapshotSource

    def resolver(_camera_id: str) -> CameraSnapshotSource | None:
        return None

    fake_r2 = FakeR2()
    fake_recorder = FakeRecorder()
    grabber_calls: list[str] = []
    app = create_app(
        fake_r2,
        recorder=fake_recorder,
        camera_resolver=resolver,
        snapshot_grabber=lambda url, _t: grabber_calls.append(url) or b"\xff\xd8",
    )
    app.config["TESTING"] = True

    resp = app.test_client().get("/cameras/cam-never-seen/snapshot")

    assert resp.status_code == 404
    assert b"cam-never-seen" in resp.data
    # The grabber must NOT be touched for an unknown camera — otherwise
    # an attacker could probe arbitrary URLs by guessing camera_ids.
    assert grabber_calls == []


def test_get_camera_snapshot_returns_503_with_camera_id_when_grabber_fails(caplog) -> None:
    """Issue #41 AC: ``Returns 503 if the camera RTSP connection fails /
    times out (with reason in body)``. 503 (Service Unavailable) is the
    right shape: the route itself is fine — it's the camera that's
    offline. 500 would imply a server bug and confuse the operator.

    The body names the ``camera_id`` so the operator can correlate
    failures without scanning logs; the *reason* (e.g. ``cv2.VideoCapture
    failed to open``) is logged server-side at WARNING — keeping it out
    of the body protects against credentials embedded in the RTSP URL
    leaking via ``str(exc)`` (see
    ``test_get_camera_snapshot_503_body_does_not_leak_creds_from_rtsp_url``)."""
    import logging

    from client_agent.web import CameraSnapshotSource

    def resolver(camera_id: str) -> CameraSnapshotSource | None:
        return CameraSnapshotSource(rtsp_url="rtsp://offline.local/stream")

    def exploding_grabber(_url: str, _t: float) -> bytes:
        raise RuntimeError("cv2.VideoCapture failed to open: timed out after 5s")

    fake_r2 = FakeR2()
    fake_recorder = FakeRecorder()
    app = create_app(
        fake_r2,
        recorder=fake_recorder,
        camera_resolver=resolver,
        snapshot_grabber=exploding_grabber,
    )
    app.config["TESTING"] = True
    # Without this Flask re-raises the RuntimeError instead of letting
    # our try/except convert it to a 503.
    app.config["PROPAGATE_EXCEPTIONS"] = False

    with caplog.at_level(logging.WARNING, logger="client_agent.web"):
        resp = app.test_client().get("/cameras/cam-abc/snapshot")

    assert resp.status_code == 503
    body = resp.get_data(as_text=True)
    assert "cam-abc" in body
    # Reason is logged, not echoed — operator finds it in journald.
    assert any("cv2.VideoCapture failed" in r.getMessage() for r in caplog.records), (
        "503 reason should be logged server-side"
    )


def test_get_camera_snapshot_resolves_from_last_discovery_by_ip() -> None:
    """Issue #41 implementation note: ``Reuse last_discovery cache keyed
    by camera_id``. In Docker standalone mode, the operator addresses
    cameras by IP (there is no platform UUID), so the route must accept
    the camera's IP as the ``camera_id`` URL fragment and look it up in
    the in-memory cache populated by ``GET /cameras/discover``.

    No ``camera_resolver`` is injected here — proves last_discovery
    alone is sufficient for the Docker flow."""
    from client_agent.discovery import DiscoveredCamera

    cams = [
        DiscoveredCamera(
            ip="192.168.50.2",
            port=554,
            vendor="Hikvision",
            model="DS-2CD2042",
            rtsp_url="rtsp://192.168.50.2:554/Streaming/Channels/101",
            snapshot_url=None,
        ),
    ]
    fake_r2 = FakeR2()
    fake_recorder = FakeRecorder()
    grab_calls: list[str] = []

    def fake_grabber(url: str, _t: float) -> bytes:
        grab_calls.append(url)
        return b"\xff\xd8FAKE\xff\xd9"

    app = create_app(
        fake_r2,
        recorder=fake_recorder,
        discover_fn=lambda: cams,
        snapshot_grabber=fake_grabber,
    )
    app.config["TESTING"] = True
    test_client = app.test_client()
    # Populate last_discovery.
    test_client.get("/cameras/discover")

    resp = test_client.get("/cameras/192.168.50.2/snapshot")

    assert resp.status_code == 200
    assert resp.mimetype == "image/jpeg"
    assert resp.data == b"\xff\xd8FAKE\xff\xd9"
    # Grabber received the RTSP URL from last_discovery — no resolver
    # involved, no URL fabrication.
    assert grab_calls == ["rtsp://192.168.50.2:554/Streaming/Channels/101"]


def test_get_camera_snapshot_serves_cached_jpeg_within_ttl() -> None:
    """Issue #41 AC: ``Returns the same encoded image for two requests
    within 30s (TTL cache)``. Two GETs hit the route while the injected
    clock advances by less than the TTL → the grabber is called exactly
    once and both responses carry identical bytes."""
    from client_agent.web import CameraSnapshotSource

    def resolver(_camera_id: str) -> CameraSnapshotSource | None:
        return CameraSnapshotSource(rtsp_url="rtsp://camera.local/stream")

    grab_count = [0]

    def fake_grabber(_url: str, _t: float) -> bytes:
        grab_count[0] += 1
        # Returning a distinct payload per call lets the test prove that
        # the cached value (not a fresh one) is what came back on call 2.
        return f"jpeg-{grab_count[0]}".encode()

    now = [1000.0]
    fake_r2 = FakeR2()
    fake_recorder = FakeRecorder()
    app = create_app(
        fake_r2,
        recorder=fake_recorder,
        camera_resolver=resolver,
        snapshot_grabber=fake_grabber,
        clock=lambda: now[0],
    )
    app.config["TESTING"] = True
    test_client = app.test_client()

    first = test_client.get("/cameras/cam-abc/snapshot")
    # Advance by 29 s — still within the 30 s TTL window.
    now[0] += 29.0
    second = test_client.get("/cameras/cam-abc/snapshot")

    assert first.status_code == 200 and second.status_code == 200
    assert first.data == b"jpeg-1"
    assert second.data == b"jpeg-1"  # cached, not regenerated
    assert grab_count[0] == 1


def test_get_camera_snapshot_refreshes_after_ttl_expiry() -> None:
    """Issue #41 AC: ``Returns a fresh image after 30s``. Clock advances
    past the TTL → grabber called again, second response carries the
    fresh payload."""
    from client_agent.web import CameraSnapshotSource

    def resolver(_camera_id: str) -> CameraSnapshotSource | None:
        return CameraSnapshotSource(rtsp_url="rtsp://camera.local/stream")

    grab_count = [0]

    def fake_grabber(_url: str, _t: float) -> bytes:
        grab_count[0] += 1
        return f"jpeg-{grab_count[0]}".encode()

    now = [1000.0]
    fake_r2 = FakeR2()
    fake_recorder = FakeRecorder()
    app = create_app(
        fake_r2,
        recorder=fake_recorder,
        camera_resolver=resolver,
        snapshot_grabber=fake_grabber,
        clock=lambda: now[0],
    )
    app.config["TESTING"] = True
    test_client = app.test_client()

    first = test_client.get("/cameras/cam-abc/snapshot")
    # Past the 30 s TTL — cache entry is stale, grabber must re-run.
    now[0] += 31.0
    second = test_client.get("/cameras/cam-abc/snapshot")

    assert first.data == b"jpeg-1"
    assert second.data == b"jpeg-2"
    assert grab_count[0] == 2


def test_get_camera_snapshot_prefers_vendor_snapshot_url_over_rtsp() -> None:
    """Issue #41 implementation note: ``If ONVIF discovery already
    surfaced a snapshot_url (vendor's native HTTP snapshot endpoint),
    prefer that — fall back to opencv frame grab only when not available.``

    Cheaper for the camera (one HTTP GET vs. an RTSP session
    handshake) and avoids spinning up the heavy ffmpeg pipeline for a
    thumbnail. The grabber receives the HTTP URL, not the RTSP one."""
    from client_agent.discovery import DiscoveredCamera

    cams = [
        DiscoveredCamera(
            ip="192.168.50.10",
            port=80,
            vendor="Hikvision",
            model="DS-2CD2042",
            rtsp_url="rtsp://192.168.50.10:554/Streaming/Channels/101",
            snapshot_url="http://192.168.50.10/Streaming/Channels/101/picture",
        ),
    ]
    fake_r2 = FakeR2()
    fake_recorder = FakeRecorder()
    grab_calls: list[str] = []

    def fake_grabber(url: str, _t: float) -> bytes:
        grab_calls.append(url)
        return b"\xff\xd8VENDOR-HTTP\xff\xd9"

    app = create_app(
        fake_r2,
        recorder=fake_recorder,
        discover_fn=lambda: cams,
        snapshot_grabber=fake_grabber,
    )
    app.config["TESTING"] = True
    test_client = app.test_client()
    test_client.get("/cameras/discover")  # populate last_discovery

    resp = test_client.get("/cameras/192.168.50.10/snapshot")

    assert resp.status_code == 200
    assert resp.data == b"\xff\xd8VENDOR-HTTP\xff\xd9"
    # Critically: the HTTP snapshot URL is what reached the grabber, not
    # the rtsp_url. A regression here means the cheap path was skipped.
    assert grab_calls == ["http://192.168.50.10/Streaming/Channels/101/picture"]


def test_get_camera_snapshot_503_body_does_not_leak_creds_from_rtsp_url() -> None:
    """Security: a Stage-2 RTSP-scan ``DiscoveredCamera`` carries the
    operator's credentials inline by design (``rtsp://user:pass@host/…``)
    and ``last_discovery`` stores the *raw* row, not the cred-stripped
    version that ships out via ``/cameras/discover``. If the snapshot
    grabber raises with the URL in its message, the 503 body would echo
    those creds back to whoever called the endpoint. The route must log
    the URL server-side and return a generic body that names only the
    camera_id."""
    from client_agent.discovery import DiscoveredCamera
    from client_agent.web import CameraSnapshotSource  # noqa: F401

    secret = "Hunter2!"
    cams = [
        DiscoveredCamera(
            ip="192.168.50.2",
            port=554,
            vendor="Hikvision",
            model="",
            rtsp_url=f"rtsp://admin:{secret}@192.168.50.2:554/Streaming/Channels/101",
            snapshot_url=None,
            discovery_method="rtsp-scan",
        ),
    ]
    fake_r2 = FakeR2()
    fake_recorder = FakeRecorder()

    def exploding_grabber(url: str, _t: float) -> bytes:
        # Mimic the production helpers (snapshot.py) which interpolate
        # the URL into the exception message — the very leak path the
        # route must defang.
        raise RuntimeError(f"cv2.VideoCapture failed to open {url!r}")

    app = create_app(
        fake_r2,
        recorder=fake_recorder,
        discover_fn=lambda: cams,
        snapshot_grabber=exploding_grabber,
    )
    app.config["TESTING"] = True
    app.config["PROPAGATE_EXCEPTIONS"] = False
    test_client = app.test_client()
    test_client.get("/cameras/discover")  # populate last_discovery

    resp = test_client.get("/cameras/192.168.50.2/snapshot")

    assert resp.status_code == 503
    body = resp.get_data(as_text=True)
    # The secret password (and the userinfo prefix that pairs it with the
    # username) must NOT leak into the response body.
    assert secret not in body
    assert "admin:" not in body
    # And the body must still carry *something useful* — the camera_id —
    # so the operator sees which camera failed without scanning logs.
    assert "192.168.50.2" in body


def test_post_start_with_camera_ip_and_no_resolver_configured_returns_400() -> None:
    """``credentials_resolver=None`` is a misconfigured deploy (operator
    forgot to wire it in agent.py). Treat it the same as "no creds":
    refuse with a 400 instead of starting an auth-less recording. The
    whole point of #22 is that creds live in env, not URL."""
    from client_agent.discovery import DiscoveredCamera

    cams = [
        DiscoveredCamera(
            ip="192.168.50.2",
            port=554,
            vendor="Hikvision",
            model="",
            rtsp_url="rtsp://192.168.50.2:554/Streaming/Channels/101",
            snapshot_url=None,
        ),
    ]
    fake_r2 = FakeR2()
    fake_recorder = FakeRecorder()
    app = create_app(
        fake_r2,
        recorder=fake_recorder,
        discover_fn=lambda: cams,
        # credentials_resolver intentionally omitted
    )
    app.config["TESTING"] = True
    test_client = app.test_client()
    test_client.get("/cameras/discover")

    resp = test_client.post(
        "/start",
        data={"camera_ip": "192.168.50.2", "duration_s": "300"},
    )

    assert resp.status_code == 400
    assert fake_recorder.starts == []
