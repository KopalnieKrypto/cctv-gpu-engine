"""Tests for the client-agent Flask web UI (issue #7).

The web app exposes the operator-facing surface of the client-agent: an MP4
upload form, a job list with status badges, and a viewer/download for the
generated report. These tests pin behaviour through the public Flask
``test_client`` only — no real sockets, no real R2.

The R2 client is faked with an in-memory stand-in (same pattern as
``gpu_service/dashboard_test.py``). Mocks live only at system boundaries;
boto3 itself is exercised in ``gpu_service/r2_client_test.py``.
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
