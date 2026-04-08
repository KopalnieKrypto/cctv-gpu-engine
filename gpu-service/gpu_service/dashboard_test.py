"""Tests for the gpu-service investor dashboard (issue #6).

The dashboard surfaces R2 job history as a small read-only HTTP page served
from the same container as the worker. These tests pin behaviour through
the public surface only:

* :func:`gpu_service.dashboard.list_jobs` — pure aggregation over an R2-like
  client (duck-typed, same pattern as :class:`gpu_service.worker.R2ClientLike`)
* :func:`gpu_service.dashboard.render_dashboard_html` — Jinja2 render of a
  list of :class:`JobSummary` records
* :func:`gpu_service.dashboard.make_handler` — stdlib ``http.server`` handler
  factory exercised in-process via the WSGI-like ``do_GET`` path

The R2 client is faked with a tiny in-memory stand-in (no boto3 mocking
here — that lives in ``r2_client_test.py`` at the network boundary).
"""

from __future__ import annotations

from typing import Any

from gpu_service.dashboard import JobSummary, list_jobs, render_dashboard_html


class FakeR2:
    """In-memory stand-in for the dashboard's R2 surface.

    Mirrors the duck-typed contract :func:`list_jobs` depends on:
    ``list_all_job_statuses() -> list[tuple[str, dict]]``.
    """

    def __init__(self, jobs: list[tuple[str, dict[str, Any]]] | None = None) -> None:
        self._jobs = list(jobs or [])

    def list_all_job_statuses(self) -> list[tuple[str, dict[str, Any]]]:
        return list(self._jobs)


def test_list_jobs_returns_empty_list_when_r2_has_no_jobs() -> None:
    """Empty bucket → empty dashboard, no exceptions."""
    assert list_jobs(FakeR2([])) == []


def test_list_jobs_maps_single_status_to_jobsummary() -> None:
    """A single status.json round-trips id, status, and updated_at."""
    fake = FakeR2(
        [
            (
                "job-a",
                {
                    "job_id": "job-a",
                    "status": "completed",
                    "updated_at": "2026-04-08T10:00:00Z",
                },
            )
        ]
    )

    jobs = list_jobs(fake)

    assert jobs == [
        JobSummary(
            job_id="job-a",
            status="completed",
            updated_at="2026-04-08T10:00:00Z",
            duration_s=None,
            error=None,
        )
    ]


def test_list_jobs_sorted_by_updated_at_descending() -> None:
    """Newest jobs first — investor wants the latest at the top of the page."""
    fake = FakeR2(
        [
            ("oldest", {"status": "completed", "updated_at": "2026-04-08T08:00:00Z"}),
            ("newest", {"status": "completed", "updated_at": "2026-04-08T12:00:00Z"}),
            ("middle", {"status": "completed", "updated_at": "2026-04-08T10:00:00Z"}),
        ]
    )

    jobs = list_jobs(fake)

    assert [j.job_id for j in jobs] == ["newest", "middle", "oldest"]


def test_list_jobs_computes_duration_from_started_and_completed_at() -> None:
    """duration_s = completed_at - started_at, parsed from ISO-8601 UTC.

    None when either timestamp is missing — e.g. job still pending or the
    worker crashed before writing ``completed_at``.
    """
    fake = FakeR2(
        [
            (
                "done",
                {
                    "status": "completed",
                    "updated_at": "2026-04-08T10:05:00Z",
                    "started_at": "2026-04-08T10:00:00Z",
                    "completed_at": "2026-04-08T10:05:30Z",
                },
            ),
            (
                "pending",
                {
                    "status": "pending",
                    "updated_at": "2026-04-08T09:00:00Z",
                },
            ),
        ]
    )

    jobs = {j.job_id: j for j in list_jobs(fake)}

    assert jobs["done"].duration_s == 330.0
    assert jobs["pending"].duration_s is None


def test_list_jobs_propagates_error_message_for_failed_jobs() -> None:
    """`status: failed` jobs surface the worker's error string verbatim.

    The investor scanning the dashboard needs to see *why* a job failed
    without clicking through to R2 logs (user story 18).
    """
    fake = FakeR2(
        [
            (
                "boom",
                {
                    "status": "failed",
                    "updated_at": "2026-04-08T10:00:00Z",
                    "error": "RuntimeError: CUDA out of memory",
                },
            ),
        ]
    )

    [job] = list_jobs(fake)
    assert job.status == "failed"
    assert job.error == "RuntimeError: CUDA out of memory"


def test_render_dashboard_html_includes_all_jobsummary_fields() -> None:
    """Each JobSummary appears in the rendered table with every field visible.

    We assert on substrings rather than DOM structure — the page is small
    and refactoring the markup shouldn't break the test as long as the
    user-visible data is still there.
    """
    jobs = [
        JobSummary(
            job_id="job-7",
            status="failed",
            updated_at="2026-04-08T10:00:00Z",
            duration_s=42.5,
            error="RuntimeError: boom",
        ),
    ]

    html = render_dashboard_html(jobs)

    assert "job-7" in html
    assert "failed" in html
    assert "2026-04-08T10:00:00Z" in html
    assert "42" in html  # duration rendered (formatting checked elsewhere)
    assert "RuntimeError: boom" in html


def test_render_dashboard_html_shows_placeholder_when_no_jobs() -> None:
    """First-launch state — bucket is empty, page should say so explicitly."""
    html = render_dashboard_html([])
    assert "no jobs yet" in html
    # No table rendered when empty.
    assert "<tbody>" not in html


def test_render_dashboard_html_has_meta_refresh_tag() -> None:
    """Dashboard auto-refreshes — investor doesn't have to hit reload.

    Acceptance criterion from issue #6: "Dashboard refreshes (auto or
    manual)". Auto via meta refresh keeps the page dependency-free (no JS).
    """
    html = render_dashboard_html([])
    assert 'http-equiv="refresh"' in html.lower()


# ----- HTTP handler -----
#
# We exercise the BaseHTTPRequestHandler subclass without binding a real
# socket by driving its do_GET method through a tiny in-process harness:
# fake `rfile`, capturing `wfile`, and a stub `path`. This keeps the test
# fast and free of port-allocation flakes while still exercising the same
# code path the live server would use.


class _FakeWFile:
    def __init__(self) -> None:
        self.buf = bytearray()

    def write(self, data: bytes) -> int:
        self.buf.extend(data)
        return len(data)

    def flush(self) -> None:
        pass


def _drive(handler_cls: type, path: str) -> tuple[int, dict[str, str], bytes]:
    """Instantiate handler_cls without a socket and capture the response.

    Returns ``(status_code, headers, body)``. We bypass __init__ (which
    expects a real socket) and manually wire just the attributes do_GET
    needs: ``path``, ``rfile``, ``wfile``, and a no-op log_message.
    """
    from io import BytesIO

    handler = handler_cls.__new__(handler_cls)
    handler.path = path
    handler.rfile = BytesIO(b"")
    wfile = _FakeWFile()
    handler.wfile = wfile  # type: ignore[assignment]
    handler.request_version = "HTTP/1.0"
    handler.command = "GET"
    handler.requestline = f"GET {path} HTTP/1.0"
    handler.log_message = lambda *a, **kw: None  # type: ignore[assignment]
    handler.do_GET()

    raw = bytes(wfile.buf)
    head, _, body = raw.partition(b"\r\n\r\n")
    lines = head.split(b"\r\n")
    status_line = lines[0].decode()
    status_code = int(status_line.split(" ")[1])
    headers = {}
    for line in lines[1:]:
        if b":" in line:
            k, v = line.split(b":", 1)
            headers[k.decode().strip().lower()] = v.decode().strip()
    return status_code, headers, body


def test_handler_get_dashboard_returns_200_with_rendered_html() -> None:
    """`GET /dashboard` returns 200 with the same HTML render_dashboard_html
    would produce for the current R2 contents."""
    from gpu_service.dashboard import make_handler

    fake = FakeR2(
        [
            ("job-x", {"status": "completed", "updated_at": "2026-04-08T10:00:00Z"}),
        ]
    )
    handler_cls = make_handler(fake)

    status, headers, body = _drive(handler_cls, "/dashboard")

    assert status == 200
    assert headers["content-type"].startswith("text/html")
    assert b"job-x" in body
    assert b"completed" in body


def test_handler_get_root_redirects_to_dashboard() -> None:
    """`GET /` is a convenience redirect so the investor can just hit the host."""
    from gpu_service.dashboard import make_handler

    handler_cls = make_handler(FakeR2([]))
    status, headers, _ = _drive(handler_cls, "/")

    assert status == 302
    assert headers["location"] == "/dashboard"


def test_handler_get_unknown_path_returns_404() -> None:
    """Anything other than `/` or `/dashboard` is a 404 — no path traversal,
    no static file serving, no surprises."""
    from gpu_service.dashboard import make_handler

    handler_cls = make_handler(FakeR2([]))
    status, _, body = _drive(handler_cls, "/etc/passwd")

    assert status == 404
    assert b"not found" in body
