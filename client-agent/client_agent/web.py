"""Flask web UI for the client-agent (issue #7).

Operator-facing surface that lives on the client's LAN:

* ``GET  /``                       — MP4 upload form
* ``POST /upload``                 — multipart MP4 → R2 + status.json(pending)
* ``GET  /jobs``                   — job list with status badges (auto-refresh)
* ``GET  /jobs/<id>/report``       — proxy R2 report HTML inline
* ``GET  /jobs/<id>/report/download`` — same body as attachment

The R2 client is duck-typed via :class:`ClientR2Like` so the app stays
testable against an in-memory fake — boto3 is only mocked at the network
boundary in ``gpu_service/r2_client_test.py``. The job_id factory is
injectable for the same reason: deterministic in tests, ``uuid4`` in prod.

This module deliberately mirrors the design of ``gpu_service/dashboard.py``:
narrow Protocol, pure helpers, the HTTP layer is the only thing that touches
Flask. Anything reusable between the two services should eventually move into
a shared module — out of scope for #7.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any, Protocol

from flask import Flask, redirect, request, url_for


class ClientR2Like(Protocol):
    """Structural type for the R2 surface the client-agent web UI depends on.

    Intentionally narrower than the gpu-service worker's R2 surface — the UI
    only needs upload + status + report read paths, never download_chunks or
    list_pending_job_ids.
    """

    def upload_input_chunk(self, job_id: str, fileobj: Any) -> str: ...
    def put_status(self, job_id: str, status: dict[str, Any]) -> None: ...
    def list_all_job_statuses(self) -> list[tuple[str, dict[str, Any]]]: ...
    def get_status(self, job_id: str) -> dict[str, Any] | None: ...
    def get_report(self, job_id: str) -> bytes: ...


def _default_job_id() -> str:
    """Generate a fresh job_id. UUID4 hex truncated to 12 chars for legibility
    in URLs and R2 keys; collision probability is negligible at MVP scale."""
    return f"job-{uuid.uuid4().hex[:12]}"


_JOBS_PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="10">
<title>cctv client-agent — jobs</title>
<style>
 body { font: 14px system-ui, sans-serif; margin: 2rem; }
 table { border-collapse: collapse; width: 100%; }
 th, td { text-align: left; padding: .4rem .6rem; border-bottom: 1px solid #ddd; }
 th { background: #f4f4f4; }
 .status-completed  { color: #060; font-weight: 600; }
 .status-failed     { color: #a00; font-weight: 600; }
 .status-processing { color: #036; font-weight: 600; }
 .status-pending    { color: #666; font-weight: 600; }
</style>
</head>
<body>
<h1>Submitted jobs</h1>
<p><a href="/">← Upload another MP4</a></p>
{% if jobs %}
<table>
<thead><tr>
 <th>Job ID</th><th>Status</th><th>Updated</th><th>Report</th>
</tr></thead>
<tbody>
{% for j in jobs %}
<tr>
 <td>{{ j.job_id }}</td>
 <td class="status-{{ j.status }}">{{ j.status }}</td>
 <td>{{ j.updated_at }}</td>
 <td>{% if j.status == 'completed' %}
   <a href="/jobs/{{ j.job_id }}/report">view</a>
   &nbsp;|&nbsp;
   <a href="/jobs/{{ j.job_id }}/report/download">download</a>
 {% endif %}</td>
</tr>
{% endfor %}
</tbody>
</table>
{% else %}
<p>no jobs yet</p>
{% endif %}
</body>
</html>
"""


_UPLOAD_FORM_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>cctv client-agent — upload</title>
<style>
 body { font: 14px system-ui, sans-serif; margin: 2rem; max-width: 40rem; }
 form { display: flex; flex-direction: column; gap: 1rem; }
 button { padding: .6rem 1rem; font-size: 1rem; }
</style>
</head>
<body>
<h1>Upload surveillance MP4</h1>
<form action="/upload" method="post" enctype="multipart/form-data">
  <input type="file" name="mp4" accept="video/mp4" required>
  <button type="submit">Upload &amp; queue for analysis</button>
</form>
<p><a href="/jobs">View job list →</a></p>
</body>
</html>
"""


def create_app(client: ClientR2Like, *, job_id_factory=_default_job_id) -> Flask:
    """Build a Flask app bound to the given R2 client and job_id factory.

    The factory is injectable so tests can pin job_ids; production wires the
    default UUID4-based generator.
    """
    app = Flask(__name__)

    @app.get("/")
    def upload_form() -> tuple[str, int, dict[str, str]]:
        return _UPLOAD_FORM_HTML, 200, {"Content-Type": "text/html; charset=utf-8"}

    @app.post("/upload")
    def upload():
        # ``request.files["mp4"]`` is a Werkzeug FileStorage — its ``stream``
        # attribute is a SpooledTemporaryFile, so the multipart body never
        # gets fully loaded into RAM. We pass the FileStorage straight to R2;
        # boto3's upload_fileobj will call .read() on it. This is what makes
        # the 2 GB MP4 acceptance criterion possible.
        # Edge validation — refuse early so a noisy automation client never
        # creates a phantom job that the gpu-service worker has to fail.
        upload_file = request.files.get("mp4")
        if upload_file is None or not upload_file.filename:
            return ("missing 'mp4' file in upload", 400)
        if not upload_file.filename.lower().endswith(".mp4"):
            return ("only .mp4 files are accepted", 400)
        job_id = job_id_factory()
        # Surface R2 errors as a readable 500 instead of a Flask traceback —
        # the operator on a customer LAN needs to see *which* boto3 error
        # so they can correlate with their R2 dashboard / .env file.
        # We catch BaseException's saner subset (Exception) because anything
        # boto3 throws — ClientError, EndpointConnectionError, NoCredentials —
        # is an Exception subclass.
        try:
            client.upload_input_chunk(job_id, upload_file.stream)
            # status.json is the gpu-service worker's only job-discovery hook
            # (SPEC §6.2). The worker's list_pending_job_ids walk only sees
            # jobs whose status.json says ``pending``. Order matters: we only
            # write status.json *after* a successful chunk upload, so the
            # worker never picks up a job whose video doesn't exist yet.
            now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
            client.put_status(
                job_id,
                {
                    "job_id": job_id,
                    "status": "pending",
                    "updated_at": now,
                    "submitted_at": now,
                },
            )
        except Exception as exc:  # noqa: BLE001 — see comment above
            return (f"upload to R2 failed: {exc}", 500)
        return redirect(url_for("jobs"))

    @app.get("/jobs")
    def jobs() -> tuple[str, int, dict[str, str]]:
        summaries = _summarize_jobs(client.list_all_job_statuses())
        html = _render_jobs_html(summaries)
        return html, 200, {"Content-Type": "text/html; charset=utf-8"}

    def _fetch_report_or_404(job_id: str) -> bytes | None:
        """Fetch a report from R2, returning None on any read error.

        Both ``KeyError`` (in-memory FakeR2) and boto3's ``NoSuchKey`` /
        ``ClientError`` (real R2) collapse to the same outcome from the
        operator's point of view: the report isn't there yet. We translate
        them all into a single 404 in the routes below so the UI never
        leaks a 500 traceback for a not-yet-finished job.
        """
        try:
            return client.get_report(job_id)
        except Exception:  # noqa: BLE001 — boto3 errors + KeyError + None case
            return None

    @app.get("/jobs/<job_id>/report")
    def view_report(job_id: str):
        report = _fetch_report_or_404(job_id)
        if report is None:
            return ("report not found", 404)
        return (report, 200, {"Content-Type": "text/html; charset=utf-8"})

    @app.get("/jobs/<job_id>/report/download")
    def download_report(job_id: str):
        report = _fetch_report_or_404(job_id)
        if report is None:
            return ("report not found", 404)
        return (
            report,
            200,
            {
                "Content-Type": "text/html; charset=utf-8",
                # ``attachment`` forces the browser to save instead of render;
                # the filename mirrors the gpu-service R2 key convention so
                # the operator can match files back to jobs without renaming.
                "Content-Disposition": f'attachment; filename="report-{job_id}.html"',
            },
        )

    return app


def _summarize_jobs(
    raw: list[tuple[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Project raw R2 status payloads into the minimal shape the template
    needs. Sorted by ``updated_at`` descending so the newest jobs render at
    the top — ISO-8601 UTC strings sort lexically the same as chronologically
    so a plain string sort is enough."""
    rows = [
        {
            "job_id": job_id,
            "status": str(status.get("status", "unknown")),
            "updated_at": str(status.get("updated_at", "")),
        }
        for job_id, status in raw
    ]
    rows.sort(key=lambda r: r["updated_at"], reverse=True)
    return rows


def _render_jobs_html(jobs: list[dict[str, Any]]) -> str:
    """Render the job list page. Jinja2 autoescape is on so any user-derived
    field (job_id from URL, error strings) is HTML-escaped automatically."""
    from jinja2 import Environment, select_autoescape

    env = Environment(autoescape=select_autoescape(["html", "xml"]))
    template = env.from_string(_JOBS_PAGE_TEMPLATE)
    return template.render(jobs=jobs)
