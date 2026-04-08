"""Investor dashboard for the gpu-service worker (issue #6).

A read-only HTTP page served from the same container as the worker that
shows job history pulled from R2: id, status, timestamp, duration, error.

Design notes:

* Pure aggregation (:func:`list_jobs`) is decoupled from rendering and HTTP
  so the bulk of behaviour can be tested without spinning up sockets.
* The R2 surface this module depends on is intentionally narrower than the
  worker's: only ``list_all_job_statuses() -> list[tuple[str, dict]]``.
* No Flask — stdlib ``http.server`` keeps the gpu-service image lean. The
  HTML is rendered with Jinja2 (already a project dep for the report).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol


def _parse_iso(value: Any) -> datetime | None:
    """Parse an ISO-8601 UTC timestamp written by the worker, or return None.

    The worker writes ``YYYY-MM-DDTHH:MM:SSZ``; ``fromisoformat`` accepts
    that on Python 3.11+ as long as we replace the trailing ``Z`` with
    ``+00:00`` (the explicit UTC offset). Anything malformed → None so the
    dashboard never crashes on a half-written status.json.
    """
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _compute_duration_s(status: dict[str, Any]) -> float | None:
    started = _parse_iso(status.get("started_at"))
    completed = _parse_iso(status.get("completed_at"))
    if started is None or completed is None:
        return None
    return (completed - started).total_seconds()


class DashboardR2Like(Protocol):
    """Structural type for the R2 surface the dashboard depends on."""

    def list_all_job_statuses(self) -> list[tuple[str, dict[str, Any]]]: ...


@dataclass(frozen=True)
class JobSummary:
    """Display-ready row for the dashboard table.

    Built from a raw R2 ``status.json`` payload by :func:`list_jobs`. Kept as
    a frozen dataclass so the rendering layer can rely on a stable shape and
    so equality checks make tests trivial.
    """

    job_id: str
    status: str
    updated_at: str
    duration_s: float | None
    error: str | None


def _summarize(job_id: str, status: dict[str, Any]) -> JobSummary:
    return JobSummary(
        job_id=job_id,
        status=str(status.get("status", "unknown")),
        updated_at=str(status.get("updated_at", "")),
        duration_s=_compute_duration_s(status),
        error=status.get("error") or None,
    )


def list_jobs(client: DashboardR2Like) -> list[JobSummary]:
    """Aggregate every R2 status.json into a flat list of job summaries.

    Sorted by ``updated_at`` descending so the newest jobs render at the top.
    ISO-8601 UTC timestamps (the format the worker writes) sort lexically the
    same way they sort chronologically, so a plain string sort is enough.
    """
    summaries = [_summarize(job_id, status) for job_id, status in client.list_all_job_statuses()]
    summaries.sort(key=lambda s: s.updated_at, reverse=True)
    return summaries


_DASHBOARD_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="10">
<title>cctv-gpu-engine — jobs</title>
<style>
 body { font: 14px system-ui, sans-serif; margin: 2rem; }
 table { border-collapse: collapse; width: 100%; }
 th, td { text-align: left; padding: .4rem .6rem; border-bottom: 1px solid #ddd; }
 th { background: #f4f4f4; }
 .status-completed { color: #060; }
 .status-failed    { color: #a00; }
 .status-processing { color: #036; }
 .status-pending   { color: #666; }
 .error { color: #a00; font-family: monospace; font-size: 12px; }
</style>
</head>
<body>
<h1>GPU service — job history</h1>
{% if jobs %}
<table>
<thead><tr>
 <th>Job ID</th><th>Status</th><th>Updated</th><th>Duration</th><th>Error</th>
</tr></thead>
<tbody>
{% for j in jobs %}
<tr>
 <td>{{ j.job_id }}</td>
 <td class="status-{{ j.status }}">{{ j.status }}</td>
 <td>{{ j.updated_at }}</td>
 <td>{{ j.duration_s | round(1) if j.duration_s is not none else "—" }}</td>
 <td class="error">{{ j.error or "" }}</td>
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


def make_handler(client: DashboardR2Like) -> type:
    """Factory returning a ``BaseHTTPRequestHandler`` bound to ``client``.

    The handler closes over ``client`` so it can be passed to
    ``http.server.HTTPServer`` without globals. Routes:

    * ``GET /dashboard`` → 200 + rendered HTML
    * ``GET /``         → 302 redirect to ``/dashboard``
    * anything else     → 404
    """
    from http.server import BaseHTTPRequestHandler

    class DashboardHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 — http.server convention
            if self.path == "/":
                self.send_response(302)
                self.send_header("Location", "/dashboard")
                self.end_headers()
                return
            if self.path == "/dashboard":
                html = render_dashboard_html(list_jobs(client)).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html)))
                self.end_headers()
                self.wfile.write(html)
                return
            self.send_response(404)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"not found")

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            # Quiet by default — the worker uses the logging module instead
            # of stderr noise. Tests rely on this no-op too.
            return

    return DashboardHandler


def serve(client: DashboardR2Like, host: str = "0.0.0.0", port: int = 5000) -> None:
    """Run the dashboard HTTP server forever (called from a daemon thread).

    Bound to ``0.0.0.0`` so it's reachable from outside the container; the
    operator constrains exposure via ``docker compose`` port mapping.
    """
    from http.server import HTTPServer

    server = HTTPServer((host, port), make_handler(client))
    server.serve_forever()


def render_dashboard_html(jobs: list[JobSummary]) -> str:
    """Render the dashboard HTML page from a list of summaries.

    Uses Jinja2 (already a project dep for the report generator) so we get
    auto-escaping for the error column for free — error strings come from
    pipeline exceptions and could in principle contain ``<script>`` if a
    filename or path were ever embedded.
    """
    from jinja2 import Environment, select_autoescape

    env = Environment(autoescape=select_autoescape(["html", "xml"]))
    template = env.from_string(_DASHBOARD_TEMPLATE)
    return template.render(jobs=jobs)
