"""Flask web UI for the client-agent (issue #7).

Operator-facing surface that lives on the client's LAN:

* ``GET  /``                       — MP4 upload form
* ``POST /upload``                 — multipart MP4 → R2 + status.json(pending)
* ``GET  /jobs``                   — job list with status badges (auto-refresh)
* ``GET  /jobs/<id>/report``       — proxy R2 report HTML inline
* ``GET  /jobs/<id>/report/download`` — same body as attachment

The R2 client is duck-typed via :class:`ClientR2Like` so the app stays
testable against an in-memory fake — boto3 is only mocked at the network
boundary in ``client_agent/r2_client_test.py``. The job_id factory is
injectable for the same reason: deterministic in tests, ``uuid4`` in prod.

This module deliberately mirrors the design of the gpu-side dashboard
module: narrow Protocol, pure helpers, the HTTP layer is the only thing
that touches Flask.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from flask import Flask, jsonify, redirect, request, url_for

from client_agent.discovery import DiscoveredCamera, strip_credentials_from_url

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CameraSnapshotSource:
    """Where to fetch a thumbnail JPEG for a given camera_id (issue #41).

    ``rtsp_url`` is the always-present fallback — the snapshot endpoint
    opens it via OpenCV and grabs a frame. ``snapshot_url`` is the
    vendor's native HTTP snapshot path (ONVIF ``GetSnapshotUri``) which
    is much cheaper than an RTSP handshake; the route prefers it when
    present and falls back to RTSP otherwise.

    Issue #44: ``name``/``vendor``/``model`` carry the operator-facing
    labels from the platform heartbeat. The snapshot route ignores them;
    the Managed cameras panel's lister reads them. Optional so the
    Docker-mode last_discovery → CameraSnapshotSource path stays
    construction-compatible without naming a vendor."""

    rtsp_url: str
    snapshot_url: str | None = None
    name: str | None = None
    vendor: str | None = None
    model: str | None = None


class ClientR2Like(Protocol):
    """Structural type for the R2 surface the client-agent web UI depends on.

    Intentionally narrower than the gpu-service worker's R2 surface — the UI
    only needs upload + status + report read paths, never download_chunks or
    list_pending_job_ids.
    """

    def upload_input_chunk(
        self, job_id: str, fileobj: Any, chunk_name: str = "chunk_001.mp4"
    ) -> str: ...
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
 form { display: flex; flex-direction: column; gap: 1rem; margin-bottom: 2rem; }
 button { padding: .6rem 1rem; font-size: 1rem; }
 fieldset { border: 1px solid #ccc; padding: 1rem; }
 legend { font-weight: 600; }
 .recording { background: #fff5d6; padding: .5rem 1rem; border-left: 3px solid #d49a00; }
 .managed-cam { display: flex; gap: .8rem; align-items: center;
   border: 1px solid #ccc; padding: .5rem; margin: .4rem 0; }
 .managed-cam img { width: 160px; height: 90px; object-fit: cover;
   background: #eee; border: 1px solid #aaa; }
 .badge { padding: .1rem .4rem; border-radius: .2rem; font-size: .75em;
   color: #fff; margin-right: .4rem; }
 .badge-recording { background: #0a7; }
 .badge-idle      { background: #888; }
 .badge-failed    { background: #c33; }
</style>
</head>
<body>
<h1>cctv client-agent</h1>
{% if recording_state in ('recording', 'uploading') %}
<p class="recording">recording in progress (state: {{ recording_state }})</p>
{% endif %}

{% if managed_cameras %}
<fieldset>
<legend>Managed cameras</legend>
<ul style="list-style: none; padding: 0; margin: 0;">
{% for cam in managed_cameras %}
<li class="managed-cam">
  <img src="/cameras/{{ cam.id }}/snapshot" alt="{{ cam.name }}" loading="lazy">
  <div>
    {% set state = cam.recording_state or 'idle' %}
    {% set badge = state if state in ('recording', 'idle', 'failed') else 'idle' %}
    <span class="badge badge-{{ badge }}">{{ state }}</span>
    <strong>{{ cam.name }}</strong><br>
    <small>{{ cam.vendor }} {{ cam.model }}</small><br>
    <a href="/cameras/{{ cam.id }}/snapshot" target="_blank">Pokaż większy podgląd</a>
  </div>
</li>
{% endfor %}
</ul>
</fieldset>
{% endif %}

<fieldset>
<legend>Upload surveillance MP4</legend>
<form action="/upload" method="post" enctype="multipart/form-data">
  <input type="file" name="mp4" accept="video/mp4" required>
  <button type="submit">Upload &amp; queue for analysis</button>
</form>
</fieldset>

<fieldset>
<legend>Wykryj kamery ONVIF w LAN</legend>
<button type="button" onclick="discoverCameras()">Wykryj kamery</button>
<p id="discover-status" style="margin: .5rem 0; color: #555;"></p>
<ul id="discover-results" style="list-style: none; padding: 0; margin: 0;"></ul>
</fieldset>

<fieldset>
<legend>Record from RTSP camera</legend>
<form action="/start" method="post">
  <label>RTSP URL (manual)
    <input type="text" id="rtsp_url" name="rtsp_url" placeholder="rtsp://host/stream">
  </label>
  <button type="button" onclick="testConnection()">Test connection</button>
  <input type="hidden" id="camera_ip" name="camera_ip" value="">
  <p id="selected-camera" style="margin: .25rem 0; color: #036; min-height: 1.2em;"></p>
  <label>Duration
    <select name="duration_s" required>
      <option value="300">5 minutes</option>
      <option value="900">15 minutes</option>
      <option value="1800">30 minutes</option>
      <option value="2700">45 minutes</option>
      <option value="3600">1 hour</option>
      <option value="7200">2 hours</option>
      <option value="14400">4 hours</option>
      <option value="28800">8 hours</option>
    </select>
  </label>
  <button type="submit">Start recording</button>
</form>
<form action="/stop" method="post">
  <button type="submit">Stop current recording</button>
</form>
</fieldset>

<p><a href="/jobs">View job list →</a></p>
<script>
function testConnection() {
  var url = document.getElementById('rtsp_url').value.trim();
  if (!url) { alert('Enter an RTSP URL first.'); return; }
  var btn = event.target;
  btn.disabled = true; btn.textContent = 'Testing...';
  fetch('/test-connection', {
    method: 'POST',
    headers: {'Content-Type': 'application/x-www-form-urlencoded'},
    body: 'rtsp_url=' + encodeURIComponent(url)
  })
  .then(function(r) { return r.json(); })
  .then(function(data) {
    if (data.ok) { alert('Connection OK'); }
    else { alert('Connection failed:\\n' + (data.message || 'unknown error')); }
  })
  .catch(function(e) { alert('Request error: ' + e); })
  .finally(function() { btn.disabled = false; btn.textContent = 'Test connection'; });
}

function discoverCameras() {
  var btn = event.target;
  var status = document.getElementById('discover-status');
  var list = document.getElementById('discover-results');
  btn.disabled = true; btn.textContent = 'Skanuję...';
  status.textContent = 'Wysyłam ONVIF probe (timeout 5s)...';
  list.innerHTML = '';
  fetch('/cameras/discover')
   .then(function(r) { return r.json(); })
   .then(function(data) {
     if (data.error) {
       status.textContent = 'Błąd: ' + data.error;
       return;
     }
     if (!data.cameras.length) {
       status.textContent = 'Nie znaleziono kamer ONVIF (skan: ' + data.scanned_at + ').';
       return;
     }
     status.textContent = 'Znaleziono ' + data.cameras.length +
       ' kamer (skan: ' + data.scanned_at + '). Kliknij aby wybrać.';
     data.cameras.forEach(function(cam) {
       var li = document.createElement('li');
       li.style.cssText = 'border:1px solid #ccc;padding:.5rem;margin:.4rem 0;'
         + 'cursor:pointer;display:flex;gap:.8rem;align-items:center;';
       li.title = 'Kliknij, aby wkleić ' + cam.rtsp_url + ' do formularza';
       var img = document.createElement('img');
       img.alt = cam.vendor + ' ' + cam.model;
       img.style.cssText = 'width:120px;height:90px;object-fit:cover;'
         + 'background:#eee;border:1px solid #aaa;';
       if (cam.snapshot_url) { img.src = cam.snapshot_url; }
       var meta = document.createElement('div');
       // Three-way badge: ONVIF (Stage 1, highest trust) / RTSP scan
       // (Stage 2) / Tuya local (Stage 3, broadcast-only, no usable
       // stream URL until operator enables RTSP in the vendor app).
       var badgeColor, badgeText;
       if (cam.discovery_method === 'onvif') {
         badgeColor = '#0a7'; badgeText = 'ONVIF';
       } else if (cam.discovery_method === 'tuya-local') {
         badgeColor = '#a36'; badgeText = 'Tuya local';
       } else {
         badgeColor = '#d49a00'; badgeText = 'RTSP scan';
       }
       var badge = '<span style="background:' + badgeColor + ';color:#fff;'
         + 'padding:.1rem .4rem;border-radius:.2rem;font-size:.75em;'
         + 'margin-right:.4rem;">' + badgeText + '</span>';
       var needsManual = cam.needs_manual_url === true;
       var urlOrHint;
       var credsHint = '';
       if (cam.discovery_method === 'tuya-local') {
         // Issue #38: Tuya broadcast cameras (Setti+/Tapo/Vstarcam/…) ship
         // with RTSP disabled by default — the local broadcast only tells
         // us the device exists, not its stream URL. Tell the operator how
         // to turn RTSP on in the Setti+ / Tuya Smart app and then paste
         // the displayed URL into the manual RTSP input below.
         urlOrHint = '<em style="color:#a60;font-size:.85em;">'
           + 'Tuya cloud-only camera. Open the vendor app (Setti+ / Tuya Smart) '
           + '→ Camera settings → enable RTSP, then paste the URL below.</em>';
       } else if (needsManual) {
         // Issue #37: nginx-RTSP / per-device URI cameras (AnyKa-based IPCs)
         // have no usable URL template — the path is generated by the vendor
         // app. Replace the bogus rtsp://...:554/ with a hint pointing the
         // operator at the vendor app and ask them to paste the URL into the
         // manual RTSP input below.
         urlOrHint = '<em style="color:#a60;font-size:.85em;">'
           + 'Per-device URI — paste the full URL from the vendor app '
           + '(Tuya Smart / Setti+) into the RTSP URL field below.</em>';
       } else {
         urlOrHint = '<code style="font-size:.85em;">' + cam.rtsp_url + '</code>';
         if (cam.discovery_method === 'rtsp-scan') {
           credsHint = '<br><em style="color:#a60;font-size:.85em;">'
             + 'Wymaga RTSP_DEFAULT_USER/PASS (lub RTSP_CAM_<ip>_USER/PASS) w env</em>';
         }
       }
       meta.innerHTML = badge + '<strong>' + (cam.vendor || '?') + ' '
         + (cam.model || '?') + '</strong><br>' + cam.ip + ':' + cam.port
         + '<br>' + urlOrHint + credsHint;
       li.appendChild(img);
       li.appendChild(meta);
       li.onclick = function() {
         if (needsManual) {
           // No camera_ip flow — the vendor template can't build a working
           // URL for per-device URIs. Focus the manual input so the operator
           // can paste the URL straight from the vendor app.
           document.getElementById('camera_ip').value = '';
           var manualInput = document.getElementById('rtsp_url');
           manualInput.focus();
           document.getElementById('selected-camera').textContent =
             'Kamera ' + cam.ip + ' wymaga ręcznego URL — wklej go z aplikacji '
             + 'producenta (vendor app) w polu RTSP URL poniżej.';
           status.textContent = 'Wklej URL z aplikacji producenta dla ' + cam.ip;
           return;
         }
         // Issue #22: send camera_ip, not the full URL — server-side
         // resolver attaches creds from env so the password never reaches
         // the DOM. Clear rtsp_url so a stale manual entry doesn't carry
         // over (route gives camera_ip priority anyway, but the visible
         // input would be confusing).
         document.getElementById('camera_ip').value = cam.ip;
         document.getElementById('rtsp_url').value = '';
         document.getElementById('selected-camera').textContent =
           'Wybrana kamera: ' + cam.ip + ' (' + (cam.vendor || '?') + ' '
           + (cam.model || '?') + ') — kliknij "Start recording".';
         status.textContent = 'Wybrano ' + cam.ip
           + ' — creds zostaną pobrane z env po stronie serwera.';
       };
       list.appendChild(li);
     });
   })
   .catch(function(e) { status.textContent = 'Request error: ' + e; })
   .finally(function() { btn.disabled = false; btn.textContent = 'Wykryj kamery'; });
}
</script>
</body>
</html>
"""


DiscoverFn = Callable[[], list[DiscoveredCamera]]
CredentialsResolverFn = Callable[[str], "tuple[str, str] | None"]
CameraResolverFn = Callable[[str], "CameraSnapshotSource | None"]
SnapshotGrabberFn = Callable[[str, float], bytes]
ClockFn = Callable[[], float]
# Issue #44: appliance-supplied snapshot of currently-managed cameras (the
# join of the platform's heartbeat registry and the live ``active_recorders``
# map). Each row carries the bits the Managed cameras panel needs
# (``id``, ``name``, ``vendor``, ``model``, ``recording_state``); the web
# layer is dumb and just relays the payload. ``None`` disables the route
# and the panel — Docker mode.
ManagedCamerasFn = Callable[[], list[dict[str, Any]]]

# 30 s matches the polling cadence the gpu-exchange "My cameras" view is
# expected to use — short enough that the operator sees a near-current
# thumbnail, long enough that one tab open in the UI doesn't hammer the
# camera's HTTP snapshot endpoint.
_SNAPSHOT_CACHE_TTL_S = 30.0
# 5 s ffmpeg/cv2 open timeout. Matches the issue's "5s timeout" — long
# enough for a TCP handshake + a few RTP packets, short enough that the
# UI thread doesn't hang on a dead camera.
_SNAPSHOT_GRAB_TIMEOUT_S = 5.0


def create_app(
    client: ClientR2Like | None,
    *,
    job_id_factory=_default_job_id,
    recorder=None,
    discover_fn: DiscoverFn | None = None,
    credentials_resolver: CredentialsResolverFn | None = None,
    camera_resolver: CameraResolverFn | None = None,
    snapshot_grabber: SnapshotGrabberFn | None = None,
    managed_cameras_lister: ManagedCamerasFn | None = None,
    snapshot_cache_ttl_s: float = _SNAPSHOT_CACHE_TTL_S,
    clock: ClockFn = time.monotonic,
) -> Flask:
    """Build a Flask app bound to the given R2 client and job_id factory.

    The factory is injectable so tests can pin job_ids; production wires the
    default UUID4-based generator. The optional ``recorder`` is a
    :class:`client_agent.recorder.Recorder`-shaped object — duck-typed via
    ``start``/``stop``/``status``/``probe`` so the tests can pass a fake.
    Passing ``None`` disables the RTSP routes; production wires the real
    Recorder in ``agent.py``.

    ``discover_fn`` runs ONVIF WS-Discovery and returns the enriched camera
    list. ``None`` disables ``GET /cameras/discover``; production wires the
    real :func:`client_agent.discovery.discover_cameras`.

    ``credentials_resolver`` (issue #22) maps an IP to ``(user, pass)`` from
    the env-driven hierarchy. ``/start`` with ``{camera_ip, ...}`` looks up
    the camera in the in-memory cache populated by ``/cameras/discover``,
    asks the resolver for creds, builds a vendor-specific RTSP URL with
    creds inline, and hands it to the recorder. The cache + resolver split
    is what keeps passwords out of the DOM — the UI never sees them.
    """
    app = Flask(__name__)
    # Issue #22: cache the most recent discovery so /start with camera_ip
    # can rebuild the full RTSP URL without re-running multicast probe on
    # every recording start. Keyed by IP because that's what the UI sends;
    # value carries vendor + port so the URL template lookup is exact.
    last_discovery: dict[str, DiscoveredCamera] = {}

    @app.get("/")
    def upload_form() -> tuple[str, int, dict[str, str]]:
        from jinja2 import Environment, select_autoescape

        recording_state = recorder.status().state if recorder is not None else "idle"
        # Issue #44: render the Managed cameras panel only when the appliance
        # is wired (platform mode) AND the lister returns a non-empty list.
        # Docker mode → lister None → []; the {% if managed_cameras %} guard
        # in the template skips the panel entirely.
        managed_cameras = managed_cameras_lister() if managed_cameras_lister else []
        env = Environment(autoescape=select_autoescape(["html", "xml"]))
        template = env.from_string(_UPLOAD_FORM_HTML)
        html = template.render(
            recording_state=recording_state,
            managed_cameras=managed_cameras,
        )
        return html, 200, {"Content-Type": "text/html; charset=utf-8"}

    @app.post("/upload")
    def upload():
        # ``request.files["mp4"]`` is a Werkzeug FileStorage — its ``stream``
        # attribute is a SpooledTemporaryFile, so the multipart body never
        # gets fully loaded into RAM. We pass the FileStorage straight to R2;
        # boto3's upload_fileobj will call .read() on it. This is what makes
        # the 2 GB MP4 acceptance criterion possible.
        # Edge validation — refuse early so a noisy automation client never
        # creates a phantom job that the gpu-service worker has to fail.
        if client is None:
            return ("R2 backend disabled in platform mode", 503)
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

    # Presets cover sub-hour smoke/test windows (5/15/30/45m) and the
    # original MVP hourly slots (1/2/4/8h). The allowlist lives here — not
    # in the recorder — because the recorder core is length-agnostic and
    # we only want to gate user-supplied values at the HTTP boundary.
    _ALLOWED_DURATIONS_S = {300, 900, 1800, 2700, 3600, 7200, 14400, 28800}

    @app.post("/start")
    def start_recording():
        from client_agent.discovery import rtsp_template_for_vendor

        if recorder is None:
            return ("recorder not configured", 404)
        rtsp_url = (request.form.get("rtsp_url") or "").strip()
        camera_ip = (request.form.get("camera_ip") or "").strip()
        duration_raw = (request.form.get("duration_s") or "").strip()
        if camera_ip:
            # Issue #22: build the URL server-side from the last discovery
            # so the password never needs to round-trip through the DOM.
            cam = last_discovery.get(camera_ip)
            if cam is None:
                return (
                    f"camera_ip {camera_ip!r} not in last discovery — re-run scan",
                    400,
                )
            creds = credentials_resolver(camera_ip) if credentials_resolver else None
            if creds is None:
                return (
                    f"no credentials for {camera_ip!r}: set RTSP_DEFAULT_USER/PASS "
                    f"or RTSP_CAM_<ip>_USER/PASS in env",
                    400,
                )
            rtsp_url = rtsp_template_for_vendor(cam.vendor, cam.ip, cam.port, creds)
        elif not rtsp_url:
            return ("missing rtsp_url or camera_ip", 400)
        try:
            duration_s = int(duration_raw)
        except ValueError:
            return (f"invalid duration_s: {duration_raw!r}", 400)
        if duration_s not in _ALLOWED_DURATIONS_S:
            return (
                f"duration_s must be one of {sorted(_ALLOWED_DURATIONS_S)}",
                400,
            )
        # Production wires a Recorder whose .start() spawns a thread —
        # the route stays synchronous and returns immediately. Tests use
        # a synchronous fake; both paths converge here.
        try:
            from client_agent.recorder import RecorderBusy
        except ImportError:  # pragma: no cover — recorder module always present
            RecorderBusy = RuntimeError  # type: ignore[assignment, misc]
        try:
            recorder.start(url=rtsp_url, duration_s=duration_s)
        except RecorderBusy as exc:
            return (f"recorder busy: {exc}", 409)
        return redirect(url_for("jobs"))

    @app.post("/stop")
    def stop_recording():
        if recorder is None:
            return ("recorder not configured", 404)
        recorder.stop()
        return ("stopped", 200)

    @app.post("/test-connection")
    def test_connection():
        # No recorder wired (e.g. tests of the upload-only surface) → 404
        # so the operator gets a clear "not configured" rather than a
        # silent ok=false.
        if recorder is None:
            return ("recorder not configured", 404)
        rtsp_url = (request.form.get("rtsp_url") or "").strip()
        if not rtsp_url:
            return jsonify({"ok": False, "message": "missing rtsp_url"}), 400
        # 10s probe — long enough for a TCP handshake + a few packets,
        # short enough that the operator won't think the UI hung.
        result = recorder.probe(rtsp_url, timeout=10)
        return jsonify({"ok": result.ok, "message": result.message})

    @app.get("/jobs")
    def jobs() -> tuple[str, int, dict[str, str]]:
        if client is None:
            return ("R2 backend disabled in platform mode", 503, {"Content-Type": "text/plain"})
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
        if client is None:
            return None
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

    @app.get("/cameras/discover")
    def cameras_discover():
        # Same pattern as /start when no recorder is wired — a clear 404
        # so the operator (or a curl probe) sees "not configured" rather
        # than a confusing 500 traceback.
        if discover_fn is None:
            return ("camera discovery not configured", 404)
        scanned_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        # Discovery touches the network (ONVIF multicast + per-device SOAP),
        # so any failure mode — multicast blocked, slow camera, vendor
        # quirk — must surface as JSON ``error: <msg>`` instead of a 500.
        # Issue #21: response shape is ``{cameras, scanned_at, error: null|str}``
        # *always*, so the UI's fetch().then() can render uniformly.
        try:
            cameras = discover_fn()
        except Exception as exc:  # noqa: BLE001 — ONVIF/network is broad
            return jsonify({"cameras": [], "scanned_at": scanned_at, "error": str(exc)})
        # Issue #22: stash the result keyed by IP so /start with camera_ip
        # can rebuild the URL with creds without re-running multicast probe.
        # Each scan replaces the cache so a removed camera doesn't linger.
        last_discovery.clear()
        for c in cameras:
            last_discovery[c.ip] = c
        # Issue #22: scrub creds before they reach the DOM. Some ONVIF
        # firmwares return ``user:pass@`` in GetStreamUri and the Stage 2
        # RTSP-scan path embeds creds inline by design — both paths must
        # be sanitised here so the UI never sees a password.
        rows = []
        for c in cameras:
            row = asdict(c)
            row["rtsp_url"] = strip_credentials_from_url(row["rtsp_url"])
            rows.append(row)
        return jsonify({"cameras": rows, "scanned_at": scanned_at, "error": None})

    @app.get("/cameras/managed")
    def cameras_managed():
        # Issue #44: list of platform-approved cameras + their live recording
        # state. Docker mode never wires the lister (no platform registry to
        # join), so a 404 mirrors the discover_fn=None pattern: the operator
        # sees "not configured" rather than an empty 200 they have to debug.
        if managed_cameras_lister is None:
            return ("managed cameras not configured", 404)
        return jsonify({"cameras": managed_cameras_lister()})

    # Issue #41: per-camera snapshot endpoint. Two-tier cache: an in-memory
    # JPEG cache keyed by camera_id with a 30 s TTL absorbs UI polling so a
    # tab open in gpu-exchange's "My cameras" view doesn't hammer the camera.
    # Source resolution checks last_discovery first (Docker mode: operator
    # uses the camera's IP as the id), then falls back to ``camera_resolver``
    # (appliance mode: platform-supplied UUID resolves against heartbeat
    # config). JPEGs are never written to disk — privacy invariant.
    snapshot_cache: dict[str, tuple[float, bytes]] = {}

    def _resolve_snapshot_source(camera_id: str) -> CameraSnapshotSource | None:
        cam = last_discovery.get(camera_id)
        if cam is not None:
            return CameraSnapshotSource(rtsp_url=cam.rtsp_url, snapshot_url=cam.snapshot_url)
        if camera_resolver is not None:
            return camera_resolver(camera_id)
        return None

    @app.get("/cameras/<camera_id>/snapshot")
    def camera_snapshot(camera_id: str):
        if recorder is None or snapshot_grabber is None:
            return ("camera snapshot not configured", 404)
        now = clock()
        cached = snapshot_cache.get(camera_id)
        if cached is not None and (now - cached[0]) < snapshot_cache_ttl_s:
            return (cached[1], 200, {"Content-Type": "image/jpeg"})
        source = _resolve_snapshot_source(camera_id)
        if source is None:
            return (f"camera not found: {camera_id}", 404)
        # Prefer the vendor's native HTTP snapshot when discovery surfaced
        # it (ONVIF GetSnapshotUri) — single HTTP GET vs. opening an RTSP
        # session. Falls back to RTSP frame-grab when absent.
        url = source.snapshot_url or source.rtsp_url
        try:
            jpeg = snapshot_grabber(url, _SNAPSHOT_GRAB_TIMEOUT_S)
        except Exception as exc:  # noqa: BLE001 — cv2/ffmpeg/network failures
            # ``last_discovery`` stores the *raw* DiscoveredCamera, which on
            # the Stage-2 RTSP-scan path embeds ``user:pass@`` in
            # ``rtsp_url`` by design. The grabber's exception message
            # typically interpolates that URL — echoing ``exc`` verbatim
            # into the 503 body would leak the operator's creds to whoever
            # called the endpoint. Log server-side (where the platform
            # mode's journald is the appropriate audience), return only
            # the camera_id in the body.
            logger.warning("snapshot grab failed for camera_id=%s: %s", camera_id, exc)
            return (f"snapshot grab failed for {camera_id}", 503)
        snapshot_cache[camera_id] = (now, jpeg)
        return (jpeg, 200, {"Content-Type": "image/jpeg"})

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
