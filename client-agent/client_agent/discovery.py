"""ONVIF camera discovery for the client-agent (issue #21).

Two-stage discovery:

1. **Probe** — WS-Discovery multicast (UDP 239.255.255.250:3702) finds every
   ONVIF-capable device on the LAN and returns each one's ``xaddr`` (its
   ONVIF device service URL).
2. **Enrich** — for each probe match, query the device's ONVIF service for
   ``GetDeviceInformation``, ``GetStreamUri`` and ``GetSnapshotUri``, build
   a :class:`DiscoveredCamera` row.

Both stages are injectable so unit tests never touch the network. The real
implementations live alongside as ``_real_probe`` / ``_real_enrich`` and use
``wsdiscovery`` + ``onvif-zeep`` — they are covered only by the manual test
on a physical camera in the operator's LAN (issue #21 acceptance #7).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProbeMatch:
    """Raw output of one WS-Discovery ProbeMatch.

    ``xaddr`` is the device's ONVIF service URL, the input to enrichment.
    ``ip`` and ``port`` are pulled out for convenience so the UI can show
    them even when enrichment fails (operator can still type the URL by
    hand)."""

    ip: str
    port: int
    xaddr: str


@dataclass(frozen=True)
class DiscoveredCamera:
    """A camera discovered on the LAN, fully enriched.

    ``rtsp_url`` is what the operator actually wants — clicks on the UI list
    paste this into the recording form. ``snapshot_url`` is shown as a
    thumbnail so the operator can identify *which* physical camera they're
    selecting (numbers alone aren't memorable). It's optional because not
    every ONVIF device exposes ``GetSnapshotUri``."""

    ip: str
    port: int
    vendor: str
    model: str
    rtsp_url: str
    snapshot_url: str | None = None
    discovery_method: str = "onvif"


ProbeFn = Callable[[float], list[ProbeMatch]]
EnrichFn = Callable[[ProbeMatch, "tuple[str, str] | None"], "DiscoveredCamera | None"]
RtspScanFn = Callable[[float], list["DiscoveredCamera"]]
CredentialsResolver = Callable[[str], "tuple[str, str] | None"]


def discover_cameras(
    *,
    timeout: float = 5.0,
    probe_fn: ProbeFn | None = None,
    enrich_fn: EnrichFn | None = None,
    rtsp_scan_fn: RtspScanFn | None = None,
    credentials_resolver: CredentialsResolver | None = None,
) -> list[DiscoveredCamera]:
    """Two-stage camera discovery: ONVIF first, then RTSP port-scan.

    Stage 1 (ONVIF): ``probe_fn(timeout)`` → multicast WS-Discovery probe;
    each match goes through ``enrich_fn`` which fetches vendor/model/RTSP
    URL/snapshot via SOAP. ``timeout`` is capped at 5s by the Flask handler
    so the request thread can't block longer.

    Stage 2 (RTSP scan, optional): ``rtsp_scan_fn(timeout)`` scans the local
    subnet for port 554 and identifies vendors via RTSP OPTIONS Server
    headers. Catches cameras that don't expose ONVIF (disabled, vendor
    quirks). Results whose IP already came back from Stage 1 are dropped —
    ONVIF wins because it carries authoritative vendor/model.

    All three callables default to the real network-touching implementations;
    tests pass fakes. Pass ``rtsp_scan_fn=None`` (the default) to disable
    Stage 2 entirely — used by tests that only care about Stage 1 behavior."""
    probe_fn = probe_fn or _real_probe
    enrich_fn = enrich_fn or _real_enrich
    resolve = credentials_resolver or (lambda _ip: None)

    matches = probe_fn(timeout)
    onvif_cams = [cam for m in matches if (cam := enrich_fn(m, resolve(m.ip))) is not None]

    if rtsp_scan_fn is None:
        return onvif_cams

    onvif_ips = {c.ip for c in onvif_cams}
    rtsp_cams = [c for c in rtsp_scan_fn(timeout) if c.ip not in onvif_ips]
    return onvif_cams + rtsp_cams


def _ip_to_env_suffix(ip: str) -> str:
    """``192.168.50.2`` → ``192_168_50_2``. Env-var names can't contain ``.``."""
    return ip.replace(".", "_").replace(":", "_")


def resolve_camera_credentials(ip: str, env: Mapping[str, str]) -> tuple[str, str] | None:
    """Per-IP override → ``RTSP_DEFAULT_*`` → ``None``.

    Env layout (issue #21 / Phase 2):

    * ``RTSP_DEFAULT_USER`` + ``RTSP_DEFAULT_PASS`` — apply to every camera
      that lacks a per-IP override.
    * ``RTSP_CAM_<sanitized_ip>_USER`` + ``..._PASS`` — override for one
      camera. Sanitisation: dots and colons → underscores.

    A half-configured pair (only USER, only PASS) collapses to ``None``
    instead of silently sending blank credentials to the camera. The
    caller (ONVIF enrich, RTSP template builder) treats ``None`` as
    "fall back to auth-less / leave URL without user:pass@" so the
    discovery still produces a useful row even when the operator hasn't
    populated the env yet.
    """
    suffix = _ip_to_env_suffix(ip)
    user = env.get(f"RTSP_CAM_{suffix}_USER")
    password = env.get(f"RTSP_CAM_{suffix}_PASS")
    if user and password:
        return (user, password)
    user = env.get("RTSP_DEFAULT_USER")
    password = env.get("RTSP_DEFAULT_PASS")
    if user and password:
        return (user, password)
    return None


_VENDOR_FINGERPRINTS = (
    # (lowercase substring, canonical vendor name) — order matters: more
    # specific tokens first. ``Netwave`` is the chipset vendor used by
    # Foscam-clone IPCs (Wansview, Vstarcam, etc.), so we collapse it to
    # Foscam since the URL template is the same.
    ("hikvision", "Hikvision"),
    ("webs/", "Hikvision"),  # Hikvision RTSP server token
    ("dahua", "Dahua"),
    ("axis", "Axis"),
    ("reolink", "Reolink"),
    ("foscam", "Foscam"),
    ("netwave", "Foscam"),
)


_VENDOR_RTSP_PATHS = {
    # Pinned to the **main** stream — operators want full-quality footage
    # for analysis, not the sub-stream's lower bitrate. Sub streams differ
    # per vendor and aren't worth the UI complexity in Phase 1.
    "Hikvision": "/Streaming/Channels/101",
    "Dahua": "/cam/realmonitor?channel=1&subtype=0",
    "Axis": "/axis-media/media.amp",
    "Reolink": "/h264Preview_01_main",
    "Foscam": "/videoMain",
}


def rtsp_template_for_vendor(
    vendor: str,
    ip: str,
    port: int,
    credentials: tuple[str, str] | None,
) -> str:
    """Build a vendor-specific RTSP URL with creds embedded.

    Vendor → path mapping is curated from real Hikvision/Dahua/Axis/Reolink/
    Foscam camera defaults. Unknown vendors get ``/`` so the operator at
    least sees IP:port and can patch the path by hand — better than guessing
    wrong and giving them a confidently broken URL.

    ``credentials=None`` produces ``rtsp://host:port/path`` without
    ``user:pass@``; the UI flags it as "needs credentials in .env.client".
    Special characters in the password are URL-encoded so passwords with
    ``@``/``:``/``/`` don't break URL parsing downstream (ffmpeg, the
    recorder, etc.)."""
    from urllib.parse import quote

    path = _VENDOR_RTSP_PATHS.get(vendor, "/")
    if credentials is not None:
        user, password = credentials
        userinfo = f"{quote(user, safe='')}:{quote(password, safe='')}@"
    else:
        userinfo = ""
    return f"rtsp://{userinfo}{ip}:{port}{path}"


def strip_credentials_from_url(url: str) -> str:
    """Remove the ``user:pass@`` userinfo from a URL.

    Issue #22: passwords must never reach the DOM or be visible to the
    browser. The web layer runs every ``rtsp_url`` produced by discovery
    through this helper before serialising to JSON, so even URLs that
    came back with creds embedded (ONVIF GetStreamUri on some firmwares,
    Stage-2 RTSP-scan templates) are safe to render. The credential-free
    URL is paired with ``camera_ip`` in the UI; ``/start`` re-attaches
    creds server-side from the env-driven resolver.

    Implementation note: ``urlparse`` correctly identifies the userinfo
    boundary (the ``@`` before host:port), so a path containing ``@`` —
    legal per RFC 3986 §3.3 — is preserved.
    """
    parsed = urlparse(url)
    if not parsed.username and not parsed.password:
        return url
    host = parsed.hostname or ""
    netloc = host
    if parsed.port is not None:
        netloc = f"{host}:{parsed.port}"
    return parsed._replace(netloc=netloc).geturl()


def guess_vendor_from_open_ports(open_ports: set[int]) -> str:
    """Secondary vendor fingerprint based on management-port presence.

    Real cameras (manual test on the operator's LAN) often strip the RTSP
    ``Server:`` header to thwart trivial fingerprinting. The proprietary
    management ports leak through anyway:

    * Hikvision (and most rebrands): port 8000 (private SDK / SADP)
    * Dahua: port 37777 (CMS / DMSS)

    Returns the canonical vendor name when the signature matches, else
    ``Unknown`` — used as a fallback after :func:`identify_vendor_from_rtsp_options`
    comes back empty. Pure / no network."""
    if 8000 in open_ports:
        return "Hikvision"
    if 37777 in open_ports:
        return "Dahua"
    return "Unknown"


def identify_vendor_from_rtsp_options(server_header: str | None) -> str:
    """Map an RTSP ``Server:`` header value to a canonical vendor name.

    Used by the Stage 2 RTSP-scan path to pick a URL template (Hikvision's
    ``/Streaming/Channels/101`` differs from Dahua's ``/cam/realmonitor``,
    etc.). Returns ``"Unknown"`` for empty/missing/unrecognised headers —
    the UI still shows the camera, just without a pre-filled stream path.

    Pure / case-insensitive / no network."""
    if not server_header:
        return "Unknown"
    lowered = server_header.lower()
    for token, canonical in _VENDOR_FINGERPRINTS:
        if token in lowered:
            return canonical
    return "Unknown"


def _real_probe(timeout: float) -> list[ProbeMatch]:
    """WS-Discovery probe via the ``wsdiscovery`` package.

    Sends a multicast Probe to ``239.255.255.250:3702`` and collects every
    ProbeMatch reply within ``timeout`` seconds. Each device may publish
    multiple ``XAddrs``; we take the first because it's the one the device
    advertised first and that's typically the device-service URL.

    No unit test exercises this — it's covered only by the manual test on
    a real camera (issue #21 #7). If the import fails (deps not installed
    on a slim image) we surface a readable RuntimeError instead of an
    ImportError so the Flask error path renders something useful.
    """
    try:
        from wsdiscovery.discovery import ThreadedWSDiscovery
    except ImportError as exc:
        raise RuntimeError(
            "WSDiscovery not installed — run `uv sync` to pull discovery deps"
        ) from exc

    wsd = ThreadedWSDiscovery()
    wsd.start()
    try:
        services = wsd.searchServices(timeout=int(max(1, round(timeout))))
    finally:
        wsd.stop()

    matches: list[ProbeMatch] = []
    seen: set[str] = set()
    for svc in services:
        xaddrs = list(svc.getXAddrs() or [])
        if not xaddrs:
            continue
        xaddr = xaddrs[0]
        if xaddr in seen:
            continue
        seen.add(xaddr)
        parsed = urlparse(xaddr)
        if not parsed.hostname:
            continue
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        matches.append(ProbeMatch(ip=parsed.hostname, port=port, xaddr=xaddr))
    return matches


def _real_enrich(
    match: ProbeMatch, credentials: tuple[str, str] | None = None
) -> DiscoveredCamera | None:
    """Per-device ONVIF enrichment via ``onvif-zeep``.

    ``credentials`` come from :func:`resolve_camera_credentials` (env-driven)
    and are passed straight to ``ONVIFCamera``. The manual test on the
    operator's LAN (192.168.50.2-9 Hikvision-clones) showed that auth-less
    SOAP fails with a 404 ``ONVIF integrate function is disabled``-style
    fault on most camera firmware — env creds are the realistic happy
    path. We still try with empty creds when the resolver returns ``None``
    so the caller's discovery flow keeps working before the operator
    populates ``.env.client``. Failed auth → SOAP fault → ``None``, and
    :func:`discover_cameras` skips the row.

    No unit test exercises this — covered only by the manual test on a real
    camera (issue #21 #7).
    """
    try:
        from onvif import ONVIFCamera
    except ImportError as exc:
        raise RuntimeError(
            "onvif-zeep not installed — run `uv sync` to pull discovery deps"
        ) from exc

    user, password = credentials if credentials is not None else ("", "")
    try:
        cam = ONVIFCamera(match.ip, match.port, user, password)
        info = cam.devicemgmt.GetDeviceInformation()
        media = cam.create_media_service()
        profiles = media.GetProfiles()
        if not profiles:
            return None
        token = profiles[0].token
        stream = media.GetStreamUri(
            {
                "StreamSetup": {
                    "Stream": "RTP-Unicast",
                    "Transport": {"Protocol": "RTSP"},
                },
                "ProfileToken": token,
            }
        )
        rtsp_uri = str(stream.Uri)
        # Reject GetStreamUri responses that don't carry a real path. Some
        # firmwares (and the auth-less probe path on Hikvision-clones) return
        # bare `rtsp://host:port/` — recording from that 404s. Returning None
        # lets discover_cameras fall through to the Stage 2 RTSP-scan
        # vendor-template URL for the same IP, which is the one ffmpeg can
        # actually open.
        from urllib.parse import urlparse as _urlparse

        path = _urlparse(rtsp_uri).path
        if not path or path == "/":
            logger.info(
                "ONVIF GetStreamUri at %s bare path (%r); skipping for RTSP scan",
                match.xaddr,
                rtsp_uri,
            )
            return None
        snapshot_url: str | None = None
        try:
            snap = media.GetSnapshotUri({"ProfileToken": token})
            snapshot_url = getattr(snap, "Uri", None)
        except Exception as exc:  # noqa: BLE001 — vendor quirks vary widely
            logger.debug("GetSnapshotUri failed for %s: %s", match.xaddr, exc)
        return DiscoveredCamera(
            ip=match.ip,
            port=match.port,
            vendor=str(getattr(info, "Manufacturer", "") or ""),
            model=str(getattr(info, "Model", "") or ""),
            rtsp_url=rtsp_uri,
            snapshot_url=snapshot_url,
        )
    except Exception as exc:  # noqa: BLE001 — ONVIF surface is huge
        logger.info("ONVIF enrich failed for %s: %s", match.xaddr, exc)
        return None


def _local_ipv4_subnet() -> str | None:
    """Detect the host's primary IPv4 subnet as ``a.b.c.0/24``.

    Uses the standard "connect to a public IP and read the local end"
    trick — no packets actually leave because UDP connect is just a
    routing-table lookup. Returns ``None`` if no public route exists
    (offline appliance, link-only network) so the caller can skip Stage 2
    instead of crashing.
    """
    import socket as _socket

    s = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 1))
        local_ip = s.getsockname()[0]
    except OSError:
        return None
    finally:
        s.close()
    parts = local_ip.split(".")
    if len(parts) != 4:
        return None
    return f"{parts[0]}.{parts[1]}.{parts[2]}.0/24"


_CAMERA_PORTS = (554, 80, 8000, 37777)
"""Ports we scan during Stage 2:

* 554 — RTSP, the headline signal "this is a camera"
* 80  — HTTP management page (informational)
* 8000 — Hikvision proprietary port (vendor fingerprint)
* 37777 — Dahua proprietary port (vendor fingerprint)
"""


def _scan_camera_ports(subnet: str, timeout: float) -> dict[str, set[int]]:
    """Concurrent TCP-connect scan returning ``{ip: {open_ports...}}``.

    Per-IP×port timeout is bounded so a 5s outer cap stays ~5s even on a
    full /24 with 4 ports. Only IPs with port 554 open end up in the
    result — port 80 alone could be anything (router, IoT, NAS), the
    proprietary ports are vendor hints that only matter alongside RTSP."""
    import socket as _socket
    from concurrent.futures import ThreadPoolExecutor

    base = subnet.rsplit(".", 1)[0]
    targets = [(f"{base}.{i}", port) for i in range(1, 255) for port in _CAMERA_PORTS]
    per_host_timeout = max(0.15, timeout / 16)

    def probe(ip_port):
        ip, port = ip_port
        s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        s.settimeout(per_host_timeout)
        try:
            s.connect((ip, port))
            return (ip, port)
        except OSError:
            return None
        finally:
            s.close()

    found: dict[str, set[int]] = {}
    with ThreadPoolExecutor(max_workers=128) as ex:
        for hit in ex.map(probe, targets):
            if hit is None:
                continue
            ip, port = hit
            found.setdefault(ip, set()).add(port)
    # Filter to camera-shaped hosts: must speak RTSP.
    return {ip: ports for ip, ports in found.items() if 554 in ports}


def _rtsp_options_server(ip: str, port: int, timeout: float) -> str | None:
    """Send an RTSP OPTIONS request and return the ``Server:`` header value.

    Pure socket — RTSP is line-oriented, no library needed. Returns
    ``None`` on connection failure or absent header. Servers identify
    themselves here even before auth (OPTIONS is auth-free in nearly all
    RTSP implementations), which is exactly what we want for vendor
    fingerprinting in Stage 2."""
    import socket as _socket

    request = (
        f"OPTIONS rtsp://{ip}:{port}/ RTSP/1.0\r\n"
        "CSeq: 1\r\n"
        "User-Agent: cctv-client-agent/discovery\r\n\r\n"
    ).encode("ascii")
    s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect((ip, port))
        s.sendall(request)
        data = s.recv(4096).decode("latin-1", errors="replace")
    except OSError:
        return None
    finally:
        s.close()
    for line in data.splitlines():
        if line.lower().startswith("server:"):
            return line.split(":", 1)[1].strip()
    return None


def make_real_rtsp_scan(
    credentials_resolver: CredentialsResolver,
    *,
    subnet: str | None = None,
) -> RtspScanFn:
    """Factory that binds a :class:`RtspScanFn` to a credentials resolver.

    Stage 2 needs the resolver because the URL templates embed creds
    inline. Defining the scan as a closure keeps :func:`discover_cameras`'s
    signature simple and the resolver injection consistent with Stage 1
    (where ``discover_cameras`` calls the resolver directly).

    Subnet defaults to whatever :func:`_local_ipv4_subnet` detects; pass
    explicitly for tests or non-/24 deployments. Returns an empty list
    when no subnet can be detected — the appliance is offline and Stage 2
    has nothing to do.
    """

    def _scan(timeout: float) -> list[DiscoveredCamera]:
        target_subnet = subnet or _local_ipv4_subnet()
        if target_subnet is None:
            logger.info("RTSP scan skipped: no local /24 detected")
            return []
        ports_by_ip = _scan_camera_ports(target_subnet, timeout)
        cams: list[DiscoveredCamera] = []
        per_host_options_timeout = max(0.5, timeout / 8)
        for ip, open_ports in ports_by_ip.items():
            # Two-layer vendor fingerprint: try the RTSP Server: header first
            # (most informative when present), then fall back to proprietary-
            # port heuristics. Real Hikvision cameras strip the Server header,
            # so the port-based fallback is what actually labels them.
            server = _rtsp_options_server(ip, 554, per_host_options_timeout)
            vendor = identify_vendor_from_rtsp_options(server)
            if vendor == "Unknown":
                vendor = guess_vendor_from_open_ports(open_ports)
            creds = credentials_resolver(ip)
            cams.append(
                DiscoveredCamera(
                    ip=ip,
                    port=554,
                    vendor=vendor,
                    model="",
                    rtsp_url=rtsp_template_for_vendor(vendor, ip, 554, creds),
                    snapshot_url=None,
                    discovery_method="rtsp-scan",
                )
            )
        return cams

    return _scan
