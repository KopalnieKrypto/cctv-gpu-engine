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
    needs_manual_url: bool = False


ProbeFn = Callable[[float], list[ProbeMatch]]
EnrichFn = Callable[[ProbeMatch, "tuple[str, str] | None"], "DiscoveredCamera | None"]
RtspScanFn = Callable[[float], list["DiscoveredCamera"]]
TuyaScanFn = Callable[[float], list["DiscoveredCamera"]]
CredentialsResolver = Callable[[str], "tuple[str, str] | None"]


def discover_cameras(
    *,
    timeout: float = 5.0,
    probe_fn: ProbeFn | None = None,
    enrich_fn: EnrichFn | None = None,
    rtsp_scan_fn: RtspScanFn | None = None,
    tuya_scan_fn: TuyaScanFn | None = None,
    credentials_resolver: CredentialsResolver | None = None,
) -> list[DiscoveredCamera]:
    """Three-stage camera discovery: ONVIF → RTSP port-scan → Tuya broadcast.

    Stage 1 (ONVIF): ``probe_fn(timeout)`` → multicast WS-Discovery probe;
    each match goes through ``enrich_fn`` which fetches vendor/model/RTSP
    URL/snapshot via SOAP. ``timeout`` is capped at 5s by the Flask handler
    so the request thread can't block longer.

    Stage 2 (RTSP scan, optional): ``rtsp_scan_fn(timeout)`` scans the local
    subnet for port 554 and identifies vendors via RTSP OPTIONS Server
    headers. Catches cameras that don't expose ONVIF (disabled, vendor
    quirks). Results whose IP already came back from Stage 1 are dropped —
    ONVIF wins because it carries authoritative vendor/model.

    Stage 3 (Tuya local broadcast, optional, issue #38): ``tuya_scan_fn``
    listens for the Tuya UDP 6666/6667 local-discovery broadcast. Catches
    cloud-paired Tuya/Setti+/Tapo IPCs that don't expose ONVIF and have
    RTSP disabled by default — invisible to both Stage 1 and Stage 2.
    Results whose IP already came back from Stage 1 or Stage 2 are dropped
    (ONVIF > RTSP > Tuya): the upstream stages carry higher-trust info
    (authoritative vendor + working stream URL).

    All callables default to the real network-touching implementations;
    tests pass fakes. Pass ``rtsp_scan_fn=None`` / ``tuya_scan_fn=None``
    (the default) to disable a stage entirely."""
    probe_fn = probe_fn or _real_probe
    enrich_fn = enrich_fn or _real_enrich
    resolve = credentials_resolver or (lambda _ip: None)

    matches = probe_fn(timeout)
    onvif_cams = [cam for m in matches if (cam := enrich_fn(m, resolve(m.ip))) is not None]

    rtsp_cams: list[DiscoveredCamera] = []
    if rtsp_scan_fn is not None:
        onvif_ips = {c.ip for c in onvif_cams}
        rtsp_cams = [c for c in rtsp_scan_fn(timeout) if c.ip not in onvif_ips]

    tuya_cams: list[DiscoveredCamera] = []
    if tuya_scan_fn is not None:
        upstream_ips = {c.ip for c in onvif_cams} | {c.ip for c in rtsp_cams}
        tuya_cams = [c for c in tuya_scan_fn(timeout) if c.ip not in upstream_ips]

    return onvif_cams + rtsp_cams + tuya_cams


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


def inject_credentials(url: str, credentials: tuple[str, str] | None) -> str:
    """Re-attach ``user:pass@`` userinfo to an ``rtsp://`` URL.

    The exact inverse of :func:`strip_credentials_from_url`. The platform stores
    every stream URL credential-free (issue #22), so any consumer that must
    actually *open* the stream — the buffer recorder handing the URL to ffmpeg —
    has to re-attach the operator's ``cameras.env`` creds first. Without this an
    ONVIF-discovered camera (whose ``GetStreamUri`` returns a bare
    ``rtsp://host:port/path``) makes ffmpeg 401 and the recorder dies on loop.

    No-op when ``credentials is None`` or the URL already carries userinfo (so it
    is safe to apply unconditionally). The password is percent-encoded
    (``quote(safe='')``, matching :func:`build_rtsp_url`) so specials like ``#``,
    ``@`` and ``:`` survive URL parsing downstream.
    """
    if credentials is None:
        return url
    parsed = urlparse(url)
    if parsed.username or parsed.password or not parsed.hostname:
        return url
    from urllib.parse import quote

    user, password = credentials
    userinfo = f"{quote(user, safe='')}:{quote(password, safe='')}@"
    netloc = f"{userinfo}{parsed.hostname}"
    if parsed.port is not None:
        netloc = f"{userinfo}{parsed.hostname}:{parsed.port}"
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

    Unit tests cover the transport-timeout wiring + happy path + ReadTimeout
    swallow (issue #39) by monkeypatching ``onvif.ONVIFCamera``; the original
    network roundtrip stays covered by the manual test on a real camera in
    the operator's LAN (issue #21 #7).
    """
    import os

    try:
        from onvif import ONVIFCamera
    except ImportError as exc:
        raise RuntimeError(
            "onvif-zeep not installed — run `uv sync` to pull discovery deps"
        ) from exc
    from zeep.transports import Transport

    user, password = credentials if credentials is not None else ("", "")
    # Issue #39: bound the SOAP transport so a stalled endpoint can't hang the
    # appliance startup loop. ``operation_timeout`` is the load-bearing one —
    # zeep's POST path (the SOAP-call codepath) uses it, not ``load_timeout``.
    # Tunable via ``ONVIF_ENRICH_TIMEOUT_S`` for slow PoE switches / quiet LANs.
    timeout_s = int(os.environ.get("ONVIF_ENRICH_TIMEOUT_S", "5"))
    transport = Transport(timeout=timeout_s, operation_timeout=timeout_s)
    try:
        cam = ONVIFCamera(match.ip, match.port, user, password, transport=transport)
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


def _build_rtsp_scan_camera(
    ip: str,
    port: int,
    vendor: str,
    credentials: tuple[str, str] | None,
) -> DiscoveredCamera:
    """Build a Stage-2 :class:`DiscoveredCamera` from a vendor fingerprint.

    Issue #37: when ``vendor == "Unknown"`` (both the RTSP Server header and
    the proprietary-port fingerprint failed), the camera's RTSP path is
    almost certainly per-device — Tuya/Setti+/AnyKa cloud-paired IPCs
    generate an opaque token URL surfaced only in the vendor app. Returning
    ``rtsp://...:554/`` here makes ffmpeg silently 404 and gives the
    operator no signal that the URL is wrong, so we instead emit an empty
    ``rtsp_url`` + ``needs_manual_url=True`` and a verbose vendor string
    that the UI can render alongside a "paste the URL from the vendor app"
    hint.

    Known vendors (Hikvision/Dahua/Axis/Reolink/Foscam) keep the previous
    behavior: a vendor-specific URL pre-filled via
    :func:`rtsp_template_for_vendor`, ``needs_manual_url=False``."""
    if vendor == "Unknown":
        return DiscoveredCamera(
            ip=ip,
            port=port,
            vendor="Unknown (nginx-RTSP / per-device URI)",
            model="",
            rtsp_url="",
            snapshot_url=None,
            discovery_method="rtsp-scan",
            needs_manual_url=True,
        )
    return DiscoveredCamera(
        ip=ip,
        port=port,
        vendor=vendor,
        model="",
        rtsp_url=rtsp_template_for_vendor(vendor, ip, port, credentials),
        snapshot_url=None,
        discovery_method="rtsp-scan",
    )


_TUYA_VENDOR_LABEL = "Tuya (Setti+/Tapo/Vstarcam/…)"
"""Verbose vendor string for Stage-3 Tuya broadcast rows.

Tuya is the dominant white-label firmware behind Polish OEM brands (Setti+,
Hama, Orno) and global rebrands (Tapo, Wyze, Vstarcam). The verbose label
helps the operator recognise the camera even when their box has a generic
brand sticker — the underlying chipset/firmware is what answers the
broadcast, not the brand name."""

_TUYA_LOCAL_PORT = 6668
"""Tuya local-protocol TCP port. The broadcast lives on UDP 6666/6667 but
6668 is the data port the camera streams cloud-P2P over — that's what we
record as the port for the discovered row, since the broadcast ports are
discovery-only and not useful to the operator."""


def _build_tuya_camera(ip: str, gw_id: str, product_key: str) -> DiscoveredCamera:
    """Pure builder for a Stage-3 :class:`DiscoveredCamera` from one Tuya
    ``deviceScan`` payload entry.

    Tuya broadcast payloads carry ``gwId`` (per-device gateway id) and
    ``productKey`` (firmware/SKU id). We surface ``productKey`` as
    ``model`` because it's the operator-recognisable identifier in the
    Setti+/Tuya app product catalogue; ``gwId`` is unique-per-device noise.

    The RTSP URL is empty by design: the broadcast tells us the camera
    exists but the streaming URI is generated by the vendor app and never
    appears in the broadcast payload. ``needs_manual_url=True`` lets the
    UI render a hint pointing the operator at the vendor app (Setti+ /
    Tuya Smart) → Camera settings → enable RTSP → paste the URL.
    ``gw_id`` is currently unused but accepted in the signature so the
    factory closure (which already has it from tinytuya) doesn't drop it
    — a future debug-render of the broadcast payload may want it."""
    del gw_id  # accepted for forward-compat; not surfaced in the row yet
    return DiscoveredCamera(
        ip=ip,
        port=_TUYA_LOCAL_PORT,
        vendor=_TUYA_VENDOR_LABEL,
        model=product_key,
        rtsp_url="",
        snapshot_url=None,
        discovery_method="tuya-local",
        needs_manual_url=True,
    )


def make_real_tuya_scan(timeout: float = 5.0) -> TuyaScanFn:
    """Factory binding a :class:`TuyaScanFn` to a ``tinytuya.deviceScan``.

    Tuya's local-discovery protocol broadcasts each device's ``gwId`` /
    ``productKey`` on UDP 6666/6667 in clear text — no cloud credentials
    required, purely passive listening. ``tinytuya.deviceScan`` does the
    listening for us (``maxretry=2`` keeps the total wallclock bounded
    even on a quiet LAN).

    Like :func:`_real_probe` and :func:`make_real_rtsp_scan`, the network
    call lives behind a thin closure so unit tests inject a synthetic
    ``tuya_scan_fn`` and never hit the network. Returned signature matches
    ``Callable[[float], list[DiscoveredCamera]]`` — the bound ``timeout``
    here is the *outer* bound; the inner ``timeout`` argument is currently
    unused by ``tinytuya.deviceScan`` but kept so the signature matches
    other scan factories and a future timeout-honoring tinytuya release
    can pick it up without changing the caller.
    """

    def _scan(_timeout: float) -> list[DiscoveredCamera]:
        try:
            import tinytuya
        except ImportError as exc:
            raise RuntimeError(
                "tinytuya not installed — run `uv sync` to pull discovery deps"
            ) from exc

        results = tinytuya.deviceScan(verbose=False, maxretry=2)
        cams: list[DiscoveredCamera] = []
        for ip, payload in (results or {}).items():
            gw_id = str(payload.get("gwId") or "")
            product_key = str(payload.get("productKey") or "")
            cams.append(_build_tuya_camera(ip=ip, gw_id=gw_id, product_key=product_key))
        return cams

    del timeout  # accepted for symmetry with make_real_rtsp_scan
    return _scan


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
            cams.append(_build_rtsp_scan_camera(ip, 554, vendor, creds))
        return cams

    return _scan
