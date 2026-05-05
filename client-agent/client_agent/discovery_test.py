"""Tests for ONVIF camera discovery (issue #21).

The discovery module's job: run WS-Discovery probe, enrich every match with
vendor/model/RTSP/snapshot info, return a list. The two network surfaces
(WS-Discovery multicast, ONVIF SOAP per device) are injected as ``probe_fn``
and ``enrich_fn`` so these unit tests never touch the network — mocks live
only at the system boundary.

The real implementations of ``probe_fn``/``enrich_fn`` use ``wsdiscovery``
and ``onvif-zeep`` and are exercised only by the manual test on a physical
camera in the operator's LAN (issue #21 acceptance criterion 7).
"""

from __future__ import annotations

from client_agent.discovery import (
    DiscoveredCamera,
    ProbeMatch,
    discover_cameras,
    identify_vendor_from_rtsp_options,
)


def test_discover_cameras_returns_list_from_probe_and_enrich() -> None:
    """End-to-end shape: every ProbeMatch coming out of the probe gets
    handed to ``enrich_fn``; whatever DiscoveredCamera the enricher returns
    ends up in the list, in the same order.

    This is the tracer bullet — proves the probe→enrich→list path is wired
    before we add any of the real ONVIF behavior."""
    probe_matches = [
        ProbeMatch(ip="192.168.1.10", port=80, xaddr="http://192.168.1.10/onvif/device_service"),
        ProbeMatch(ip="192.168.1.11", port=80, xaddr="http://192.168.1.11/onvif/device_service"),
    ]
    enriched = {
        "http://192.168.1.10/onvif/device_service": DiscoveredCamera(
            ip="192.168.1.10",
            port=80,
            vendor="Hikvision",
            model="DS-2CD2042",
            rtsp_url="rtsp://192.168.1.10:554/Streaming/Channels/101",
            snapshot_url="http://192.168.1.10/Streaming/Channels/101/picture",
        ),
        "http://192.168.1.11/onvif/device_service": DiscoveredCamera(
            ip="192.168.1.11",
            port=80,
            vendor="Dahua",
            model="IPC-HFW1230S",
            rtsp_url="rtsp://192.168.1.11:554/cam/realmonitor?channel=1&subtype=0",
            snapshot_url=None,
        ),
    }

    def fake_probe(timeout: float) -> list[ProbeMatch]:
        return list(probe_matches)

    def fake_enrich(match: ProbeMatch, _creds=None) -> DiscoveredCamera | None:
        return enriched[match.xaddr]

    cameras = discover_cameras(probe_fn=fake_probe, enrich_fn=fake_enrich)

    assert cameras == [enriched[m.xaddr] for m in probe_matches]


def test_discover_cameras_skips_matches_that_fail_to_enrich() -> None:
    """One unresponsive / non-conforming camera must not sink the whole list.

    ``enrich_fn`` returning ``None`` is the signal: the device replied to
    WS-Discovery but its ONVIF service didn't answer GetStreamUri (auth
    required, vendor quirk, network blip). The operator still sees every
    camera that *did* respond — the rest can be retried on a re-scan."""
    matches = [
        ProbeMatch(ip="10.0.0.1", port=80, xaddr="http://a/onvif"),
        ProbeMatch(ip="10.0.0.2", port=80, xaddr="http://b/onvif"),
        ProbeMatch(ip="10.0.0.3", port=80, xaddr="http://c/onvif"),
    ]
    good_a = DiscoveredCamera(
        ip="10.0.0.1", port=80, vendor="V", model="M", rtsp_url="rtsp://10.0.0.1/s"
    )
    good_c = DiscoveredCamera(
        ip="10.0.0.3", port=80, vendor="V", model="M", rtsp_url="rtsp://10.0.0.3/s"
    )

    def fake_probe(_: float) -> list[ProbeMatch]:
        return list(matches)

    def fake_enrich(match: ProbeMatch, _creds=None) -> DiscoveredCamera | None:
        return {"http://a/onvif": good_a, "http://b/onvif": None, "http://c/onvif": good_c}[
            match.xaddr
        ]

    cameras = discover_cameras(probe_fn=fake_probe, enrich_fn=fake_enrich)

    assert cameras == [good_a, good_c]


def test_discover_cameras_default_timeout_is_at_most_5_seconds() -> None:
    """Issue #21: ``WS-Discovery działa z timeoutem ≤ 5 s i nie blokuje
    request thread Flask poza ten limit``. The default goes straight into
    ``probe_fn(timeout)`` so the Flask handler is bounded without the route
    needing to know how WS-Discovery works."""
    captured: list[float] = []

    def spy_probe(timeout: float) -> list[ProbeMatch]:
        captured.append(timeout)
        return []

    discover_cameras(probe_fn=spy_probe, enrich_fn=lambda m, c: None)

    assert captured == [5.0]
    assert captured[0] <= 5.0


def test_discover_cameras_forwards_explicit_timeout_to_probe() -> None:
    """Caller-supplied timeout overrides the default — useful for tests and
    for the Flask handler if we ever want to lower the bound under load."""
    captured: list[float] = []

    def spy_probe(timeout: float) -> list[ProbeMatch]:
        captured.append(timeout)
        return []

    discover_cameras(timeout=2.5, probe_fn=spy_probe, enrich_fn=lambda m: None)

    assert captured == [2.5]


def test_discovered_camera_default_discovery_method_is_onvif() -> None:
    """``discovery_method`` distinguishes Stage 1 (ONVIF) from Stage 2
    (RTSP-scan) hits in the UI. Existing call sites construct cameras
    without specifying it — they're all ONVIF, so the default keeps them
    correct without code changes."""
    cam = DiscoveredCamera(
        ip="10.0.0.1", port=80, vendor="V", model="M", rtsp_url="rtsp://10.0.0.1/"
    )
    assert cam.discovery_method == "onvif"


def test_discover_cameras_appends_rtsp_scan_results_to_onvif() -> None:
    """Two-stage discovery: ONVIF Stage 1 plus RTSP-scan Stage 2. Stage 2
    catches cameras that don't expose ONVIF (disabled, locked-down vendor
    quirks) — issue #21 follow-up: cover all cases.

    The merge is order-preserving: ONVIF cameras render first (higher-trust
    signal — vendor/model came from the device itself), RTSP-scan after."""
    onvif_match = ProbeMatch(ip="10.0.0.1", port=80, xaddr="http://10.0.0.1/onvif")
    onvif_cam = DiscoveredCamera(
        ip="10.0.0.1", port=80, vendor="Hik", model="X", rtsp_url="rtsp://10.0.0.1/s"
    )
    rtsp_cam = DiscoveredCamera(
        ip="10.0.0.2",
        port=554,
        vendor="Dahua",
        model="",
        rtsp_url="rtsp://10.0.0.2:554/cam/realmonitor",
        discovery_method="rtsp-scan",
    )

    cams = discover_cameras(
        probe_fn=lambda _: [onvif_match],
        enrich_fn=lambda _m, _c: onvif_cam,
        rtsp_scan_fn=lambda timeout: [rtsp_cam],
    )

    assert cams == [onvif_cam, rtsp_cam]


def test_discover_cameras_dedupes_rtsp_scan_against_onvif_by_ip() -> None:
    """If a camera answered both Stage 1 (ONVIF) and Stage 2 (RTSP scan),
    the ONVIF result wins — it carries vendor/model/snapshot. Without this
    dedupe the operator sees the same physical camera twice with different
    metadata, which is confusing and breaks the click-to-paste flow."""
    onvif_match = ProbeMatch(ip="10.0.0.1", port=80, xaddr="http://10.0.0.1/onvif")
    onvif_cam = DiscoveredCamera(
        ip="10.0.0.1", port=80, vendor="Hik", model="X", rtsp_url="rtsp://10.0.0.1/s"
    )
    rtsp_dup = DiscoveredCamera(
        ip="10.0.0.1",  # SAME IP as the ONVIF hit
        port=554,
        vendor="Hikvision",
        model="",
        rtsp_url="rtsp://10.0.0.1:554/Streaming/Channels/101",
        discovery_method="rtsp-scan",
    )
    rtsp_unique = DiscoveredCamera(
        ip="10.0.0.2",
        port=554,
        vendor="Dahua",
        model="",
        rtsp_url="rtsp://10.0.0.2:554/cam/realmonitor",
        discovery_method="rtsp-scan",
    )

    cams = discover_cameras(
        probe_fn=lambda _: [onvif_match],
        enrich_fn=lambda _m, _c: onvif_cam,
        rtsp_scan_fn=lambda timeout: [rtsp_dup, rtsp_unique],
    )

    # ONVIF kept, dup dropped, unique RTSP kept.
    assert cams == [onvif_cam, rtsp_unique]


def test_discover_cameras_without_rtsp_scan_fn_runs_only_stage_1() -> None:
    """No ``rtsp_scan_fn`` → behavior matches pre-extension: only ONVIF.
    Lets existing call sites stay unchanged and lets tests of Stage 1
    behavior keep passing fakes only for ``probe_fn``/``enrich_fn``."""
    onvif_cam = DiscoveredCamera(
        ip="10.0.0.1", port=80, vendor="V", model="M", rtsp_url="rtsp://10.0.0.1/"
    )
    cams = discover_cameras(
        probe_fn=lambda _: [ProbeMatch(ip="10.0.0.1", port=80, xaddr="http://10.0.0.1/")],
        enrich_fn=lambda _m, _c: onvif_cam,
        # rtsp_scan_fn not supplied
    )
    assert cams == [onvif_cam]


def test_discovered_camera_accepts_rtsp_scan_method() -> None:
    """RTSP-scan results need an explicit method tag so the UI can show a
    different badge and the merge logic can pick ONVIF over RTSP-scan when
    both find the same IP."""
    cam = DiscoveredCamera(
        ip="10.0.0.2",
        port=554,
        vendor="Hikvision",
        model="",
        rtsp_url="rtsp://10.0.0.2:554/Streaming/Channels/101",
        discovery_method="rtsp-scan",
    )
    assert cam.discovery_method == "rtsp-scan"


# ===== RTSP-scan vendor identification (Stage 2) =====


def test_identify_vendor_hikvision_from_server_header() -> None:
    """Hikvision RTSP servers identify themselves with a ``Hikvision`` or
    ``Webs/Vx.x.xx`` token in the Server header. Manual test on the
    operator's LAN (192.168.50.2-9) confirmed this — that's how the code
    knows to use ``/Streaming/Channels/101`` as the URL template."""
    assert identify_vendor_from_rtsp_options("Hikvision-Webs/V3.4.96 build 18") == "Hikvision"
    assert identify_vendor_from_rtsp_options("HIKVISION") == "Hikvision"


def test_identify_vendor_dahua_from_server_header() -> None:
    """Dahua's stock firmware reports ``Dahua Rtsp Server`` or just ``Dahua``."""
    assert identify_vendor_from_rtsp_options("Dahua Rtsp Server") == "Dahua"


def test_identify_vendor_axis_from_server_header() -> None:
    assert identify_vendor_from_rtsp_options("AXIS Q1755") == "Axis"


def test_identify_vendor_reolink_from_server_header() -> None:
    """Reolink uses a stock ``Rtsp Server`` token but the camera HTTP server
    is ``Reolink``. We accept both because operators sometimes look at the
    HTTP page first and the same string flows through."""
    assert identify_vendor_from_rtsp_options("Reolink") == "Reolink"


def test_identify_vendor_foscam_from_server_header() -> None:
    """Foscam-style RTSP servers (also Wansview rebrands) report
    ``Netwave IP Camera`` — the chipset vendor."""
    assert identify_vendor_from_rtsp_options("Foscam") == "Foscam"
    assert identify_vendor_from_rtsp_options("Netwave IP Camera") == "Foscam"


def test_identify_vendor_unknown_returns_unknown() -> None:
    """Anything we can't match (white-label IPC, custom firmware, missing
    header) collapses to ``Unknown`` — the UI still shows the IP/RTSP URL,
    just without a vendor-specific path. Operator can copy/paste manually."""
    assert identify_vendor_from_rtsp_options("Random Server String") == "Unknown"
    assert identify_vendor_from_rtsp_options("") == "Unknown"
    assert identify_vendor_from_rtsp_options(None) == "Unknown"


# ===== Credentials resolver =====


def test_resolve_credentials_returns_none_when_env_empty() -> None:
    """No defaults, no per-IP override → None. The discovery code treats
    ``None`` as the signal to fall back to auth-less ONVIF or to render the
    RTSP URL without ``user:pass@``."""
    from client_agent.discovery import resolve_camera_credentials

    assert resolve_camera_credentials("192.168.50.2", {}) is None


def test_resolve_credentials_uses_default_when_no_per_ip_override() -> None:
    """``RTSP_DEFAULT_USER`` + ``RTSP_DEFAULT_PASS`` apply to every camera
    that doesn't have a specific override. This is the common case — one
    set of creds for a whole NVR."""
    from client_agent.discovery import resolve_camera_credentials

    env = {"RTSP_DEFAULT_USER": "admin", "RTSP_DEFAULT_PASS": "Secret1!"}
    assert resolve_camera_credentials("192.168.50.2", env) == ("admin", "Secret1!")


def test_resolve_credentials_per_ip_overrides_default() -> None:
    """Per-IP override wins. Important when one camera in a set has a
    different password (e.g. previously deployed with stock creds and
    never rotated)."""
    from client_agent.discovery import resolve_camera_credentials

    env = {
        "RTSP_DEFAULT_USER": "admin",
        "RTSP_DEFAULT_PASS": "Default1!",
        "RTSP_CAM_192_168_50_2_USER": "operator",
        "RTSP_CAM_192_168_50_2_PASS": "Special2#",
    }
    assert resolve_camera_credentials("192.168.50.2", env) == ("operator", "Special2#")
    # Other IPs still get the default.
    assert resolve_camera_credentials("192.168.50.3", env) == ("admin", "Default1!")


def test_resolve_credentials_sanitizes_dots_to_underscores() -> None:
    """IPs contain ``.`` which are invalid in env-var names. We document the
    mapping (``192.168.50.2`` → ``RTSP_CAM_192_168_50_2_*``) and pin it
    here so the convention can't drift silently."""
    from client_agent.discovery import resolve_camera_credentials

    env = {
        "RTSP_CAM_10_0_0_5_USER": "u",
        "RTSP_CAM_10_0_0_5_PASS": "p",
    }
    assert resolve_camera_credentials("10.0.0.5", env) == ("u", "p")


def test_resolve_credentials_returns_none_when_only_user_or_only_pass() -> None:
    """Half-configured envs are a misconfiguration, not a partial credential.
    Returning ``("admin", "")`` would make ONVIFCamera attempt blank-password
    auth which is worse than auth-less; return ``None`` and let the caller
    decide whether to attempt auth-less or skip."""
    from client_agent.discovery import resolve_camera_credentials

    assert resolve_camera_credentials("10.0.0.1", {"RTSP_DEFAULT_USER": "admin"}) is None
    assert resolve_camera_credentials("10.0.0.1", {"RTSP_DEFAULT_PASS": "x"}) is None


# ===== RTSP URL templates per vendor =====


def test_rtsp_template_hikvision_main_stream() -> None:
    """Hikvision main stream lives on ``/Streaming/Channels/101``. Sub
    stream is ``/102`` but Phase 1 picks main — it's what operators
    actually want to record from. URL ends up like
    ``rtsp://admin:Pass@192.168.50.2:554/Streaming/Channels/101``."""
    from client_agent.discovery import rtsp_template_for_vendor

    url = rtsp_template_for_vendor("Hikvision", "192.168.50.2", 554, ("admin", "Pass"))
    assert url == "rtsp://admin:Pass@192.168.50.2:554/Streaming/Channels/101"


def test_rtsp_template_dahua_main_stream() -> None:
    from client_agent.discovery import rtsp_template_for_vendor

    url = rtsp_template_for_vendor("Dahua", "10.0.0.5", 554, ("u", "p"))
    assert url == "rtsp://u:p@10.0.0.5:554/cam/realmonitor?channel=1&subtype=0"


def test_rtsp_template_axis() -> None:
    from client_agent.discovery import rtsp_template_for_vendor

    url = rtsp_template_for_vendor("Axis", "10.0.0.6", 554, ("u", "p"))
    assert url == "rtsp://u:p@10.0.0.6:554/axis-media/media.amp"


def test_rtsp_template_reolink() -> None:
    from client_agent.discovery import rtsp_template_for_vendor

    url = rtsp_template_for_vendor("Reolink", "10.0.0.7", 554, ("u", "p"))
    assert url == "rtsp://u:p@10.0.0.7:554/h264Preview_01_main"


def test_rtsp_template_foscam_uses_port_88() -> None:
    """Foscam IP cameras serve RTSP on :88 by default, not :554. The
    template includes the path ``/videoMain``. We honor the caller's
    ``port`` argument because the operator may have remapped it; default
    is 88 from the scanner side."""
    from client_agent.discovery import rtsp_template_for_vendor

    url = rtsp_template_for_vendor("Foscam", "10.0.0.8", 88, ("u", "p"))
    assert url == "rtsp://u:p@10.0.0.8:88/videoMain"


def test_rtsp_template_unknown_vendor_returns_bare_root() -> None:
    """Without a vendor we can't pick a path — return ``/`` so the operator
    sees the IP/port and can edit the path manually. Better than guessing
    wrong and giving them a confidently broken URL."""
    from client_agent.discovery import rtsp_template_for_vendor

    url = rtsp_template_for_vendor("Unknown", "10.0.0.9", 554, ("u", "p"))
    assert url == "rtsp://u:p@10.0.0.9:554/"


def test_rtsp_template_without_credentials_omits_user_pass() -> None:
    """No env creds yet → URL is still emitted but without ``user:pass@``.
    The UI surfaces this as "needs credentials" so the operator knows to
    populate ``.env.client``. Once they do, a re-scan paints the same row
    with creds inline."""
    from client_agent.discovery import rtsp_template_for_vendor

    url = rtsp_template_for_vendor("Hikvision", "10.0.0.10", 554, None)
    assert url == "rtsp://10.0.0.10:554/Streaming/Channels/101"


def test_rtsp_template_url_encodes_password_special_chars() -> None:
    """Passwords with ``@``, ``:``, ``/`` would corrupt URL parsing.
    ``urllib.parse.quote`` with ``safe=""`` encodes them. Test pins this
    so a future "let's keep it readable" refactor doesn't reintroduce a
    bug where ``Pa$$:word`` becomes a URL parser exception."""
    from client_agent.discovery import rtsp_template_for_vendor

    url = rtsp_template_for_vendor("Hikvision", "10.0.0.11", 554, ("admin", "Pa$$:w@rd"))
    # ":" → "%3A", "@" → "%40", "$" → "%24"
    assert url == "rtsp://admin:Pa%24%24%3Aw%40rd@10.0.0.11:554/Streaming/Channels/101"


# ===== Credentials resolver wired into discover_cameras =====


def test_discover_cameras_passes_resolved_credentials_to_enrich() -> None:
    """The resolver runs *per match* and its return value is handed to
    ``enrich_fn`` as a second argument. This is what lets ``_real_enrich``
    call ``ONVIFCamera(ip, port, user, pass)`` instead of auth-less — the
    tracer-bullet test on the operator's LAN showed every Hikvision-clone
    requires auth even for ``GetDeviceInformation``."""
    captured: list[tuple[ProbeMatch, tuple[str, str] | None]] = []

    def spy_enrich(match, creds):
        captured.append((match, creds))
        return DiscoveredCamera(
            ip=match.ip, port=match.port, vendor="V", model="M", rtsp_url=f"rtsp://{match.ip}/"
        )

    matches = [
        ProbeMatch(ip="192.168.50.2", port=80, xaddr="http://192.168.50.2/onvif"),
        ProbeMatch(ip="192.168.50.99", port=80, xaddr="http://192.168.50.99/onvif"),
    ]
    creds_per_ip = {
        "192.168.50.2": ("admin", "SecretPass"),
        "192.168.50.99": None,  # no override, no default
    }
    cams = discover_cameras(
        probe_fn=lambda _: list(matches),
        enrich_fn=spy_enrich,
        credentials_resolver=lambda ip: creds_per_ip[ip],
    )
    assert len(cams) == 2
    assert captured == [
        (matches[0], ("admin", "SecretPass")),
        (matches[1], None),
    ]


def test_discover_cameras_default_resolver_returns_none() -> None:
    """Without a configured resolver, every camera is treated as no-creds —
    auth-less ONVIF and credential-free RTSP templates. Same flow as before
    the resolver was wired in, so existing tests and call sites stay valid
    without touching them."""
    captured = []

    def spy_enrich(match, creds):
        captured.append(creds)
        return DiscoveredCamera(ip=match.ip, port=80, vendor="V", model="M", rtsp_url="rtsp://x/")

    discover_cameras(
        probe_fn=lambda _: [ProbeMatch(ip="1.2.3.4", port=80, xaddr="http://1.2.3.4/")],
        enrich_fn=spy_enrich,
    )
    assert captured == [None]


# ===== Vendor fallback by open-port fingerprint =====


def test_guess_vendor_hikvision_from_port_8000() -> None:
    """Manual test on the operator's LAN (192.168.50.2-9) showed: Hikvision
    (and many rebrands) strip the ``Server:`` header from RTSP OPTIONS
    responses for security-through-obscurity. The cameras still ship a
    proprietary management port at 8000 which is the practical fingerprint.
    Combined with port 554, that's a Hikvision."""
    from client_agent.discovery import guess_vendor_from_open_ports

    assert guess_vendor_from_open_ports({80, 554, 8000}) == "Hikvision"
    assert guess_vendor_from_open_ports({554, 8000}) == "Hikvision"


def test_guess_vendor_dahua_from_port_37777() -> None:
    """Dahua's proprietary management port is 37777 (their CMS/DMSS protocol).
    Like Hikvision, Dahua firmware sometimes drops the RTSP Server header on
    purpose; 37777 + 554 is the reliable shape."""
    from client_agent.discovery import guess_vendor_from_open_ports

    assert guess_vendor_from_open_ports({554, 37777}) == "Dahua"


def test_guess_vendor_unknown_when_no_signature_port() -> None:
    """Just RTSP open isn't enough to guess — could be any number of camera
    or NVR vendors that aren't Hikvision/Dahua. Return ``Unknown`` so the
    UI shows a generic ``rtsp://ip/`` template the operator can edit."""
    from client_agent.discovery import guess_vendor_from_open_ports

    assert guess_vendor_from_open_ports({554}) == "Unknown"
    assert guess_vendor_from_open_ports({80, 554}) == "Unknown"
    assert guess_vendor_from_open_ports(set()) == "Unknown"
