"""Tests for the client-agent Flask-app factory (``build_app``).

Since #29 there is no Docker container entrypoint and no R2 credential
path: ``build_app`` wires the shared Flask UI with ``client=None`` (the
appliance uploads via presigned URLs). These tests pin that wiring so a
regression can't reintroduce an R2 backend or drop the recorder. The
interesting route behaviour lives in ``web_test.py``.
"""

from __future__ import annotations

from client_agent.agent import BuiltApp, build_app


def test_build_app_wires_recorder_and_recordings_root(tmp_path) -> None:  # noqa: ANN001
    """``build_app`` always wires a recorder (so the appliance can drive it
    from the heartbeat loop) and honours the injected recordings_root,
    creating it if absent."""
    recordings = tmp_path / "recs"
    built = build_app({}, recordings_root=recordings)

    assert isinstance(built, BuiltApp)
    assert built.recorder is not None
    assert built.recordings_root == recordings
    assert recordings.is_dir()


def test_build_app_has_no_r2_backend_so_legacy_routes_503(tmp_path) -> None:  # noqa: ANN001
    """No R2 credentials are read or required (#29). The legacy R2-backed
    routes must return 503 rather than crash — the appliance serves the same
    app but its data path is presigned URLs, not these routes."""
    built = build_app({}, recordings_root=tmp_path / "recs")
    client = built.app.test_client()

    assert client.get("/jobs").status_code == 503
    resp = client.post(
        "/start",
        data={"rtsp_url": "rtsp://camera.local/stream", "duration_s": "3600"},
    )
    assert resp.status_code == 503


def test_build_app_ignores_stale_r2_env_vars(tmp_path) -> None:  # noqa: ANN001
    """Even if stale R2_* creds linger in the environment (an appliance whose
    ``r2.env`` hasn't been deleted yet), ``build_app`` must NOT construct an
    R2 client — the routes stay 503. Guards against a silent regression that
    revives the credential path the #29 cleanup removed."""
    env = {
        "R2_ENDPOINT": "https://acct.r2.cloudflarestorage.com",
        "R2_ACCESS_KEY_ID": "AK",
        "R2_SECRET_ACCESS_KEY": "SK",
        "R2_BUCKET": "surveillance-data",
    }
    built = build_app(env, recordings_root=tmp_path / "recs")

    assert built.app.test_client().get("/jobs").status_code == 503
