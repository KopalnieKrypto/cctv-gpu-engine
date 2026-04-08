"""Tests for the client-agent container entrypoint (issue #7).

The entrypoint is a thin wire-up: read R2 credentials from env, construct
an :class:`R2Client`, build the Flask app via :func:`create_app`, run on
:8080. The interesting behaviour lives in ``web_test.py`` and
``r2_client_test.py``; this module just pins the wire-up so a regression
in env-var names or port can't slip past CI.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def test_main_constructs_r2_client_from_env_and_runs_flask_on_8080(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``main()`` should:

    1. Read R2 creds from env (``R2_ENDPOINT``, ``R2_ACCESS_KEY_ID``,
       ``R2_SECRET_ACCESS_KEY``, ``R2_BUCKET``).
    2. Construct an :class:`R2Client` with those values.
    3. Build the Flask app via :func:`create_app`.
    4. Run the app on ``0.0.0.0:8080``.

    We patch ``R2Client`` and the returned app's ``run`` so the test never
    binds a real socket or talks to R2.
    """
    monkeypatch.setenv("R2_ENDPOINT", "https://acct.r2.cloudflarestorage.com")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "AK-test")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "SK-test")
    monkeypatch.setenv("R2_BUCKET", "surveillance-data")

    fake_app = MagicMock()
    fake_client = MagicMock()

    with (
        patch("client_agent.agent.R2Client", return_value=fake_client) as r2_ctor,
        patch("client_agent.agent.create_app", return_value=fake_app) as create_app,
    ):
        from client_agent.agent import main

        main()

    # R2Client built with env-derived creds.
    r2_ctor.assert_called_once()
    kwargs = r2_ctor.call_args.kwargs
    assert kwargs["endpoint"] == "https://acct.r2.cloudflarestorage.com"
    assert kwargs["access_key"] == "AK-test"
    assert kwargs["secret_key"] == "SK-test"
    assert kwargs["bucket"] == "surveillance-data"

    # Flask app built with the constructed R2 client and a recorder
    # wired up — the recorder identity isn't asserted here (that's the
    # recorder unit tests' job); we just pin that *some* recorder is
    # passed so a regression that drops it can't slip past CI.
    create_app.assert_called_once()
    args, kwargs = create_app.call_args
    assert args == (fake_client,)
    assert "recorder" in kwargs
    assert kwargs["recorder"] is not None

    # And served on the port the Dockerfile / docker-compose.client.yml expose.
    fake_app.run.assert_called_once()
    run_kwargs = fake_app.run.call_args.kwargs
    assert run_kwargs.get("host") == "0.0.0.0"
    assert run_kwargs.get("port") == 8080
