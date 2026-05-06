"""Standalone appliance entrypoint for the client-agent (issue #23).

Run on bare metal (mini-PC, Raspberry Pi 5) without Docker:

    python -m client_agent.appliance --env-dir /etc/cctv-client

Reads camera + R2 credentials from disk (``cameras.env`` and ``r2.env`` in
the env-dir), serves the same Flask app the Docker entrypoint serves, but
through waitress (multithreaded WSGI) so concurrent UI requests do not
serialize through a single Werkzeug dev-server thread.
"""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import MutableMapping, Sequence
from pathlib import Path

from waitress import serve

from client_agent.agent import build_app

logger = logging.getLogger(__name__)


def load_env_files(env_dir: Path, environ: MutableMapping[str, str]) -> None:
    """Read ``cameras.env`` and ``r2.env`` from ``env_dir`` into ``environ``.

    Format follows systemd's ``EnvironmentFile=``: ``KEY=VALUE`` per line.
    """
    for name in ("cameras.env", "r2.env"):
        path = env_dir / name
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            key, sep, value = stripped.partition("=")
            if not sep:
                continue
            if key in environ:
                continue
            environ[key] = value


def default_recordings_dir(environ: MutableMapping[str, str]) -> Path:
    """Resolve the appliance default recordings directory per the XDG spec.

    Recordings are persistent state (kept across reboots until upload to R2
    succeeds), so XDG_STATE_HOME is the right base — not XDG_CACHE_HOME and
    not ``/tmp`` (PrivateTmp=yes in systemd would isolate it from operator
    inspection)."""
    base = environ.get("XDG_STATE_HOME") or f"{environ['HOME']}/.local/state"
    return Path(base) / "cctv-client" / "recordings"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI args for the appliance.

    Defaults align with the install layout of the planned systemd unit:
    env-dir at ``/etc/cctv-client`` (matches ``EnvironmentFile=``), and
    foreground execution (matches ``Type=simple`` — systemd manages the
    daemon side; the process itself just runs in the foreground)."""
    parser = argparse.ArgumentParser(
        prog="client_agent.appliance",
        description="Standalone client-agent for bare-metal (non-Docker) operation.",
    )
    parser.add_argument(
        "--env-dir",
        type=Path,
        default=Path("/etc/cctv-client"),
        help="Directory containing cameras.env and r2.env (default: /etc/cctv-client)",
    )
    parser.add_argument(
        "--foreground",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run in foreground (default; matches systemd Type=simple)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    args = parse_args(argv)
    load_env_files(args.env_dir, os.environ)
    recordings_root = default_recordings_dir(os.environ)
    built = build_app(os.environ, recordings_root=recordings_root)

    logger.info(
        "client-agent appliance starting on http://0.0.0.0:8080 "
        "(bucket=%s, recordings=%s, env_dir=%s)",
        built.bucket,
        built.recordings_root,
        args.env_dir,
    )
    serve(built.app, host="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()
