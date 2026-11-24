"""Publish a head trained on cctv-vps to the client's R2 key (gpu-exchange#314).

``train_station_head.py --package`` leaves a head and its card on cctv-vps.
This sends them where the platform says, the way the gpu-agent sends results:
the agent's own token asks data-service for presigned PUTs, the bytes go
straight to R2, and completion is reported with the card. The key comes from
the export, so nothing here names a tenant. Only the server set in the panel
(``station_head_publisher_server_id``) is let through.

Run it on cctv-vps as the agent's user, after a run worth keeping (D4):

    python benchmarks/activity/tools/publish_station_head.py \
        --export <export id> --head OUT/station-head-...onnx --card OUT/station-head-...card.json

Once completion is reported the version is closed (D9); a new run is a new export.
Standard library only, so it runs from any python3 on the box.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

#: (method, url, headers, body) -> (status, body). Injected by the tests.
Http = Callable[[str, str, dict, "bytes | None"], "tuple[int, bytes]"]

DEFAULT_PLATFORM = "https://api-panel.antra.one"
DEFAULT_STATE = Path.home() / ".local" / "state" / "gpu-agent" / "state.json"


class PublishError(Exception):
    """A step refused; the message is what the operator reads."""


def read_agent_token(state_file: Path) -> str:
    """The gpu-agent's ``agentToken`` from its state file. Never printed."""
    try:
        token = json.loads(state_file.read_text(encoding="utf-8")).get("agentToken")
    except (OSError, ValueError) as err:
        raise PublishError(f"cannot read the agent state file {state_file}: {err}") from err
    if not isinstance(token, str) or not token:
        raise PublishError(f"{state_file} holds no agentToken; is the gpu-agent registered?")
    return token


def _urllib_http(method: str, url: str, headers: dict, body: bytes | None) -> tuple[int, bytes]:
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as err:
        return err.code, err.read()


def _said(status: int, body: bytes) -> str:
    try:
        return json.loads(body).get("error") or f"HTTP {status}"
    except (ValueError, AttributeError):
        return f"HTTP {status}"


def _check_card_matches(head: Path, card: dict) -> None:
    artifact = card.get("artifact") or {}
    data = head.read_bytes()
    if artifact.get("bytes") != len(data):
        raise PublishError(
            f"the card says {artifact.get('bytes')} bytes, {head.name} has {len(data)}"
        )
    if artifact.get("sha256") != hashlib.sha256(data).hexdigest():
        raise PublishError(f"the card's sha256 does not match {head.name}")


def publish(
    export_id: str,
    head: Path,
    card_path: Path,
    *,
    platform_url: str,
    token: str,
    http: Http = _urllib_http,
) -> None:
    """Upload URLs, both PUTs, completion. Raises :class:`PublishError` on the first refusal."""
    card = json.loads(card_path.read_text(encoding="utf-8"))
    _check_card_matches(head, card)
    base = f"{platform_url.rstrip('/')}/agent/station-heads/{export_id}"
    auth = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    status, body = http("POST", f"{base}/upload-urls", auth, b"{}")
    if status != 200:
        raise PublishError(f"upload URLs refused: {_said(status, body)}")
    urls = json.loads(body)
    # Presigned URLs carry their own signature; the agent token never goes to R2.
    for part, data in (("head", head.read_bytes()), ("card", card_path.read_bytes())):
        target = urls[part]
        status, body = http("PUT", target["url"], {"Content-Type": target["content_type"]}, data)
        if status not in (200, 201):
            raise PublishError(f"R2 refused the {part}: HTTP {status}")

    status, body = http("POST", f"{base}/complete", auth, json.dumps({"card": card}).encode())
    if status != 200:
        raise PublishError(f"completion refused: {_said(status, body)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--export", required=True, help="the export id from the panel")
    parser.add_argument("--head", required=True, type=Path, help="the .onnx training wrote")
    parser.add_argument("--card", required=True, type=Path, help="the .card.json beside it")
    parser.add_argument("--platform-url", default=DEFAULT_PLATFORM)
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE)
    args = parser.parse_args(argv)
    try:
        publish(
            args.export,
            args.head,
            args.card,
            platform_url=args.platform_url,
            token=read_agent_token(args.state_file),
        )
    except PublishError as err:
        print(f"publish_station_head: {err}", file=sys.stderr)
        return 1
    print(f"published {args.head.name} for export {args.export}; the box registers the run next")
    return 0


if __name__ == "__main__":
    sys.exit(main())
