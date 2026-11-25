"""Publishing a trained head from cctv-vps (gpu-exchange#314).

What these tests encode:

* **Input**: the export id, the head and card ``train_station_head.py`` wrote,
  the agent's state file and an injected HTTP call.
* **Output**: the platform's upload URLs asked for with the agent token, both
  files PUT to them, then completion reported with the card.
* **Boundaries**: a card that does not describe the head (checked before any
  request), the platform refusing (no upload follows), a state file without a
  token.
* **Not tested here**: the platform's own checks (gpu-exchange data-service)
  and R2.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import publish_station_head as pub  # noqa: E402

EXPORT = "22222222-2222-4222-8222-222222222222"
APPLIANCE = "33333333-3333-4333-8333-333333333333"
PLATFORM = "https://api-panel.example"
#: What `train_station_head.py --package` records: the package's head name and directory.
FROM_THE_DROP = {
    "bound_to": "hala/v1.0.0",
    "dir": f"/srv/annotation-drop/{APPLIANCE}/{EXPORT}",
}


def a_head(
    tmp_path: Path, payload: bytes = b"onnx-bytes", package: dict | None = FROM_THE_DROP
) -> tuple[Path, Path, dict]:
    head = tmp_path / "station-head-hala-v1.0.0.onnx"
    head.write_bytes(payload)
    card = {
        "artifact": {
            "version": "1.0.0",
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        },
        "station": {"id": "hala"},
        "accuracy": {"union": {"spawanie": {"recall": 0.9, "time_ratio": 1.0, "support": 10}}},
    }
    if package is not None:
        card["training_material"] = {"package": package}
    card_path = tmp_path / "station-head-hala-v1.0.0.card.json"
    card_path.write_text(json.dumps(card), encoding="utf-8")
    return head, card_path, card


class FakeHttp:
    """The platform and R2 as far as the script sees them: canned answers, every call kept."""

    def __init__(self, answers: dict[str, tuple[int, dict]] | None = None) -> None:
        self.calls: list[tuple[str, str, dict, bytes | None]] = []
        self.answers = answers or {
            f"{PLATFORM}/agent/station-heads/{EXPORT}/upload-urls": (
                200,
                {
                    "head": {
                        "url": "https://r2.example/head?sig",
                        "content_type": "application/octet-stream",
                    },
                    "card": {
                        "url": "https://r2.example/card?sig",
                        "content_type": "application/json",
                    },
                    "bound_to": "hala/v1.0.0",
                    "appliance_id": APPLIANCE,
                },
            ),
            f"{PLATFORM}/agent/station-heads/{EXPORT}/complete": (200, {"ok": True}),
        }

    def __call__(
        self, method: str, url: str, headers: dict, body: bytes | None
    ) -> tuple[int, bytes]:
        self.calls.append((method, url, headers, body))
        status, answer = self.answers.get(url, (200, {}))
        return status, json.dumps(answer).encode()


def test_publishes_the_head_and_card_and_reports_done_with_the_card(tmp_path):
    head, card_path, card = a_head(tmp_path)
    http = FakeHttp()

    pub.publish(EXPORT, head, card_path, platform_url=PLATFORM, token="agt_secret", http=http)

    [(m1, u1, h1, _), (m2, u2, h2, b2), (m3, u3, h3, b3), (m4, u4, h4, b4)] = http.calls
    assert (m1, u1) == ("POST", f"{PLATFORM}/agent/station-heads/{EXPORT}/upload-urls")
    assert h1["Authorization"] == "Bearer agt_secret"
    assert (m2, u2, h2["Content-Type"], b2) == (
        "PUT",
        "https://r2.example/head?sig",
        "application/octet-stream",
        b"onnx-bytes",
    )
    assert (m3, u3, h3["Content-Type"]) == (
        "PUT",
        "https://r2.example/card?sig",
        "application/json",
    )
    assert json.loads(b3) == card
    assert (m4, u4, h4["Authorization"]) == (
        "POST",
        f"{PLATFORM}/agent/station-heads/{EXPORT}/complete",
        "Bearer agt_secret",
    )
    assert json.loads(b4) == {"card": card}
    # The token goes to the platform only, never to R2.
    assert "Authorization" not in h2 and "Authorization" not in h3


def test_refuses_a_card_that_describes_another_file_before_any_request(tmp_path):
    head, card_path, _ = a_head(tmp_path)
    head.write_bytes(b"onnx-bytez")  # same size, other bytes: only the digest can tell
    http = FakeHttp()

    with pytest.raises(pub.PublishError, match="sha256"):
        pub.publish(EXPORT, head, card_path, platform_url=PLATFORM, token="t", http=http)
    assert http.calls == []


def test_stops_on_the_platforms_refusal_in_its_words_without_uploading(tmp_path):
    head, card_path, _ = a_head(tmp_path)
    url = f"{PLATFORM}/agent/station-heads/{EXPORT}/upload-urls"
    http = FakeHttp(
        {url: (409, {"error": "This export's head is already published", "code": "HEAD_CLOSED"})}
    )

    with pytest.raises(pub.PublishError, match="already published"):
        pub.publish(EXPORT, head, card_path, platform_url=PLATFORM, token="t", http=http)
    assert [c[0] for c in http.calls] == ["POST"]


def test_reads_the_agent_token_from_its_state_file_and_refuses_a_file_without_one(tmp_path):
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"agentToken": "agt_1", "serverId": "s"}), encoding="utf-8")
    assert pub.read_agent_token(state) == "agt_1"

    state.write_text(json.dumps({"serverId": "s"}), encoding="utf-8")
    with pytest.raises(pub.PublishError, match="agentToken"):
        pub.read_agent_token(state)


def test_publishes_a_head_trained_on_this_exports_package(tmp_path):
    head, card_path, _ = a_head(tmp_path, package=FROM_THE_DROP)
    http = FakeHttp()

    pub.publish(EXPORT, head, card_path, platform_url=PLATFORM, token="t", http=http)

    assert [c[0] for c in http.calls] == ["POST", "PUT", "PUT", "POST"]


@pytest.mark.parametrize(
    ("package", "said"),
    [
        # A typo in --export: another project's head under this export's key, maybe another tenant.
        ({**FROM_THE_DROP, "bound_to": "magazyn/v2.0.0"}, "bound to magazyn/v2.0.0"),
        ({**FROM_THE_DROP, "dir": f"/srv/annotation-drop/{APPLIANCE}/other"}, "not this export"),
        ({**FROM_THE_DROP, "dir": f"/tmp/{EXPORT}"}, "not this export"),
    ],
)
def test_refuses_a_head_trained_on_another_package_before_uploading(tmp_path, package, said):
    head, card_path, _ = a_head(tmp_path, package=package)
    http = FakeHttp()

    with pytest.raises(pub.PublishError, match=said):
        pub.publish(EXPORT, head, card_path, platform_url=PLATFORM, token="t", http=http)
    assert [c[0] for c in http.calls] == ["POST"]


def test_refuses_when_the_platform_does_not_say_which_package_the_export_is(tmp_path):
    head, card_path, _ = a_head(tmp_path, package=FROM_THE_DROP)
    http = FakeHttp()
    url = f"{PLATFORM}/agent/station-heads/{EXPORT}/upload-urls"
    status, urls = http.answers[url]
    http.answers[url] = (status, {k: v for k, v in urls.items() if k != "appliance_id"})

    with pytest.raises(pub.PublishError, match="appliance_id"):
        pub.publish(EXPORT, head, card_path, platform_url=PLATFORM, token="t", http=http)
    assert [c[0] for c in http.calls] == ["POST"]


def test_refuses_a_card_trained_on_no_package_before_any_request(tmp_path):
    """The platform refuses it at completion too; here it costs no upload and no open version."""
    head, card_path, _ = a_head(tmp_path, package=None)
    http = FakeHttp()

    with pytest.raises(pub.PublishError, match="training_material.package"):
        pub.publish(EXPORT, head, card_path, platform_url=PLATFORM, token="t", http=http)
    assert http.calls == []
