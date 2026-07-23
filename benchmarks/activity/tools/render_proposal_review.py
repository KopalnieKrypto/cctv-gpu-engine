# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Render a proposal-review page for the human confirmation pass.

Reads a `manifest.proposed.json` (from `propose_activity_labels.py`, with crop
reads applied) and the fixture's `crops/`, and emits `review.html`: every person
as a card showing the crop, the proposed label colour-coded, its source and
evidence, and a highlight on `contested` people. Ordered contested-first, then by
label, then largest-first, so the reviewer confirms the doubtful calls while the
scene is fresh and rubber-stamps the obvious standers last.

Nothing here decides a label; it only presents the proposals for a human to
confirm or override in `manifest.proposed.json`.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

COLOR = {
    "sitting": "#e0b000",
    "standing": "#3fa7ff",
    "walking": "#4fd06a",
    "running": "#ff6a4f",
    "unresolved": "#888",
}


def render(manifest_path: Path, out_path: Path) -> dict:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    people = data["people"]

    def sort_key(p: dict) -> tuple:
        return (
            0 if p.get("contested") else 1,
            str(p["activity_proposed"]),
            -p["bbox_height_native"],
        )

    cards = []
    for p in sorted(people, key=sort_key):
        act = str(p["activity_proposed"])
        crop = "crops/" + p["person_id"].replace("#", "_") + ".jpg"
        flag = '<span style="color:#ff5">⚑ CONFIRM</span> ' if p.get("contested") else ""
        cards.append(
            f'<figure style="border-color:{COLOR.get(act, "#555")}">'
            f'<img src="{crop}" loading="lazy">'
            f'<figcaption><b style="color:{COLOR.get(act, "#fff")}">{act}</b> '
            f"{flag}<br>{html.escape(p['person_id'])} · {p['bbox_height_native']}px"
            f"<br><span class=src>{html.escape(p['proposal_source'])}</span>: "
            f"{html.escape(p.get('proposal_evidence', ''))}</figcaption></figure>"
        )

    counts: dict[str, int] = {}
    for p in people:
        counts[str(p["activity_proposed"])] = counts.get(str(p["activity_proposed"]), 0) + 1
    contested = sum(1 for p in people if p.get("contested"))
    summary = (
        " · ".join(f"{k} {v}" for k, v in sorted(counts.items())) + f" · contested {contested}"
    )

    out_path.write_text(
        "<!doctype html><meta charset=utf-8>"
        f"<title>{html.escape(data['fixture_id'])} — activity proposals</title>"
        "<style>body{background:#111;color:#ddd;font:13px system-ui;margin:1rem}"
        "main{display:flex;flex-wrap:wrap;gap:.6rem}"
        "figure{margin:0;max-width:190px;border:2px solid #555;border-radius:6px;"
        "padding:4px;background:#191919}"
        "img{max-width:180px;display:block}figcaption{font-size:11px;color:#bbb;margin-top:3px}"
        ".src{color:#7aa}</style>"
        f"<h1>{html.escape(data['fixture_id'])} — activity proposals</h1>"
        f"<p><b>{summary}</b>. Proposals only — confirm/override in "
        "<code>manifest.proposed.json</code>, then commit as <code>manifest.json</code>. "
        "Contested (⚑) are shown first.</p>"
        "<main>" + "".join(cards) + "</main>",
        encoding="utf-8",
    )
    return {"people": len(people), "contested": contested, "counts": counts}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(render(args.manifest, args.out), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
