# /// script
# requires-python = ">=3.11"
# dependencies = ["pillow"]
# ///
"""Build a per-person activity-annotation package from a detection fixture.

Generic across cameras: give it any `magazyn-hall-v1`-shaped detection manifest
(a `frames[].persons[].bbox` fixture like the pose-resolution benchmark produces)
and it emits everything a human needs to label *what each detected person is
doing* — `sitting` / `standing` / `walking` / `running` — without a classifier at
any point.

It does three things and nothing else:

1. Assigns every person a stable `person_id` (`{frame_id}#p{index:02d}`), because
   the detection fixture identifies people only by list position.
2. Writes a scaffold activity manifest with `activity: null` per person, plus the
   native bbox height and a `posture_prior` (a *prior*, not a filter — the human
   overrides it per crop).
3. Renders one native-resolution crop per person (small people upscaled for
   viewing only) and an `index.html` that groups crops by frame, largest first,
   with the source detection `note` beside each so the labeler has context.

DELIBERATELY NOT A CLASSIFIER. It never guesses an activity; every `activity`
starts null and is filled by human review. The detection `note` is shown as an
aid, but a note is not a label — postures must be read off the crop and confirmed
by eye, exactly as `magazyn-hall-v1`'s detection boxes were.

The readability prior is a floor on what a *human* can label from a still, not
what a classifier can resolve; the latter is what the activity benchmark exists
to measure. People below the floor are scaffolded as `unresolved` so the fixture
never claims a posture it cannot support.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from PIL import Image

# Native bbox height below which a human cannot reliably read sitting-vs-standing
# from a single still. A PRIOR the reviewer overrides per crop, not a hard gate.
# On magazyn-hall-v1 this keeps ~89% of people (264/296) in scope.
POSTURE_READABLE_MIN_H = 80

# Context padding around each bbox so posture is legible (shoulders, feet, the
# chair/bench a person sits on), as a fraction of bbox height/width.
CROP_MARGIN_FRAC = 0.35

# Upscale small crops for VIEWING ONLY so the reviewer is not squinting; the
# manifest records native pixels regardless.
VIEW_MIN_H = 320


def _crop_person(frame: Image.Image, bbox: list[int]) -> Image.Image:
    x1, y1, x2, y2 = bbox
    mw = int((x2 - x1) * CROP_MARGIN_FRAC)
    mh = int((y2 - y1) * CROP_MARGIN_FRAC)
    box = (
        max(0, x1 - mw),
        max(0, y1 - mh),
        min(frame.width, x2 + mw),
        min(frame.height, y2 + mh),
    )
    crop = frame.crop(box)
    if crop.height < VIEW_MIN_H and crop.height > 0:
        scale = VIEW_MIN_H / crop.height
        crop = crop.resize((max(1, int(crop.width * scale)), VIEW_MIN_H), Image.LANCZOS)
    return crop


def build(detection_manifest: Path, frames_dir: Path, out_dir: Path) -> dict:
    manifest = json.loads(detection_manifest.read_text(encoding="utf-8"))
    fixture_id = manifest["fixture_id"]
    crops_dir = out_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    people: list[dict] = []
    cards: list[str] = []
    for frame in manifest["frames"]:
        frame_id = frame["id"]
        frame_path = frames_dir / Path(frame["path"]).name
        image = Image.open(frame_path).convert("RGB")
        frame_cards: list[tuple[int, str]] = []
        for index, person in enumerate(frame["persons"]):
            bbox = [int(v) for v in person["bbox"]]
            height = bbox[3] - bbox[1]
            person_id = f"{frame_id}#p{index:02d}"
            readable = height >= POSTURE_READABLE_MIN_H
            crop_name = f"{person_id.replace('#', '_')}.jpg"
            _crop_person(image, bbox).save(crops_dir / crop_name, quality=90)
            people.append(
                {
                    "person_id": person_id,
                    "frame_id": frame_id,
                    "window_id": frame["window_id"],
                    "bbox": bbox,
                    "bbox_height_native": height,
                    "posture_prior": "readable" if readable else "unresolved",
                    "detection_note": person.get("note", ""),
                    # Filled by human review. `activity` stays null until then.
                    "activity": None,
                    "posture_readable": None,
                    "review_status": "pending",
                }
            )
            note = html.escape(person.get("note", ""))
            frame_cards.append(
                (
                    height,
                    f'<figure><img src="crops/{crop_name}" loading="lazy">'
                    f"<figcaption><b>{person_id}</b> · {height}px"
                    f"{' · <i>unresolved prior</i>' if not readable else ''}"
                    f"<br>{note}</figcaption></figure>",
                )
            )
        # Largest people first — most reliable to label, sets the pace.
        for _, card in sorted(frame_cards, key=lambda c: -c[0]):
            cards.append(card)

    scaffold = {
        "schema_version": 1,
        "fixture_id": fixture_id,
        "task": "per-person-activity",
        "activities": ["sitting", "standing", "walking", "running"],
        "unresolved_sentinel": "unresolved",
        "posture_readable_min_h_native": POSTURE_READABLE_MIN_H,
        "derived_from": {
            "detection_fixture": fixture_id,
            "detection_manifest": str(detection_manifest),
        },
        "annotation_methodology": "METHODOLOGY.md",
        "people": people,
    }
    (out_dir / "manifest.scaffold.json").write_text(
        json.dumps(scaffold, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    (out_dir / "index.html").write_text(
        "<!doctype html><meta charset=utf-8>"
        f"<title>{html.escape(fixture_id)} — activity annotation</title>"
        "<style>body{background:#111;color:#ddd;font:14px system-ui;margin:1rem}"
        "main{display:flex;flex-wrap:wrap;gap:.75rem}"
        "figure{margin:0;max-width:220px}img{max-width:220px;border:1px solid #333}"
        "figcaption{font-size:12px;color:#aaa}</style>"
        f"<h1>{html.escape(fixture_id)} — {len(people)} people</h1>"
        "<p>Label each crop <code>sitting/standing/walking/running/unresolved</code> "
        "in <code>manifest.scaffold.json</code>. The prior is a hint, not a decision.</p>"
        "<main>" + "".join(cards) + "</main>",
        encoding="utf-8",
    )

    readable = sum(1 for p in people if p["posture_prior"] == "readable")
    return {
        "fixture_id": fixture_id,
        "people": len(people),
        "readable_prior": readable,
        "unresolved_prior": len(people) - readable,
        "out_dir": str(out_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detection-manifest", required=True, type=Path)
    parser.add_argument("--frames-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    summary = build(args.detection_manifest, args.frames_dir, args.out)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
