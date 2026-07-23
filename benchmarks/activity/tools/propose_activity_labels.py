# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Propose per-person activity labels FROM THE HUMAN DETECTION NOTES, for review.

This is an annotation *aid*, not ground truth. It structures the human-authored
`note` fields carried by the #99 detection fixture — Tomasz's own prose, which
frequently states posture ("Full standing figure visible", "standing beside the
machine", "seated at the bench") — into a proposed `sitting/standing/walking/
running` label, so the human confirmation pass starts from a draft instead of a
blank.

It never runs a classifier. The signal is the detection annotator's words. Where
the note does not disambiguate posture, the person is flagged `needs_crop_read`
rather than guessed, so a human (or an agent reading the crop, with human
sign-off — the practice #99 records) decides. People below the readability floor
inherit `unresolved` from the size prior.

Provenance is recorded per proposal (`proposal_source`, `proposal_evidence`) and
`activity` stays null and `review_status: pending`: nothing here is confirmed
ground truth. Output is `manifest.proposed.json`, never `manifest.json`.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# Ordered posture cues over the human note text; first bucket to match wins.
# Deliberately conservative: ambiguous work postures (bending/leaning) are NOT
# forced into sitting-or-standing — they are the exact judgment a human should
# make against the crop, so they route to needs_crop_read.
CUES: list[tuple[str, list[str]]] = [
    # `running` is intentionally NOT auto-detected: on magazyn-hall-v1 both note
    # matches were spurious (stationary welders), and a real runner in a work
    # hall is rare enough to be worth a crop read rather than a keyword.
    (
        "walking",
        [
            r"\bwalking\b",
            r"\bwalks\b",
            r"\bmid-?stride\b",
            r"\bmid-?step\b",
            r"\bstepping\b",
            r"\bstriding\b",
            r"\bin motion\b",
        ],
    ),
    (
        # NOTE: `\bsits\b` is deliberately excluded. Detection notes use "edge
        # sits on the tile boundary" for BOX geometry — all 7 such matches on
        # magazyn were standing workers. Only unambiguous posture words qualify.
        "sitting",
        [
            r"\bsitting\b",
            r"\bseated\b",
            r"\bon a (chair|stool)\b",
            r"\bcrouch",
            r"\bsquat",
            r"\bkneel",
        ],
    ),
    (
        "standing",
        [r"\bstanding\b", r"\bstands\b", r"\bstood\b", r"\bupright\b", r"\bstanding figure\b"],
    ),
]

# Cues that mean "do not trust a keyword — a human must look": ambiguous posture.
DEFER = [r"\bbend", r"\bbent\b", r"\bleaning\b", r"\bhunched\b", r"\bcrouched over\b"]


def _propose(note: str) -> tuple[str | None, str, str]:
    text = note.lower()
    for pattern in DEFER:
        if re.search(pattern, text):
            return None, "needs_crop_read", f"ambiguous posture cue: /{pattern}/"
    for activity, patterns in CUES:
        for pattern in patterns:
            if re.search(pattern, text):
                return activity, "note", f"/{pattern}/"
    return None, "needs_crop_read", "no posture cue in note"


def build(scaffold_path: Path, out_path: Path) -> dict:
    scaffold = json.loads(scaffold_path.read_text(encoding="utf-8"))
    stats: dict[str, int] = {}
    src_stats: dict[str, int] = {}
    for person in scaffold["people"]:
        if person["posture_prior"] == "unresolved":
            proposed, source, evidence = "unresolved", "size_prior", "bbox < floor"
        else:
            proposed, source, evidence = _propose(person["detection_note"])
        person["activity_proposed"] = proposed
        person["proposal_source"] = source
        person["proposal_evidence"] = evidence
        # Ground truth is untouched: still null, still pending.
        person["activity"] = None
        person["review_status"] = "pending"
        stats[str(proposed)] = stats.get(str(proposed), 0) + 1
        src_stats[source] = src_stats.get(source, 0) + 1
    out_path.write_text(json.dumps(scaffold, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"proposed_label_counts": stats, "proposal_source_counts": src_stats}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scaffold", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.scaffold, args.out), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
