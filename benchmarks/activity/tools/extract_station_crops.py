#!/usr/bin/env python3
"""Cut the station rectangle out of every annotated clip, at the card's stride.

The pixel arms need `crops/<slot>-native/` and nothing else, but the only thing
that produced those directories was `run_tcn_arm.py`, which also builds a pose
session and needs a YOLO checkpoint to do it. Re-cutting crops after a rectangle
change therefore meant running the pose arm for its side effect.

This is that side effect on its own. It calls the same `extract_native_crops`
rather than reimplementing the ffmpeg line, because a second implementation of
the crop is exactly how this fixture ended up with crops cut 40 px above every
record of them.

    python benchmarks/activity/tools/extract_station_crops.py \
      --manifest benchmarks/activity/hala-prawe-v1/manifest.source.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from run_tcn_arm import extract_native_crops  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path)
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text())
    fixture_dir = args.manifest.parent
    roi = manifest["station_roi"]["crop"]
    print(f"station rectangle {roi['w']}x{roi['h']} at ({roi['x']}, {roi['y']})")

    for clip in manifest["clips"]:
        if not clip.get("annotated"):
            continue
        slot = clip["slot"]
        truth = json.loads((fixture_dir / clip["annotation_file"]).read_text())
        out_dir = fixture_dir / "crops" / f"{slot}-native"
        crops = extract_native_crops(
            fixture_dir / clip["file"], roi, int(truth["stride_s"]), out_dir
        )
        # The annotation is timestamped intervals, so it survives a rectangle
        # change untouched - but the sample count has to keep lining up with it,
        # and a short ffmpeg run would silently shorten the training set.
        labelled = int(truth["coverage"]["total"])
        flag = "" if len(crops) >= labelled else "  <-- SHORT of the annotation"
        print(f"  {slot}: {len(crops)} crops for {labelled} labelled samples{flag}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
