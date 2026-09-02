#!/usr/bin/env python3
"""Export the frozen DINOv2 backbone to ONNX for `setup-models.sh` (#122).

## Why this is pinned like every other model here

The backbone is the half of the station classifier that never changes. It ships
once inside the container image and is identical at every station, so it belongs
in `setup-models.sh` next to YOLO and OSNet, fetched from a versioned release and
sha256-verified - for the reason that script already states: so every checkout
lands the *same* weights. Resolving `facebook/dinov2-base` through `transformers`
at build time would instead land whatever HuggingFace serves that day, and a
silently different backbone changes every embedding the head was fitted on.

## The export is CLS-only on purpose

The pipeline reads exactly one thing from this model: the CLS token, which is the
pooled representation DINOv2 is trained to make linearly useful. Exporting the
full `last_hidden_state` would ship 1370 patch tokens per frame that nothing
consumes. The wrapper below returns the CLS vector alone, so the artefact is the
function the system actually calls.

## Usage

    # inside the GPU container - CPU export works too, it is just slower
    python benchmarks/activity/tools/export_backbone_onnx.py \
      --out models/dinov2-base-518.onnx --image-size 518
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from run_pixel_probe_arm import BACKBONE  # noqa: E402

OPSET = 18


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--image-size", type=int, default=518)
    ap.add_argument("--backbone", default=BACKBONE)
    args = ap.parse_args()

    import torch
    from transformers import AutoModel

    class ClsOnly(torch.nn.Module):
        """DINOv2 reduced to the one output the pipeline reads."""

        def __init__(self, backbone: str):
            super().__init__()
            self.model = AutoModel.from_pretrained(backbone)

        def forward(self, pixel_values):
            return self.model(pixel_values=pixel_values).last_hidden_state[:, 0]

    model = ClsOnly(args.backbone).eval()
    dummy = torch.zeros(1, 3, args.image_size, args.image_size)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(args.out),
        input_names=["pixel_values"],
        output_names=["cls"],
        # Batch varies: the embedder sends a batch of station crops per pass.
        # The spatial dims are fixed - DINOv2's position embeddings are
        # interpolated for a given size, and the head was fitted at this one.
        dynamic_axes={"pixel_values": {0: "batch"}, "cls": {0: "batch"}},
        opset_version=OPSET,
    )
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    size = args.out.stat().st_size
    print(f"wrote {args.out} ({size / 1024 / 1024:.0f} MiB)")
    print(f"sha256 {digest}")
    print(f'\nPin it in setup-models.sh:\n  DINOV2_SHA256="{digest}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
