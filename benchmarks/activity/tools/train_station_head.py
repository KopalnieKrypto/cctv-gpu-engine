#!/usr/bin/env python3
"""Train and export the one station head production loads, with its card (#122).

## What ships, and what does not

The frozen backbone does the looking. It is identical at every station, it ships
once inside the container image, and it is not touched here. What this script
produces is the ~1 MB temporal head that is specific to `hala prawe skrzydlo`.
That separation is the economics of the offer: onboarding a station is "annotate
twenty minutes, train a head", with no new large model and no engine redeploy.

## The shipped weights cannot be scored, and the card says so

Unlike the cross-validated arms, this model trains on **every annotated window** -
which is what you want from twenty minutes of annotation, and which means it has
no held-out material at all. The accuracy quoted for it therefore comes from the
folds, on models that are not this one.

That gap is where a number drifts loose from its measurement, and this project
has already lost one there: 98.5% for `brak_na_stanowisku` reached a client
report after being copied by hand out of a detector docstring. So the card is
generated, never typed, and `station_head.build_card` refuses to emit one with a
blank where a measurement belongs.

## Which vocabulary is trained

#122 requires the delivered vocabulary trained directly to be measured against
#121's collapsed figures rather than assumed better, with those figures as the
floor. `--direct-report` supplies the directly-trained arm's numbers,
`--report` the collapsed ones, and `station_head.choose_vocabulary` decides. The
losing option is not the one that ships, and `build_card` refuses to write a card
that disagrees with its own comparison.

## The artefact is self-contained

The head consumes standardised embeddings, so the standardisation is baked into
the exported graph as a leading layer rather than shipped beside it in a JSON
nobody reads. One file, one input, no way to pair the weights with the wrong
normalisation.

## Usage

    # inside the GPU container, on cctv-vps GPU 1
    python benchmarks/activity/tools/train_station_head.py \
      --manifest benchmarks/activity/hala-prawe-v1/manifest.source.json \
      --crops-root benchmarks/activity/hala-prawe-v1/crops \
      --cache-dir runs/tcn-pixel/cache \
      --report benchmarks/activity/hala-prawe-v1/C0-report.json \
      --direct-report benchmarks/activity/hala-prawe-v1/C0-delivered-report.json \
      --arm tcn-pixel-518 --direct-arm tcn-pixel-518-delivered \
      --version 1.0.0 --out-dir models

## Correcting a card without retraining

The training is NOT reproducible - `torch.manual_seed` does not cover cuDNN's
convolution atomics, and a re-run produces a different head (measured: 20 of 25
tensors differ, up to ~15% relative). So a card that has to be corrected for
weights which already shipped is regenerated against the file on disk instead,
which needs no GPU. The digest is required, so a card can never be attached to
weights nobody checked:

    python benchmarks/activity/tools/train_station_head.py \
      ... same arguments ... \
      --card-only 38e678de782d887d59b50c2992820d0a862fcf7b528f1de955c1e439877e2c0e
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from evaluate_arms import collapse_classes, resolve_collapse  # noqa: E402
from run_pixel_probe_arm import BACKBONE, embed_windows  # noqa: E402
from run_tcn_arm import (  # noqa: E402
    CHANNELS,
    DILATIONS,
    EPOCHS,
    KERNEL,
    LR,
    SEED,
    WEIGHT_DECAY,
    WINDOW,
    fit_model,
)
from station_head import (  # noqa: E402
    artifact_record,
    assert_single_file,
    build_card,
    choose_vocabulary,
    remap_labels,
)

IMAGE_SIZE = 518
# DINOv2-base CLS width. Named here so the card-only path can state it without
# an embedding pass; the digest check is what ties it to the actual artefact.
FEATURE_WIDTH = 768
OPSET = 18


def _scores(report: dict, arm: str, label: str) -> dict:
    entry = next((a for a in report["arms"] if a["name"] == arm), None)
    if entry is None or not entry.get("collapsed"):
        sys.exit(
            f"{label}: no arm `{arm}` with a delivery-vocabulary section. Score it "
            "with evaluate_arms.py against a manifest that declares "
            "`delivery_vocabulary` before shipping anything that quotes it."
        )
    return entry["collapsed"]["scores"]


def build_exportable(model, mu, sd):
    """Wrap the fitted body so the graph normalises its own input.

    The head was fitted on standardised features. Shipping the weights without
    the standardisation, or beside it, invites exactly one bug - and it is a
    silent one, because wrong normalisation still produces confident logits.
    """
    import torch

    class StationHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("mu", torch.tensor(mu.reshape(1, -1, 1), dtype=torch.float32))
            self.register_buffer("sd", torch.tensor(sd.reshape(1, -1, 1), dtype=torch.float32))
            self.body = model

        def forward(self, embeddings):  # (batch, feature, time)
            return self.body((embeddings - self.mu) / self.sd)

    return StationHead().eval()


def _preprocessing_contract(image_size: int) -> dict:
    """Discovered from the processor, never assumed from the flag.

    `--image-size 518` is the *resize* target and DINOv2's processor then
    centre-crops to 224, so the tensor the model receives is 224. The card has to
    carry both numbers or a reader will feed it the size it never sees.
    """
    from PIL import Image
    from transformers import AutoImageProcessor

    probe = AutoImageProcessor.from_pretrained(BACKBONE)(
        images=[Image.new("RGB", (900, 800))],
        return_tensors="pt",
        size={"height": image_size, "width": image_size},
    )["pixel_values"]
    return {"resize": image_size, "model_input": [int(probe.shape[-2]), int(probe.shape[-1])]}


def _card_only(args, manifest: dict) -> int:
    """Rewrite the card for a head that already exists, training nothing (#123).

    ## Why this mode exists

    The trainer is **not reproducible**. `torch.manual_seed` fixes the
    initialisation and the batch order, but not cuDNN's convolution algorithms,
    whose atomics are non-deterministic unless `torch.use_deterministic_algorithms`
    is set. Measured on cctv-vps 2026-09-02: re-running this script on identical
    cached embeddings produced a head differing from the released one in 20 of 25
    tensors, by up to ~15% relative.

    So when a card has to be *corrected* for weights that have already shipped —
    as in #123, where the recorded zone rectangle was 40 px off the one every crop
    was actually cut at — training again is not the fix. It would describe a
    different model, and one whose equivalence on unseen footage cannot be shown,
    since these weights have no held-out material by construction.

    Every figure still comes from `C0-report.json` and the manifest; only the
    artefact's identity comes from the file, and the caller has to name its digest.
    Nothing here is typed.

    ## What it cannot check

    It takes the vocabulary and hyperparameters from this module's constants,
    i.e. from the code that produced the artefact at this version. The digest
    check is what ties the two together: if the file is not the one those
    constants built, the run stops.
    """
    collapse = resolve_collapse(manifest, None)
    if collapse is None:
        sys.exit(f"{args.manifest} declares no `delivery_vocabulary` - nothing to deliver")

    report = json.loads(args.report.read_text())
    direct_report = json.loads(args.direct_report.read_text())
    comparison = choose_vocabulary(
        _scores(direct_report, args.direct_arm, str(args.direct_report)),
        _scores(report, args.arm, str(args.report)),
    )
    classes = [a["id"] for a in manifest["activities"]]
    output_classes = (
        collapse_classes(classes, collapse) if comparison["ships"] == "direct" else classes
    )

    name = f"station-head-{manifest['fixture']}-v{args.version}.onnx"
    onnx_path = args.out_dir / name
    n_feat = FEATURE_WIDTH
    training = {
        "backbone": BACKBONE,
        "preprocessing": _preprocessing_contract(args.image_size),
        "output_classes": output_classes,
        "trained_as": comparison["ships"],
        "comparison": comparison,
        "hyperparameters": {
            "window": WINDOW,
            "channels": CHANNELS,
            "dilations": list(DILATIONS),
            "kernel": KERNEL,
            "epochs": EPOCHS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "seed": SEED,
            "receptive_field_frames": 1 + (KERNEL - 1) * sum(DILATIONS),
            "feature_width": n_feat,
            "normalisation": "baked into the exported graph as a leading layer",
        },
        "artifact": artifact_record(
            onnx_path,
            version=args.version,
            expect_sha256=args.card_only,
            feature_width=n_feat,
            n_class=len(output_classes),
            opset=OPSET,
            backbone=BACKBONE,
        ),
    }
    card = build_card(manifest, report, args.arm, training)
    card_path = args.out_dir / f"station-head-{manifest['fixture']}-v{args.version}.card.json"
    card_path.write_text(json.dumps(card, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {card_path} for {name} (sha256 {args.card_only[:16]}…, weights untouched)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--crops-root", required=True, type=Path)
    ap.add_argument("--cache-dir", type=Path)
    ap.add_argument("--report", required=True, type=Path, help="C0-report.json (collapsed floor)")
    ap.add_argument(
        "--direct-report",
        required=True,
        type=Path,
        help="the delivered-vocabulary arm's report, for the comparison #122 requires",
    )
    ap.add_argument("--arm", default="tcn-pixel-518")
    ap.add_argument("--direct-arm", default="tcn-pixel-518-delivered")
    ap.add_argument("--version", required=True)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    ap.add_argument(
        "--card-only",
        metavar="SHA256",
        help=(
            "Regenerate the card for the head ALREADY at --out-dir, whose sha256 "
            "must equal this value, without training anything. For correcting a "
            "card that describes weights which have already shipped: the trainer "
            "is not reproducible (see below), so re-running it would describe a "
            "different model. Needs no GPU."
        ),
    )
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text())
    if args.card_only:
        return _card_only(args, manifest)

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        sys.exit("no CUDA device - the backbone pass needs a fleet GPU")

    collapse = resolve_collapse(manifest, None)
    if collapse is None:
        sys.exit(f"{args.manifest} declares no `delivery_vocabulary` - nothing to deliver")

    report = json.loads(args.report.read_text())
    direct_report = json.loads(args.direct_report.read_text())
    comparison = choose_vocabulary(
        _scores(direct_report, args.direct_arm, str(args.direct_report)),
        _scores(report, args.arm, str(args.report)),
    )
    print(f"vocabulary comparison → ships `{comparison['ships']}`", file=sys.stderr)
    for c, why in comparison["regressions"].items():
        print(f"  regression: {c} on {', '.join(why)}", file=sys.stderr)

    data = embed_windows(
        args.manifest,
        args.crops_root,
        args.cache_dir or (args.out_dir / "cache"),
        args.image_size,
        device,
    )
    classes, windows = data["classes"], data["windows"]

    # Both paths train on every annotated window. They differ only in the label
    # space, exactly as the cross-validated comparison did.
    if comparison["ships"] == "direct":
        for w in windows.values():
            w["y"], output_classes = remap_labels(w["y"], classes, collapse)
    else:
        output_classes = classes

    train_clips = [(w["x"], w["y"]) for _, w in sorted(windows.items())]
    print(
        f"training on {len(train_clips)} windows, "
        f"{sum(len(y) for _, y in train_clips)} samples, {len(output_classes)} classes",
        file=sys.stderr,
    )
    model, mu, sd = fit_model(train_clips, len(output_classes), device)

    exportable = build_exportable(model, mu, sd).to(device)
    n_feat = train_clips[0][0].shape[1]
    dummy = torch.zeros(1, n_feat, WINDOW, device=device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    name = f"station-head-{manifest['fixture']}-v{args.version}.onnx"
    onnx_path = args.out_dir / name
    torch.onnx.export(
        exportable,
        dummy,
        str(onnx_path),
        input_names=["embeddings"],
        output_names=["logits"],
        # Batch and time both vary: a clip is longer than the training window and
        # is scored with overlapping segments.
        dynamic_axes={"embeddings": {0: "batch", 2: "time"}, "logits": {0: "batch", 2: "time"}},
        opset_version=OPSET,
        # One file: setup-models.sh verifies one sha256 per model.
        external_data=False,
    )
    assert_single_file(onnx_path)
    digest = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
    size = onnx_path.stat().st_size
    print(f"wrote {onnx_path} ({size / 1024:.0f} KiB, sha256 {digest[:16]}…)")

    # Discovered, not assumed: the processor centre-crops, so `--image-size 518`
    # yields a 224 tensor. The card must carry both numbers or a reader will feed
    # the model the resize target it never sees.
    from PIL import Image
    from transformers import AutoImageProcessor

    probe = AutoImageProcessor.from_pretrained(BACKBONE)(
        images=[Image.new("RGB", (900, 800))],
        return_tensors="pt",
        size={"height": args.image_size, "width": args.image_size},
    )["pixel_values"]

    training = {
        "backbone": BACKBONE,
        "preprocessing": {
            "resize": args.image_size,
            "model_input": [int(probe.shape[-2]), int(probe.shape[-1])],
        },
        "output_classes": output_classes,
        "trained_as": comparison["ships"],
        "comparison": comparison,
        "hyperparameters": {
            "window": WINDOW,
            "channels": CHANNELS,
            "dilations": list(DILATIONS),
            "kernel": KERNEL,
            "epochs": EPOCHS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "seed": SEED,
            "receptive_field_frames": 1 + (KERNEL - 1) * sum(DILATIONS),
            "feature_width": int(n_feat),
            "normalisation": "baked into the exported graph as a leading layer",
        },
        "artifact": {
            "name": name,
            "version": args.version,
            "sha256": digest,
            "bytes": size,
            "opset": OPSET,
            "input": f"embeddings (batch, {n_feat}, time) - frozen {BACKBONE} CLS vectors",
            "output": f"logits (batch, {len(output_classes)}, time)",
        },
    }
    card = build_card(manifest, report, args.arm, training)
    card_path = args.out_dir / f"station-head-{manifest['fixture']}-v{args.version}.card.json"
    card_path.write_text(json.dumps(card, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {card_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
