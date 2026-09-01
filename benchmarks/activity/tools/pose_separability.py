"""Is there any pose signal separating the C.0 activities? A probe, not an arm.

Arms 2 and 3 in #117 - temporal segmentation over pose sequences, and a
per-station classifier head - both stand on the same assumption: that YOLO-pose
sees a measurably different body on `ukladanie_pretow` than on `spawanie`. This
script tests that assumption directly and cheaply, so a dead assumption costs
half an hour instead of two weeks of arm-building.

It deliberately does the WEAKEST thing that could still show signal: one frame,
no temporal context, a linear probe. That direction of bias is what makes a
negative result cheap to trust and a positive one worth pursuing:

- **If the linear probe separates the classes**, a sequence model with the
  station's ~85 s cycle to exploit will do better, and arms 2/3 are worth
  building.
- **If it does not**, that is not yet proof they are dead - a linear probe on
  single frames is a low bar and temporal context is exactly what it lacks. It
  does mean the per-frame pose features carry nothing on their own, which is the
  cheapest possible early warning.

## Two questions, and the first one may answer more than the probe

1. **Detection rate.** Does pose find the operator in this ROI at all? An
   undetected person cannot be classified, so this caps everything downstream.
   It is also a free signal in its own right: `brak_na_stanowisku` means the
   bench is empty, so "pose found nobody" should predict it - the class the VLM
   arm scored 0.0% on.
2. **Separability.** Trained on W1's labelled frames, can a linear model recover
   the activity on W2? Per-class, on the declared fold, never an average.

## It reads the NATIVE crop

Pose runs on the full 900x800 native-pixel station crop, not the 640 px review
downscale the annotator and the VLM saw. Detection recall is resolution-bound
(#113), and the question here is whether the signal EXISTS - so the probe is
given the best pixels available. If it fails even at native resolution, the
downscale will not rescue it.

## Usage

    # inside the GPU container, on cctv-vps GPU 1
    python benchmarks/activity/tools/pose_separability.py \
      --manifest benchmarks/activity/hala-prawe-v1/manifest.source.json \
      --model models/yolo11s-pose-1280x736.onnx \
      --out benchmarks/activity/hala-prawe-v1/pose-separability.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# Held-out class rows below this many samples cannot resolve a per-class figure;
# they are reported with their support so nobody quotes a percentage over n=2.
THIN_CLASS = 20


def extract_native_crops(clip: Path, roi: dict, stride: int, out_dir: Path) -> list[Path]:
    """ROI crops at NATIVE resolution - no downscale. See the module docstring."""
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not on PATH")
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("t*.jpg"):
        stale.unlink()
    crop = f"crop={roi['w']}:{roi['h']}:{roi['x']}:{roi['y']}"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(clip),
            "-vf",
            f"fps=1/{stride},{crop}",
            "-q:v",
            "2",
            str(out_dir / "t%05d.jpg"),
        ],
        check=True,
    )
    return sorted(out_dir.glob("t*.jpg"))


def pose_features(det, crop_w: int, crop_h: int) -> np.ndarray:
    """Keypoints made translation- and scale-free, so posture is what is left.

    Raw pixel coordinates would let the probe memorise WHERE in the frame the
    operator stands rather than WHAT their body is doing, and the bench is
    static, so that shortcut would score well and mean nothing.
    """
    x1, y1, x2, y2 = det.bbox
    bw, bh = max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)
    kp = np.asarray([[k.x, k.y, k.vis] for k in det.keypoints], dtype=np.float32)
    xs = (kp[:, 0] - x1) / bw
    ys = (kp[:, 1] - y1) / bh
    conf = kp[:, 2]
    return np.concatenate(
        [
            xs,
            ys,
            conf,
            # Body shape and scale: a crouched reach and an upright stance differ
            # in aspect ratio even when the keypoints are noisy.
            [bh / max(bw, 1e-6), bh / crop_h, bw / crop_w, det.confidence],
        ]
    ).astype(np.float32)


def linear_probe(
    xtr: np.ndarray, ytr: np.ndarray, xte: np.ndarray, classes: list[str], epochs: int = 400
) -> np.ndarray:
    import torch

    mu, sd = xtr.mean(0, keepdims=True), xtr.std(0, keepdims=True) + 1e-6
    xtr_t = torch.tensor((xtr - mu) / sd)
    xte_t = torch.tensor((xte - mu) / sd)
    ytr_t = torch.tensor(ytr, dtype=torch.long)

    model = torch.nn.Linear(xtr.shape[1], len(classes))
    opt = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=1e-3)
    # Class-balanced loss: `spawanie` is 37-46% of the material, and an
    # unweighted probe would score well by predicting it everywhere - the exact
    # degenerate behaviour the VLM arm already showed.
    counts = np.bincount(ytr, minlength=len(classes)).astype(np.float32)
    weight = torch.tensor(np.where(counts > 0, counts.sum() / np.maximum(counts, 1), 0.0))
    lossf = torch.nn.CrossEntropyLoss(weight=weight)

    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(model(xtr_t), ytr_t)
        loss.backward()
        opt.step()

    with torch.no_grad():
        return model(xte_t).argmax(1).numpy()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    from pipeline.pose_detector import load_pose_model

    manifest = json.loads(args.manifest.read_text())
    fixture_dir = args.manifest.parent
    roi = manifest["station_roi"]["crop"]
    classes = [a["id"] for a in manifest["activities"]]
    split = manifest["split"]
    train_slots = split["folds"][0]["train_dev"]
    test_slots = split["folds"][0]["held_out"]

    detector = load_pose_model(str(args.model))

    import cv2

    data: dict[str, dict] = {}
    for clip_meta in manifest["clips"]:
        if not clip_meta.get("annotated"):
            continue
        slot = clip_meta["slot"]
        truth = json.loads((fixture_dir / clip_meta["annotation_file"]).read_text())
        stride = int(truth["stride_s"])
        clip = next(fixture_dir.glob(f"{slot}-*.mp4"), None)
        if clip is None:
            sys.exit(f"clip for {slot} not found")

        print(f"[{slot}] extracting native ROI crops ...", file=sys.stderr)
        crops = extract_native_crops(clip, roi, stride, fixture_dir / "crops" / f"{slot}-native")
        labels = [s["activity_id"] for s in truth["samples"]]
        print(f"[{slot}] {len(crops)} crops, {len(labels)} labels", file=sys.stderr)

        feats, ys, found = [], [], []
        for i, path in enumerate(crops[: len(labels)]):
            img = cv2.imread(str(path))
            dets = detector.detect(img)
            # The unit is the station, so the most confident person at the bench
            # is the one the label describes (assumption A5 - never an identity).
            best = max(dets, key=lambda d: d.confidence) if dets else None
            found.append(best is not None)
            if best is not None:
                feats.append(pose_features(best, img.shape[1], img.shape[0]))
                ys.append(classes.index(labels[i]))
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(crops)}", file=sys.stderr)

        data[slot] = {
            "labels": labels[: len(crops)],
            "found": found,
            "x": np.stack(feats) if feats else np.zeros((0, 1), np.float32),
            "y": np.asarray(ys, dtype=np.int64),
        }

    # --- question 1: detection rate, per activity ---
    detection: dict[str, dict[str, dict]] = {}
    for slot, d in data.items():
        per: dict[str, dict] = {}
        for c in classes:
            idx = [i for i, lab in enumerate(d["labels"]) if lab == c]
            if not idx:
                continue
            hits = sum(1 for i in idx if d["found"][i])
            per[c] = {"support": len(idx), "detected": hits, "rate": hits / len(idx)}
        detection[slot] = per

    # --- question 2: separability, on the declared fold ---
    xtr = np.concatenate([data[s]["x"] for s in train_slots if s in data])
    ytr = np.concatenate([data[s]["y"] for s in train_slots if s in data])
    xte = np.concatenate([data[s]["x"] for s in test_slots if s in data])
    yte = np.concatenate([data[s]["y"] for s in test_slots if s in data])
    pred = linear_probe(xtr, ytr, xte, classes)

    per_class: dict[str, dict] = {}
    for ci, c in enumerate(classes):
        support = int((yte == ci).sum())
        if support == 0:
            continue
        tp = int(((yte == ci) & (pred == ci)).sum())
        predicted = int((pred == ci).sum())
        per_class[c] = {
            "support": support,
            "recall": tp / support,
            "precision": (tp / predicted) if predicted else None,
            "time_ratio": predicted / support,
            "thin": support < THIN_CLASS,
        }

    # Per-sample predictions on the held-out window, so the result can be read as
    # a timeline against the video rather than only as aggregates. Frames where
    # pose found nobody are emitted as `__brak_detekcji__` rather than dropped:
    # that is a prediction ("the bench is empty"), and on W1 it is right 65 times
    # out of 66, so hiding it would understate what the detector already does.
    timeline: dict[str, list[dict]] = {}
    cursor = 0  # `pred` is the test slots' detected frames, concatenated in order
    for slot in test_slots:
        if slot not in data:
            continue
        d = data[slot]
        rows = []
        for i, truth_label in enumerate(d["labels"]):
            if d["found"][i]:
                pred_label = classes[int(pred[cursor])] if cursor < len(pred) else None
                cursor += 1
            else:
                pred_label = "__brak_detekcji__"
            rows.append(
                {
                    "t_s": i * 2,
                    "truth": truth_label,
                    "pred": pred_label,
                    "hit": pred_label == truth_label
                    or (pred_label == "__brak_detekcji__" and truth_label == "brak_na_stanowisku"),
                }
            )
        timeline[slot] = rows

    report = {
        "probe": "single-frame linear probe on YOLO-pose keypoints",
        "model": args.model.name,
        "train": train_slots,
        "test": test_slots,
        "detection_rate": detection,
        "separability": per_class,
        "timeline": timeline,
        "note": (
            "Deliberately the weakest test that could still show signal: one frame, no "
            "temporal context, a linear model. A positive result means arms 2/3 are "
            "worth building. A negative one means per-frame pose features carry nothing "
            "on their own - it does NOT by itself kill a sequence model, which has the "
            "station's ~85 s cycle that this probe cannot see."
        ),
    }
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print("\n=== detection rate (is the operator found at all?) ===")
    for slot, per in detection.items():
        print(f"[{slot}]")
        for c, s in sorted(per.items(), key=lambda kv: -kv[1]["support"]):
            print(f"   {c:22s} {s['detected']:4d}/{s['support']:<4d}  {100 * s['rate']:5.1f}%")
    print(f"\n=== separability: train {train_slots} -> test {test_slots} ===")
    print(f"   {'activity':22s} {'support':>7s} {'recall':>8s} {'precision':>10s} {'time':>7s}")
    for c, s in sorted(per_class.items(), key=lambda kv: -kv[1]["support"]):
        prec = "n/a" if s["precision"] is None else f"{100 * s['precision']:.1f}%"
        thin = "  (thin)" if s["thin"] else ""
        print(
            f"   {c:22s} {s['support']:7d} {100 * s['recall']:7.1f}% "
            f"{prec:>10s} {s['time_ratio']:6.2f}x{thin}"
        )
    print(f"\nwrote: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
