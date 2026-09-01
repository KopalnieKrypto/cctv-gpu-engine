"""C.0 arm 2: temporal segmentation over pose sequences.

The pose probe (`pose_separability.py`) established that per-frame pose carries
real signal - 73.0% on `ukladanie_pretow` where the VLM arm managed 0.6% - and
that its errors are concentrated exactly where a single frame cannot help. Inside
work blocks longer than 20 s it held 60-93%; across the 66 blocks shorter than
20 s it fell to 43%. Transitions are what it gets wrong.

This arm gives the model the one thing the probe lacked: time. The station runs a
regular ~85 s production cycle on both windows, and a per-frame classifier throws
that away by construction.

## What it is

A small dilated temporal CNN over per-frame pose features. Receptive field is 29
frames at a 2 s stride, so each prediction sees ~58 s of context - most of one
production cycle, deliberately not more, because a model that can see two full
cycles could learn this station's schedule rather than its activities and would
not transfer to another bench.

## Absence is a feature, not a dropped row

The probe scored only frames where pose found somebody, which is why it could not
predict `brak_na_stanowisku` at all. Here every frame is kept: undetected frames
get a zero feature vector and a `detected` flag of 0. On W1 pose finds nobody in
65 of 66 empty-bench frames, so that flag alone is nearly sufficient for the
class, and the sequence model can additionally learn that a one-frame dropout
mid-weld is noise rather than the operator leaving.

## Honesty rules this script enforces

- **Both folds, from the manifest.** Fold A trains W1 and predicts W2; fold B the
  reverse. Per-activity accuracy is then reported by `evaluate_arms.py` on the
  union, where every labelled sample is predicted once by a model that never
  trained on it.
- **Hyperparameters are fixed before the run** and are not adjusted against
  results. They are recorded in the output. If this arm fails, it is reported as
  failing - tuning against W2 until it passes would spend the last clean window
  in the fixture, and there is no second one.
- **The pose pass is cached** so re-running the model does not re-run detection,
  but the cache is keyed by the model file so a detector change invalidates it.

## Usage

    # inside the GPU container, on cctv-vps GPU 1
    python benchmarks/activity/tools/run_tcn_arm.py \
      --manifest benchmarks/activity/hala-prawe-v1/manifest.source.json \
      --pose-model models/yolo11s-pose-1280x736.onnx \
      --out-dir runs/tcn --box cctv-vps --gpu-index 1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# --- fixed before the run, not tuned against results ---
WINDOW = 64  # frames per training window (128 s at a 2 s stride)
CHANNELS = 96
DILATIONS = (1, 2, 4)
KERNEL = 5
EPOCHS = 60
LR = 2e-3
WEIGHT_DECAY = 1e-4
SEED = 117


def extract_native_crops(clip: Path, roi: dict, stride: int, out_dir: Path) -> list[Path]:
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not on PATH")
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("t*.jpg"))
    if existing:
        return existing
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
    """Same translation- and scale-free encoding the probe used, so the two
    results differ by the model and not by the representation."""
    x1, y1, x2, y2 = det.bbox
    bw, bh = max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)
    kp = np.asarray([[k.x, k.y, k.vis] for k in det.keypoints], dtype=np.float32)
    xs = (kp[:, 0] - x1) / bw
    ys = (kp[:, 1] - y1) / bh
    return np.concatenate(
        [xs, ys, kp[:, 2], [bh / max(bw, 1e-6), bh / crop_h, bw / crop_w, det.confidence]]
    ).astype(np.float32)


def build_sequences(manifest: Path, pose_model: Path, cache: Path) -> dict:
    """Per-frame features for every annotated window, cached by detector identity."""
    m = json.loads(manifest.read_text())
    fixture_dir = manifest.parent
    roi = m["station_roi"]["crop"]
    classes = [a["id"] for a in m["activities"]]

    key = hashlib.sha256((pose_model.name + str(pose_model.stat().st_size)).encode()).hexdigest()[
        :12
    ]
    cache_file = cache / f"pose-seq-{key}.npz"
    meta_file = cache / f"pose-seq-{key}.json"
    # Plain arrays plus a JSON sidecar rather than one pickled object array. The
    # cache is machine-written and local, but `allow_pickle` on a path a future
    # caller could point at someone else's file is not worth saving one file.
    if cache_file.exists() and meta_file.exists():
        z = np.load(cache_file)
        meta = json.loads(meta_file.read_text())
        print(f"pose cache hit: {cache_file.name}", file=sys.stderr)
        return {
            "classes": classes,
            "windows": {
                slot: {
                    "x": z[f"x_{slot}"],
                    "y": z[f"y_{slot}"],
                    "stride": info["stride"],
                    "duration": info["duration"],
                }
                for slot, info in meta.items()
            },
        }

    import cv2

    from pipeline.pose_detector import load_pose_model

    detector = load_pose_model(str(pose_model))
    windows: dict[str, dict] = {}
    for clip_meta in m["clips"]:
        if not clip_meta.get("annotated"):
            continue
        slot = clip_meta["slot"]
        truth = json.loads((fixture_dir / clip_meta["annotation_file"]).read_text())
        stride = int(truth["stride_s"])
        clip = next(fixture_dir.glob(f"{slot}-*.mp4"), None)
        if clip is None:
            sys.exit(f"clip for {slot} not found")
        crops = extract_native_crops(clip, roi, stride, fixture_dir / "crops" / f"{slot}-native")
        labels = [s["activity_id"] for s in truth["samples"]]
        n = min(len(crops), len(labels))

        feats = np.zeros((n, 55), dtype=np.float32)
        for i in range(n):
            img = cv2.imread(str(crops[i]))
            dets = detector.detect(img)
            best = max(dets, key=lambda d: d.confidence) if dets else None
            if best is not None:
                # Last channel is the detected flag: absence is information here,
                # not a row to drop. See the module docstring.
                feats[i, :54] = pose_features(best, img.shape[1], img.shape[0])
                feats[i, 54] = 1.0
            if (i + 1) % 150 == 0:
                print(f"  [{slot}] {i + 1}/{n}", file=sys.stderr)
        windows[slot] = {
            "x": feats,
            "y": np.asarray([classes.index(v) for v in labels[:n]], dtype=np.int64),
            "stride": stride,
            "duration": truth["duration_s"],
        }

    cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_file,
        **{f"x_{s}": w["x"] for s, w in windows.items()},
        **{f"y_{s}": w["y"] for s, w in windows.items()},
    )
    meta_file.write_text(
        json.dumps(
            {s: {"stride": w["stride"], "duration": w["duration"]} for s, w in windows.items()}
        )
    )
    return {"classes": classes, "windows": windows}


def make_model(n_feat: int, n_class: int):
    import torch

    layers, c_in = [], n_feat
    for d in DILATIONS:
        layers += [
            torch.nn.Conv1d(c_in, CHANNELS, KERNEL, padding=d * (KERNEL - 1) // 2, dilation=d),
            torch.nn.GELU(),
            torch.nn.BatchNorm1d(CHANNELS),
        ]
        c_in = CHANNELS
    layers += [torch.nn.Conv1d(CHANNELS, n_class, 1)]
    return torch.nn.Sequential(*layers)


def train_fold(xtr, ytr, xte, n_class: int, device: str):
    import torch

    torch.manual_seed(SEED)
    mu, sd = xtr.mean(0, keepdims=True), xtr.std(0, keepdims=True) + 1e-6
    xtr_n = (xtr - mu) / sd
    xte_n = (xte - mu) / sd

    starts = list(range(0, max(1, len(xtr_n) - WINDOW + 1)))
    batch = np.stack([xtr_n[s : s + WINDOW] for s in starts]).transpose(0, 2, 1)
    target = np.stack([ytr[s : s + WINDOW] for s in starts])

    xb = torch.tensor(batch, device=device)
    yb = torch.tensor(target, device=device)

    counts = np.bincount(ytr, minlength=n_class).astype(np.float32)
    weight = torch.tensor(
        np.where(counts > 0, counts.sum() / np.maximum(counts, 1), 0.0), device=device
    )
    model = make_model(xtr.shape[1], n_class).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    lossf = torch.nn.CrossEntropyLoss(weight=weight)

    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(len(xb), device=device)
        for k in range(0, len(xb), 32):
            idx = perm[k : k + 32]
            opt.zero_grad()
            loss = lossf(model(xb[idx]), yb[idx])
            loss.backward()
            opt.step()
        if (ep + 1) % 20 == 0:
            print(f"  epoch {ep + 1}/{EPOCHS} loss {loss.item():.3f}", file=sys.stderr)

    # Inference: overlapping windows, logits averaged, so every frame is predicted
    # with as much surrounding context as the clip allows.
    model.eval()
    logits = np.zeros((len(xte_n), n_class), dtype=np.float32)
    counts_f = np.zeros((len(xte_n), 1), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, max(1, len(xte_n) - WINDOW + 1)):
            seg = torch.tensor(xte_n[s : s + WINDOW].T[None], device=device)
            out = model(seg)[0].T.cpu().numpy()
            logits[s : s + WINDOW] += out
            counts_f[s : s + WINDOW] += 1
    return (logits / np.maximum(counts_f, 1)).argmax(1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--pose-model", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--box", required=True)
    ap.add_argument("--gpu-index", type=int)
    args = ap.parse_args()

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        sys.exit("no CUDA device - #117 requires the measurement on a fleet GPU")

    manifest = json.loads(args.manifest.read_text())
    folds = manifest["split"]["folds"]

    t0 = time.monotonic()
    data = build_sequences(args.manifest, args.pose_model, args.out_dir / "cache")
    pose_seconds = time.monotonic() - t0
    classes, windows = data["classes"], data["windows"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    total_video = sum(w["duration"] for w in windows.values())

    for fold in folds:
        tr, te = fold["train_dev"][0], fold["held_out"][0]
        if tr not in windows or te not in windows:
            continue
        print(f"\n=== fold {fold['id']}: train {tr} -> predict {te} ===", file=sys.stderr)
        t1 = time.monotonic()
        pred = train_fold(
            windows[tr]["x"], windows[tr]["y"], windows[te]["x"], len(classes), device
        )
        fit_seconds = time.monotonic() - t1
        peak = torch.cuda.max_memory_allocated() // (1024 * 1024)

        stride = windows[te]["stride"]
        doc = {
            "arm": "tcn-pose-seq",
            "window": te,
            "fold": fold["id"],
            "model": "dilated temporal CNN over YOLO-pose features",
            "hyperparameters": {
                "window": WINDOW,
                "channels": CHANNELS,
                "dilations": list(DILATIONS),
                "kernel": KERNEL,
                "epochs": EPOCHS,
                "lr": LR,
                "seed": SEED,
                "fixed_before_run": True,
            },
            "receptive_field_frames": 1 + (KERNEL - 1) * sum(DILATIONS),
            "samples": [
                {"t_s": i * stride, "activity_id": classes[int(c)]} for i, c in enumerate(pred)
            ],
            "gpu": {
                "box": args.box,
                "gpu_index": args.gpu_index,
                "gpus_used": 1,
                # Pose is amortised across both folds; the fit is per fold. A
                # production run pays pose once and no fit at all.
                "gpu_seconds": round(pose_seconds / max(len(folds), 1) + fit_seconds, 1),
                "pose_seconds_total": round(pose_seconds, 1),
                "fit_seconds": round(fit_seconds, 1),
                "video_seconds": windows[te]["duration"],
                "peak_vram_mib": int(peak) if peak else None,
            },
        }
        out = args.out_dir / f"{te}.json"
        out.write_text(json.dumps(doc, indent=2, ensure_ascii=False))
        print(f"wrote {out}  (fit {fit_seconds:.0f}s, peak {peak} MiB)")

    print(f"\npose pass: {pose_seconds:.0f}s for {total_video:.0f}s of video")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
