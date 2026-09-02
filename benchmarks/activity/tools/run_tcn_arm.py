"""C.0 arms 2 and 3: temporal segmentation over pose sequences.

Both arms live here on purpose. They share the detector, the cache, the model and
every hyperparameter, and differ only in `--features`, so any gap between their
results is attributable to the encoding and to nothing else. Running them as two
scripts would have made that claim unverifiable.

    --features pose   arm 2: posture only (55 + a found flag)
    --features rich   arm 3: posture + placement, motion at two lags, stillness,
                      and the arc-flash metric

## Why arm 3 exists, and what informed its design

Arm 2's failures were not spread evenly. `spawanie` and `ukladanie_pretow`
cleared the bar at 94.6% and 96.4%; `postoj` managed 12.1%, `inna_czynnosc`
26.2%, `sciaganie_elementu` 73.1%.

Arm 2 normalises every frame by its own bounding box. That makes posture
comparable across distances, and it also erases position - and therefore erases
motion entirely. For the two activities with distinctive silhouettes that costs
nothing. For `postoj` it is fatal, because the vocabulary defines that class by
what the body is *not* doing: "in the zone, not moving, nothing in the hands"
(`METHODOLOGY.md`). A representation with no notion of movement cannot express
it, however much temporal context sits on top.

**Stated plainly, because it matters for how these numbers are read:** arm 3's
feature set was chosen after seeing arm 2's per-class results on held-out
material. The justification is definitional rather than numerical - `postoj` is
defined by stillness, so stillness features follow from the vocabulary and not
from the scores - but the ordering is what it is, and pretending otherwise would
be the kind of quiet retrofit this fixture's split exists to prevent. Model and
hyperparameters are unchanged from arm 2 precisely to keep the comparison to one
variable.

---

Arm 2's original rationale follows.

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

- **Every fold, from the manifest, and every window inside it.** Under the 3-fold
  split each fold trains on two windows and predicts the third. Sequences are cut
  per window before pooling, so no training sample spans the seam between two
  clips. Per-activity accuracy is then reported by `evaluate_arms.py` on the
  union, where every labelled sample is predicted once by a model that never
  trained on it.
- **Hyperparameters are fixed before the run** and are not adjusted against
  results. They are recorded in the output. If this arm fails, it is reported as
  failing - tuning against the held-out window until it passes would spend the
  clean windows in the fixture, and there are no more.
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
sys.path.insert(0, str(Path(__file__).parent))

from run_vlm_arm import VramSampler  # noqa: E402  (sibling tool, same fleet-VRAM method)

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


# bbox 4 | conf 1 | keypoints 51 | crop w,h 2 | found flag 1
RAW_WIDTH = 59
FOUND = 58


def raw_detection(det, crop_w: int, crop_h: int) -> np.ndarray:
    """Everything the detector said, in crop pixels, before any encoding choice.

    Cached in this form so a new feature set costs seconds instead of a five
    minute pose pass over both windows - and so the two arms are provably built
    from identical detections rather than from two runs that might differ.
    """
    kp = np.asarray([[k.x, k.y, k.vis] for k in det.keypoints], dtype=np.float32).reshape(-1)
    return np.concatenate(
        [np.asarray(det.bbox, dtype=np.float32), [det.confidence], kp, [crop_w, crop_h], [1.0]]
    ).astype(np.float32)


def _posture(raw: np.ndarray) -> np.ndarray:
    """Arm 2's encoding: translation- and scale-free keypoints."""
    x1, y1, x2, y2 = raw[0:4]
    conf = raw[4]
    kp = raw[5:56].reshape(17, 3)
    crop_w, crop_h = raw[56], raw[57]
    bw, bh = max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)
    return np.concatenate(
        [
            (kp[:, 0] - x1) / bw,
            (kp[:, 1] - y1) / bh,
            kp[:, 2],
            [bh / max(bw, 1e-6), bh / max(crop_h, 1e-6), bw / max(crop_w, 1e-6), conf],
        ]
    ).astype(np.float32)


def _placement(raw: np.ndarray) -> np.ndarray:
    """Where in the station the body actually is, in crop coordinates.

    Arm 2 threw this away on purpose - normalising every frame by its own bbox
    makes posture comparable but erases position and therefore erases motion.
    That is defensible for `spawanie` and `ukladanie_pretow`, which have
    distinctive shapes, and fatal for `postoj`, which the vocabulary defines by
    what the body is NOT doing: "in the zone, not moving, nothing in the hands".
    """
    x1, y1, x2, y2 = raw[0:4]
    crop_w, crop_h = max(raw[56], 1e-6), max(raw[57], 1e-6)
    kp = raw[5:56].reshape(17, 3)
    cx, cy = (x1 + x2) / 2 / crop_w, (y1 + y2) / 2 / crop_h
    # Wrists relative to the hip midpoint: hands are what separates carrying,
    # placing and doing nothing, and hip-relative keeps that readable when the
    # operator moves around the bench.
    hip = (kp[11, :2] + kp[12, :2]) / 2
    wl = (kp[9, :2] - hip) / crop_h
    wr = (kp[10, :2] - hip) / crop_h
    return np.concatenate(
        [[cx, cy, (x2 - x1) / crop_w, (y2 - y1) / crop_h], wl, wr, [kp[9, 2], kp[10, 2]]]
    ).astype(np.float32)


PLACE_WIDTH = 10  # cx, cy, w, h | left wrist xy | right wrist xy | both wrist confidences


def build_feature_matrix(raw: np.ndarray, mode: str, arc: np.ndarray | None) -> np.ndarray:
    """Per-frame features for a whole window, from cached raw detections."""
    n = len(raw)
    found = raw[:, FOUND : FOUND + 1]
    posture = np.stack([_posture(r) if r[FOUND] else np.zeros(55, np.float32) for r in raw])
    if mode == "pose":
        return np.concatenate([posture, found], axis=1)

    place = np.stack(
        [_placement(r) if r[FOUND] else np.zeros(PLACE_WIDTH, np.float32) for r in raw]
    )

    # Motion, at two time scales. A frame where nothing moved for six seconds is
    # `postoj`; a brief large displacement is `sciaganie_elementu` lifting the
    # finished element clear of the bench.
    def deltas(a: np.ndarray, lag: int) -> np.ndarray:
        d = np.zeros_like(a)
        d[lag:] = a[lag:] - a[:-lag]
        # A frame either side of a detection gap has a meaningless delta.
        gap = (found[:, 0] == 0) | (np.roll(found[:, 0], lag) == 0)
        d[gap] = 0.0
        return d

    d1, d3 = deltas(place, 1), deltas(place, 3)
    speed1 = np.linalg.norm(d1, axis=1, keepdims=True)
    speed3 = np.linalg.norm(d3, axis=1, keepdims=True)
    # Stillness over +/-2 frames (a 10 s window at a 2 s stride).
    still = np.zeros((n, 1), np.float32)
    for i in range(n):
        lo, hi = max(0, i - 2), min(n, i + 3)
        still[i] = place[lo:hi].std(axis=0).mean()

    arc_col = (arc if arc is not None else np.zeros(n, np.float32)).reshape(n, 1)
    return np.concatenate(
        [posture, place, d1, d3, speed1, speed3, still, arc_col, found], axis=1
    ).astype(np.float32)


def arc_series(arc_csv: Path, window: str, stride: int, n: int) -> np.ndarray | None:
    """Per-sample arc-flash metric, normalised WITHIN the clip.

    The raw metric is clip-relative - W1 and W2 differ ~3x at the median on the
    identical crop - so a shared absolute scale would feed the model a different
    quantity per window. Normalising per clip is what makes it portable.
    """
    if not arc_csv.exists():
        return None
    import csv as _csv

    per_second: dict[int, float] = {}
    with arc_csv.open() as fh:
        for row in _csv.DictReader(fh):
            if row.get("window") == window:
                per_second[int(row["t_s"])] = float(row["arc_metric"])
    if not per_second:
        return None
    vals = np.asarray([per_second.get(t, 0.0) for t in range(0, n * stride)], dtype=np.float32)
    hi = float(np.percentile(vals, 99)) or 1.0
    out = np.zeros(n, dtype=np.float32)
    for i in range(n):
        seg = vals[i * stride : (i + 1) * stride]
        out[i] = float(seg.max()) / hi if len(seg) else 0.0
    return np.clip(out, 0.0, 2.0)


def build_sequences(manifest: Path, pose_model: Path, cache: Path) -> dict:
    """Per-frame features for every annotated window, cached by detector identity."""
    m = json.loads(manifest.read_text())
    fixture_dir = manifest.parent
    roi = m["station_roi"]["crop"]
    classes = [a["id"] for a in m["activities"]]

    # The annotated slots belong in the key. Without them, adding a window to the
    # fixture hits the old cache and returns the old window set, so a 3-fold run
    # would quietly reuse a 2-window pose pass.
    slots = ",".join(sorted(c["slot"] for c in m["clips"] if c.get("annotated")))
    key = hashlib.sha256(
        f"{pose_model.name}:{pose_model.stat().st_size}:raw{RAW_WIDTH}:{slots}".encode()
    ).hexdigest()[:12]
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

        feats = np.zeros((n, RAW_WIDTH), dtype=np.float32)
        for i in range(n):
            img = cv2.imread(str(crops[i]))
            dets = detector.detect(img)
            best = max(dets, key=lambda d: d.confidence) if dets else None
            if best is not None:
                # The found flag stays 0 on a miss: absence is information here,
                # not a row to drop. See the module docstring.
                feats[i] = raw_detection(best, img.shape[1], img.shape[0])
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


def fit_model(train_clips, n_class: int, device: str):
    """Fit on one or more training windows; return the model and its normalisation.

    Split out of `train_fold` so #122 can ship the fitted weights rather than only
    their predictions. The body is unchanged and `train_fold` is now a thin caller,
    so the cross-validated arms and the shipped head are trained by literally the
    same code - which is the claim the model card makes about them.

    `train_clips` is a list of `(x, y)` per source window, never one concatenated
    array. Sliding sequences are cut inside each window and only then pooled, so
    no training sequence ever straddles the seam between two clips - a seam that
    joins a Friday morning to a Monday evening and would teach the model a
    transition that never happened.
    """
    import torch

    torch.manual_seed(SEED)
    xtr = np.concatenate([x for x, _ in train_clips], axis=0)
    ytr = np.concatenate([y for _, y in train_clips], axis=0)
    mu, sd = xtr.mean(0, keepdims=True), xtr.std(0, keepdims=True) + 1e-6

    seqs, targets = [], []
    for x, y in train_clips:
        xn = (x - mu) / sd
        for s in range(0, max(1, len(xn) - WINDOW + 1)):
            seqs.append(xn[s : s + WINDOW])
            targets.append(y[s : s + WINDOW])
    batch = np.stack(seqs).transpose(0, 2, 1)
    target = np.stack(targets)

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
    return model, mu, sd


def train_fold(train_clips, xte, n_class: int, device: str):
    """Fit on the training windows and predict the held-out one."""
    import torch

    model, mu, sd = fit_model(train_clips, n_class, device)
    xte_n = (xte - mu) / sd

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
    ap.add_argument(
        "--features",
        default="pose",
        choices=("pose", "rich"),
        help="pose = arm 2 (posture only); rich = arm 3 (posture + placement, motion, arc)",
    )
    args = ap.parse_args()

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        sys.exit("no CUDA device - #117 requires the measurement on a fleet GPU")

    manifest = json.loads(args.manifest.read_text())
    folds = manifest["split"]["folds"]

    # Sample the whole process from nvidia-smi, not torch.cuda.max_memory_allocated.
    # The torch allocator sees only the TCN - about 17 MiB - and misses the ONNX
    # pose session entirely, which is the arm's real memory cost. Reporting the
    # torch figure as the hardware verdict would understate it by ~40x.
    vram = VramSampler()
    vram.__enter__()
    t0 = time.monotonic()
    data = build_sequences(args.manifest, args.pose_model, args.out_dir / "cache")
    pose_seconds = time.monotonic() - t0
    pose_cached = pose_seconds < 5
    classes, windows = data["classes"], data["windows"]

    # Both feature sets are derived from the same cached raw detections, so the
    # two arms differ by their encoding and by nothing else.
    feats = {
        slot: build_feature_matrix(
            w["x"],
            args.features,
            arc_series(args.manifest.parent / "arc-timeline.csv", slot, w["stride"], len(w["x"]))
            if args.features == "rich"
            else None,
        )
        for slot, w in windows.items()
    }
    print(f"features: {args.features}, {next(iter(feats.values())).shape[1]} per frame")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    total_video = sum(w["duration"] for w in windows.values())

    for fold in folds:
        # Every train_dev slot, not just the first. Under the 3-fold split each
        # fold names two training windows, and silently taking [0] would train on
        # half the data the split promised while still reporting the fold's name.
        tr_slots = [s for s in fold["train_dev"] if s in windows]
        te = fold["held_out"][0]
        if len(tr_slots) != len(fold["train_dev"]) or te not in windows:
            sys.exit(f"fold {fold['id']} names a window the fixture does not have")
        joined = "+".join(tr_slots)
        print(f"\n=== fold {fold['id']}: train {joined} -> predict {te} ===", file=sys.stderr)
        t1 = time.monotonic()
        pred = train_fold(
            [(feats[s], windows[s]["y"]) for s in tr_slots], feats[te], len(classes), device
        )
        fit_seconds = time.monotonic() - t1
        torch_peak = torch.cuda.max_memory_allocated() // (1024 * 1024)
        # A cache hit never loads the pose session, so the sampled peak would
        # cover the TCN alone. Report it as unmeasured rather than as a real
        # end-to-end figure - the harness then refuses to call the arm a pass.
        peak = None if pose_cached else (vram.peak_mib or None)

        stride = windows[te]["stride"]
        doc = {
            "arm": f"tcn-{args.features}",
            "window": te,
            "fold": fold["id"],
            "trained_on": tr_slots,
            "model": f"dilated temporal CNN, {args.features} features over YOLO-pose",
            "feature_set": args.features,
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
                "peak_vram_note": (
                    "pose session not loaded (cache hit) - end-to-end peak unmeasured"
                    if pose_cached
                    else "whole process, nvidia-smi per PID: ONNX pose session + TCN"
                ),
                "torch_allocator_peak_mib": int(torch_peak),
            },
        }
        out = args.out_dir / f"{args.features}-{te}.json"
        out.write_text(json.dumps(doc, indent=2, ensure_ascii=False))
        print(f"wrote {out}  (fit {fit_seconds:.0f}s, peak {peak} MiB)")

    vram.__exit__()
    print(f"\npose pass: {pose_seconds:.0f}s for {total_video:.0f}s of video")
    print(f"peak VRAM (proces): {vram.peak_mib or 'UNMEASURED'} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
