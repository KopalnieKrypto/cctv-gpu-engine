# Issue #101 measured result

No arm was eligible, so the production default was not changed. Unlike #86 this
is not a no-winner in the sense of "nothing moved": `full_frame_1280x736`
delivers **5.0× the recall** of the shipped config on this scene and raises
precision from 64% to 89%. It fails on throughput, and so — newly measured here
— does the config already in production.

The full write-up, including what the numbers redirect the work toward, is in
[#97](https://github.com/KopalnieKrypto/cctv-gpu-engine/issues/97).

## Aggregate result

All arms scored against `benchmarks/pose-resolution/magazyn-hall-v1` — 60
manually annotated 3840×2160 frames across 3 windows, **296 human-confirmed
people**, whole-frame scoring zone, confidence 0.25, NMS IoU 0.45, TP IoU 0.5.
Films 1+2 are the same whole-frame regression fixtures #86 used.

| Arm | TP/FP/FN | Precision | Recall | F1 | Pose p50/p95 | Wallclock | Extrapolated min/h | vs baseline | Peak VRAM | Film 1 / 2 recall | Eligible |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full frame → 640 (**shipped**) | 21/12/275 | 63.64% | 7.09% | 12.77% | 114.6 / 116.4 ms | 83.053 s | 27.68 | baseline | 7,868 MiB | 100.0% / 96.7% | no |
| full frame → 640×384 | 20/14/276 | 58.82% | 6.76% | 12.12% | 79.2 / 80.0 ms | 76.849 s | 25.61 | −7.47% | 7,868 MiB | 100.0% / 96.7% | no |
| full frame → 1280×736 | 105/13/191 | 88.98% | 35.47% | 50.72% | 231.5 / 233.6 ms | 129.958 s | 43.31 | +56.48% | 8,122 MiB | 97.1% / 96.7% | no |

The extrapolation in every JSON artifact is
`measured_wallclock_s / 180.037988 * 3600`, with the explicit assumption that
wallclock scales linearly with decoded duration at the locked 1 fps sampling
rate, model, classifier, and hardware. The measured frame count was 180.

## Recall by ground-truth person height

The result that matters more than the aggregate, because it says where the next
lever must aim.

| GT height (native px) | people | `baseline_640` | `baseline_640x384` | `full_frame_1280x736` |
|---|---:|---:|---:|---:|
| < 80 | 32 | 0.0% | 0.0% | 0.0% |
| 80–120 | 90 | 0.0% | 0.0% | 0.0% |
| 120–180 | 34 | 0.0% | 0.0% | 11.8% |
| 180–260 | 91 | 1.1% | 1.1% | 71.4% |
| ≥ 260 | 49 | 40.8% | 38.8% | 73.5% |

**The effective detection floor is ~60 px of person height at model input.**
Below it recall is flatly zero; above it it steps to ~70%. Two consequences:

- 80–120 px native is the largest bucket — 90 of 296, 30% of the fixture — and
  is 0.0% at every arm. Reaching a 60 px input height for that band needs model
  input ≥ 1920 px wide on a 3840 px frame. Only native-resolution tiling gets
  there.
- At ≥ 260 px native (87 px at input, far above the floor) recall still tops out
  at 73.5%. That 26% is model quality, not sampling geometry, and it caps any
  resolution-or-tiling strategy at roughly three quarters.

## Cost split

| Arm | Pose p50 | Pipeline | Pose share | Non-pose | Detections on fixture |
|---|---:|---:|---:|---:|---:|
| `baseline_640` | 114.6 ms | 461 ms/frame | 24.8% | 347 ms/frame | 33 |
| `baseline_640x384` | 79.2 ms | 427 ms/frame | 18.5% | 348 ms/frame | 34 |
| `full_frame_1280x736` | 231.5 ms | 722 ms/frame | 32.1% | 490 ms/frame | 118 |

Pose is a minority of the shipped pipeline. Non-pose cost rises 347 → 490
ms/frame at the higher resolution while detections go 33 → 118: finding more
people costs downstream work per detection (Re-ID embedding, tracking) even
though the VLM is one call per frame.

## Reproducibility evidence

- Host `cctv-vps` (`mvp-serwer`), NVIDIA GeForce RTX 5070 **GPU 1**, 12,227 MiB,
  driver 595.45.04. GPU 0 was excluded because the SGLang staging LLM occupied
  it (9,222 MiB) — the same exclusion #86 made.
- Container
  `ghcr.io/kopalniekrypto/cctv-gpu-engine/gpu-service@sha256:ef0557417b519c8de84f8394c035f13c503f86400773e6d2818381c91ade7c50`.
  **Not** #86's pinned `332b8b1c…`, which is no longer on the box: the three arms
  here are mutually comparable, but absolute wallclock is not bit-comparable with
  #86's numbers.
- Model SHA-256, read from the bytes each session opened (#98):
  - `baseline_640` → `models/yolo11s-pose.onnx`
    `469beac503fdc788ea3980331bc4bfbd2bd00de3772eb0984f4c53032740583f` —
    byte-identical to the `setup-models.sh` pin, so this row is the shipped
    config rather than a look-alike re-export.
  - `baseline_640x384` →
    `8c9e97a5d5ce5f77d1f4b64a57c4e981e1d8d7484276efb7dd9588570620ecfb`
  - `full_frame_1280x736` →
    `7ee0fcd86efdf953678a92649a682b4369ee5372ebf2623876602286c37731e9`
- Fixture `magazyn-hall-v1` at `89a8287`; all 63 asset SHA-256 verified on the
  box before the run, and `manifest.json` re-verifies every frame hash before
  CUDA loads.
- Per-process VRAM sampled by PID through `nvidia-smi`; `--pid=host` made the
  container PID namespace match telemetry. The 1280×736 delta is **+254 MiB**
  against the +258 MiB #101 predicted detection-only, and clears the 8 GiB policy
  gate by 70 MiB.
- One arm per fresh container, identical ordered `--throughput-clip` arguments.
- `baseline_640.json`, `baseline_640x384.json` and `full_frame_1280x736.json`
  retain raw per-frame detections and timing. Eligibility verdicts come from
  `pose_benchmark.evaluate_eligibility`, so the bounds are the locked ones rather
  than re-typed.

The arm shape `full_frame_1280x736` did not exist in `ARM_INPUT_SIZES` before
`fe4f963`; `run-arm` rejected the model until it was registered. No aggregate
`select` artifact is produced here — `build_results_artifact` stays locked to
#86's three-arm triple, which is #86's record and not this run's.

## Regression to note

`film_1` whole-frame recall drops 1.0 → 0.9706 on the 1280×736 arm. Small, but a
real fail on the no-regression check: the higher-resolution arm loses one
detection on close-framed footage.
