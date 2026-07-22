# Non-square pose input — measured evidence (issue #100)

Letterboxing a 16:9 frame into a square model input spends a fixed fraction of
every tensor on constant grey. This records what was measured after widening
`preprocessing`/`pose_detector` to accept an explicit `[1, 3, H, W]` export.

All numbers below are measured on `cctv-vps` (RTX 5070, 12 227 MiB, CUDA 12.8,
onnxruntime-gpu), engine commit `549a3c0`, 2026-07-22.

## Exports

`ultralytics` 8.4.104 exports `yolo11s-pose` non-square without complaint. The
`imgsz` argument is `(height, width)`.

| Model | `imgsz` | ONNX input | Output | sha256 |
|---|---|---|---|---|
| `yolo11s-pose-640x384.onnx` | `(384, 640)` | `[1, 3, 384, 640]` | `[1, 56, 5040]` | `8c9e97a5d5ce5f77d1f4b64a57c4e981e1d8d7484276efb7dd9588570620ecfb` |
| `yolo11s-pose-1280x736.onnx` | `(736, 1280)` | `[1, 3, 736, 1280]` | `[1, 56, 19320]` | `7ee0fcd86efdf953678a92649a682b4369ee5372ebf2623876602286c37731e9` |

Ultralytics warns that `yolo val` on a non-PyTorch model requires square input.
That is a limitation of its own validation entry point, not of the export — our
scoring runs through `pipeline.pose_benchmark`, which reads the shape from the
ONNX.

## CUDA execution provider

Every export loads with `CUDAExecutionProvider` active after
`session.get_providers()` — the silent-CPU-fallback guard
(microsoft/onnxruntime#25145) passes for non-square just as for square. Square
`640×640` and `1280×1280` were loaded in the same process and same run, so the
widened contract demonstrably did not replace the old one.

## Compute

Detection only (`PoseDetector.detect`, 3840×2160 input frame, 5 warm-up +
40 timed iterations, no JPEG decode):

| Arm | Input | ms/frame | Tensor vs 640×640 | **Measured vs 640×640** |
|---|---|---:|---:|---:|
| `baseline_640` | 640×640 | 99.42 | 1.00× | **1.00×** |
| `baseline_640x384` | 640×384 | 62.28 | 0.60× | **0.63×** |
| `full_frame_1280` | 1280×1280 | 384.81 | 4.00× | **3.87×** |
| (non-square 1280 arm) | 1280×736 | 222.34 | 2.30× | **2.24×** |

Measured cost tracks tensor area closely. The saving is real and it is not
assumed: **640×384 costs 0.63× of 640×640**, i.e. 37% off detection wall-clock
for the same detection scale. The higher-resolution arm #101 wants costs
**2.24×**, not 3.87× — which is the difference the issue predicted would decide
whether that arm is affordable at all.

Padding measured directly on a 16:9 tile: **43.92%** of a 640×640 tensor is
`PAD_VALUE`, against **6.54%** at 640×384.

## Quality

### Aspect-ratio guard (issue #83's evidence, re-run)

Scored on the `[1280, 0]–[2560, 720]` rectangle of all 60 `bending-pilot-v1`
native frames — the four-worker reference tile recovered in that fixture's
`METHODOLOGY.md` — with a deliberately squashed control standing in for the
pre-#83 behaviour:

| Preprocessing | Input | Detections (60 frames) | Frames agreeing with letterboxed 640×640 |
|---|---|---:|---:|
| letterbox | 640×640 | 61 | — |
| letterbox | 640×384 | 60 | 57 / 60 |
| **squash (control)** | 640×640 | **41** | **37 / 60** |

The control is what makes this a guard rather than a formality: breaking aspect
ratio still costs a third of the detections, exactly the failure mode #83
recorded (1 of 4 workers found, versus 4 of 4). Non-square letterboxing does
not show that signature — it tracks the square letterbox.

On the full 3840×2160 frames: 640×640 finds 10, 640×384 finds 11, agreeing on
59 of 60 frames.

### `bending-pilot-v1` in-zone score

| Arm | TP | FP | FN | Recall | `film_1` recall | `film_2` recall |
|---|---:|---:|---:|---:|---:|---:|
| `baseline_640` | 0 | 0 | 74 | 0.000 | 1.000 | 0.967 |
| `baseline_640x384` | 0 | 0 | 74 | 0.000 | 1.000 | 0.967 |

640×384 is **no worse** — but read this honestly: the shipped 640 baseline
already scores zero in-zone on this fixture, which is #86's own recorded result
(its focused-ROI arm found 41 where the baseline found 0). A fixture on which
the reference arm detects nothing cannot discriminate between two arms, so the
`films-v1` recall guard and the tile counts above are where the comparison
actually carries information. Both are bit-identical or better.

**The second half of that criterion — the same comparison against #99's
`magazyn-hall-v1` fixture — is not measured here.** #99 is blocked (no source
footage reachable, and the fixture requires manual annotation), so no such
fixture exists yet. This is the same gap #101 is blocked on.

## Reproducing

```bash
# on cctv-vps, repo root
uv run python - <<'PY'
from pipeline.pose_detector import load_pose_model
d = load_pose_model("models/yolo11s-pose-640x384.onnx")
print(d.input_size, d.session.get_providers())
PY
```

The benchmark harness gained a `baseline_640x384` arm, so a full measured run
lands in the same shape as #86's:

```bash
uv run python -m pipeline.pose_benchmark run-arm \
  --arm baseline_640x384 \
  --model models/yolo11s-pose-640x384.onnx \
  ...   # remaining flags exactly as in POSE_RESOLUTION_BENCHMARK.md
```

`ARM_INPUT_SIZES` pins each arm's exact `(w, h)` and the loader refuses a model
that declares anything else, so an arm cannot silently be measured at the wrong
shape. #86's three-arm `select` gate is deliberately untouched.
