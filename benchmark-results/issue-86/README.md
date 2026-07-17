# Issue #86 measured result

No arm was eligible, so the production default was not changed. The required
narrow follow-up is [#88](https://github.com/KopalnieKrypto/cctv-gpu-engine/issues/88),
which validates a station-framed camera stream rather than adding guessed
multi-inference software tiling.

## Aggregate result

All arms used the same 60 annotated 3840×2160 frames from three windows, 74
in-zone people, YOLO11s-pose weights, confidence 0.25, NMS IoU 0.45, and TP IoU
0.5. Films 1+2 used the same 60-frame whole-frame fixtures for every arm.

| Arm | TP/FP/FN | Precision | Recall | F1 | Pose p50/p95 | VLM wallclock | Extrapolated min/h | vs baseline | Peak VRAM | Film 1 / 2 recall | Eligible |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full frame → 640 | 0/0/74 | 0.0% | 0.0% | 0.0% | 114.7 / 116.0 ms | 76.267 s | 25.14 | baseline | 7,866 MiB | 100.0% / 96.7% | no |
| full frame → 1280 | 1/0/73 | 100.0% | 1.4% | 2.7% | 383.8 / 384.8 ms | 126.393 s | 41.67 | +65.7% | 8,122 MiB | 97.1% / 96.7% | no |
| station ROI + 160 px → 640 | 24/17/50 | 58.5% | 32.4% | 41.7% | 113.0 / 114.3 ms | 96.435 s | 31.79 | +26.4% | 7,866 MiB | 100.0% / 96.7% | no |

The extrapolation in every JSON artifact is
`measured_wallclock_s / 181.995411 * 3600`, with the explicit assumption that
wallclock scales linearly with decoded duration at the locked 1 fps sampling
rate, model, classifier, and hardware. The measured VLM frame count was 182.

## Reproducibility evidence

- Host: `cctv-vps`, GPU 1, NVIDIA GeForce RTX 5070, 12,227 MiB, driver
  595.45.04. GPU 0 was excluded because an unrelated workload occupied it.
- Container:
  `ghcr.io/kopalniekrypto/cctv-gpu-engine/gpu-service@sha256:332b8b1cda232519b19341d9f894c8b0a73c68d026964320f041e8120ef2ef81`
  (image ID `0dd916d968da`, PyTorch 2.11.0+cu128).
- Fixed-640 model SHA-256:
  `b77299b0adf66cef11bf3b958ada5c311b8318ac8f8f89ecbfac29d247e4647d`.
- Fixed-1280 model SHA-256:
  `1be930081fa982dbe5942d16dcc6d3fe3a9e1e4233b1ce6e29bbfa9488de813b`.
- Per-process VRAM was sampled by PID through `nvidia-smi` every 0.5 seconds;
  `--pid=host` made the container PID namespace match telemetry.
- `baseline_640.json`, `full_frame_1280.json`, and
  `focused_roi_640.json` retain raw per-frame detections and timing.
  `selection.json` combines them with every automated eligibility check.

The host-only first attempt is not a measurement: it stopped before VLM work
with `ModuleNotFoundError: torch` and produced no arm result. All reported
numbers come from fresh, isolated containers using the pinned image above.
