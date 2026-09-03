# Issue #123 — a real station task, end to end

`#211`'s commit recorded that "AC6 (a real station task end to end) is not
covered here". This is that run.

`cctv-vps` GPU 1, image `ghcr.io/kopalniekrypto/cctv-gpu-engine/gpu-service:latest`,
engine at `88091a4`, 2026-09-03. All three `hala-prawe-v1` windows through the
shipped path:

```bash
python -m pipeline.analyze <clip>.mp4 \
  --classifier station \
  --zones benchmarks/activity/hala-prawe-v1/zones.station.json \
  --output <slot>.result.json
```

## What ran

All three clips completed, ~175 s wall for 1198 s of video each, matching the
521 GPU-s/video-hour already recorded in `README.md`. Sample counts came back
599 / 600 / 600, equal to the annotation's own grid on every window.
`diagnostics` carries the head's sha256 `38e678de…`, the zone at
`900x800 (1700, 1360)`, and no `pose_mode`, `model_sha256` or `detection_scale`,
which is the honest signal that no pose pass happened.

## What it can and cannot show

**The shipped head trained on all three of these windows.** This run is
therefore **in-sample** and is not an accuracy measurement. Read it as a
plumbing test: if the crop rectangle, the backbone preprocessing, the stride or
the interval folding were wrong, an in-sample run could not score near the
ceiling.

| Delivered category | in-sample, this run | cross-validated (`tcn-pixel-518`) |
|---|---:|---:|
| `spawanie` | 98.7% (1.03x) | 89.0% (1.06x) |
| `ukladanie_pretow` | 98.4% (1.02x) | 88.2% (1.10x) |
| `pozostale` | 89.8% (0.93x) | 45.1% (0.84x) |
| `nierozpoznane` | 95.2% (1.00x) | 0.0% (0.50x) |

Scored by `evaluate_arms.py` under the manifest's `delivery_vocabulary`, the
same tool and rules as every arm. **The right-hand column is what may be
quoted.** The left-hand one only says the pipeline reproduces what the model
knows.

The gap between the columns is the expected shape of an in-sample run and not a
finding. A run that scored *badly* here would have been the finding.

## What it found

`coverage.fraction` came back **1.0017 on W2**: 600 samples predicted against
599 "possible". A fraction above 1.0 is impossible by the field's own
definition.

`samples_possible` was `int(duration_s // stride_s)`, while the sampler takes
frame indices 0, k, 2k, … and so still produces a sample on the last step when
the frame count is odd. W2 is 1199.03 s at a 2 s stride: 600 samples, denominator
599. Now derived as `ceil(round(duration_s * fps) / frames_per_sample)`, which
matches the sampler on all three windows, with a regression test built from W2's
shape.

The magnitude is one sample. What made it worth fixing is which field it was:
coverage exists so that a missing sample cannot quietly flatter every share
above it, and a denominator that can be too small breaks that guarantee in the
direction nobody would check.
