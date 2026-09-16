# The resolution floor for a station zone (#124)

Measured 2026-09-16 on `cctv-vps` GPU 1, inside
`ghcr.io/kopalniekrypto/cctv-gpu-engine/gpu-service:latest`.

`zone-annotator` has to warn an annotator, before a session starts, that a zone is
too small to be worth labelling. The threshold `#119` named came from the pose
detector's recall floor (`#113`), and the winning arm runs no pose detector, so
that basis no longer exists. This is the replacement, measured rather than
inherited.

## What was done

The station crop was degraded to a fraction of its native resolution and restored
to full size, so the tensor reaching the backbone is identical at every scale and
the only variable is how much detail the camera resolved. The head was **refit at
every scale** over the same five cross-validation folds, and every set of
predictions went through `evaluate_arms.py`, the same scorer as every other arm in
this fixture.

Refit rather than evaluate: running the shipped head on shrunken inputs would
measure train/test mismatch, which is an easier and different question than
whether the information survives at all.

Person height comes from `measure_person_height.py`, which reads the tallest
detection above the pipeline's 0.25 confidence floor on every tenth native crop.
The pose detector is used there purely as a ruler. It is not in the inference
path and does not run at annotation time.

**Median person height at full resolution: 558 px** inside the 1670x800 rectangle,
which is 0.70 of its height. 300 frames sampled, 287 with a person.

## Results, delivered vocabulary, union of the five folds

Recall, then the ratio between reported and true duration. `n` is held-out support.

| detail | person px | spawanie n=1287 | ukladanie_pretow n=1081 | pozostale n=568 | nierozpoznane n=62 |
|---:|---:|---|---|---|---|
| 1.0 | 558 | **90.5%**  0.97x | **86.4%**  0.96x | 57.7%  1.04x | 0.0%  1.79x |
| 0.75 | 419 | 94.3%  1.03x | 83.6%  0.95x | 53.2%  0.92x | 6.5%  2.02x |
| 0.5 | 279 | 95.3%  1.05x | 84.8%  0.91x | 51.4%  0.87x | 4.8%  2.63x |
| 0.35 | 195 | 93.9%  1.08x | 82.5%  0.92x | 51.9%  0.84x | 3.2%  2.21x |
| 0.25 | 140 | 92.5%  1.05x | 80.9%  0.87x | 54.4%  0.99x | 1.6%  2.39x |
| 0.18 | 100 | 92.9%  **1.10x** | 78.7%  0.85x | 50.0%  0.94x | 0.0%  1.92x |
| 0.125 | **70** | 92.3%  1.09x | 82.6%  0.91x | 46.7%  0.88x | 0.0%  1.81x |
| 0.09 | 50 | **83.5%**  0.91x | 84.3%  0.94x | 60.7%  **1.22x** | 1.6%  2.00x |
| 0.06 | 34 | 78.9%  0.86x | 74.7%  0.97x | 60.7%  **1.35x** | 1.6%  1.27x |
| 0.045 | 25 | 81.7%  0.95x | 61.1%  0.73x | 61.8%  **1.59x** | 0.0%  1.35x |
| 0.03 | 17 | 83.4%  0.95x | 63.2%  0.76x | 60.7%  **1.56x** | 0.0%  1.10x |

## The proposed floor, as a rule

**A station zone in which a person stands less than about 70 pixels tall is below
the floor. Between 70 and roughly 110 pixels, hand-work activities are already
measurably degraded even though welding is not.**

The rule that produced the first number was declared in the committed script
before anything ran: the smallest detail scale at which `spawanie` recall stays
within 5.0 percentage points of the full-resolution figure **and** its reported
duration stays within 10% of the truth. It selects detail scale 0.125, which is
70 px of person height.

Recall alone would not do. An arc-flash threshold in this project once hit 99.4%
recall while reporting 2.18x the real welding time, and this sweep reproduces the
same trap in a different place: below 0.09 the `pozostale` bucket's recall *rises*
to around 61% while its reported duration goes to 1.59x. The model is not seeing
more, it is calling everything "other" as the detail disappears.

## Three things the table says that the headline does not

**Welding is the wrong class to be reassured by.** `spawanie` holds above 92% all
the way down to 70 px and never really collapses, sitting at 83% even at 17 px. An
arc is a large, saturated, blue-white blob and stays visible long after the person
does. The class that actually tracks resolution is the hand work:
`ukladanie_pretow` falls from 86.4% to 78.7% by 100 px and to 61% by 25 px. For a
tool whose next job is a different domain with no arc in it, the hand-work curve
is the one that generalises.

**The rule operates near noise at the bottom.** The passing set is 1.0, 0.75, 0.5,
0.35, 0.25 and 0.125 — with 0.18 failing, and failing only because its reported
duration is 1.105x against a band edge of 1.10x. A non-contiguous passing set is a
sign the criterion is discriminating between neighbours it cannot really tell
apart. The honest statement is that the floor lies between 0.125 and 0.09, which
is between 70 and 50 px, and 70 px is the conservative end of that.

**The sweep's declared range did not bind on the first pass.** The seven scales
fixed before the run bottomed out at 0.125 without finding a collapse, so the rule
selected the smallest scale it had been given, which is not a measurement. The
four scales below it were chosen *after* seeing that, purely to locate where the
curve turns, and they carry a weaker status than the declared seven. They are
reported separately for that reason rather than folded in silently.

## What this does not measure

The human half. Whether an annotator can still tell these activities apart at
70 px is a different question, it needs inter-annotator agreement, and this
fixture does not have any. `zone-annotator` phase 9
(`KopalnieKrypto/zone-annotator#10`) must therefore present this threshold as
provisional in the interface until the other half exists.

Also unmeasured: whether the floor transfers to another station, another camera
height or another vocabulary. One station, one rectangle, five windows.

## Reproducing

Cut the native crops if they are not present, then:

```bash
docker run --rm --gpus '"device=1"' --user $(id -u):$(id -g) \
  -v /home/mvp/cctv-gpu-engine:/work -w /work \
  -e HF_HOME=/hf -v /home/mvp/hf-cache:/hf \
  --entrypoint python \
  ghcr.io/kopalniekrypto/cctv-gpu-engine/gpu-service:latest \
  benchmarks/activity/tools/measure_person_height.py \
    --manifest benchmarks/activity/hala-prawe-v1/manifest.source.json \
    --crops-root benchmarks/activity/hala-prawe-v1/crops \
    --pose-model models/yolo11s-pose.onnx \
    --out runs/resolution/person-height.json

# same container, then:
  benchmarks/activity/tools/run_resolution_sweep.py \
    --manifest benchmarks/activity/hala-prawe-v1/manifest.source.json \
    --crops-root benchmarks/activity/hala-prawe-v1/crops \
    --out-dir runs/resolution --box cctv-vps --gpu-index 1 \
    --person-height-px 558.2
```

The refit is **not** reproducible bit for bit: the seed does not cover cuDNN's
convolution atomics, which is why `--score-only` exists and why the 55 fold
documents under `runs/resolution/predictions` on the box are the record of what
was actually measured, not something regenerable on demand.

## Files

- `summary.json` — the table above plus the knee verdict and the rule behind it
- `sweep-report.json`, `sweep-report.md` — the full `evaluate_arms` output for all
  eleven arms, including per-window and per-fold breakdowns
- `person-height.json` — the ruler's output, per window and pooled
