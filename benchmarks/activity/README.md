# Per-person activity benchmark (framework)

Measures **activity-description accuracy** — is each detected person correctly
called `sitting` / `standing` / `walking` / `running` — as opposed to detection
recall (whether people are found), which is `benchmarks/pose-resolution/`.

The two axes are deliberately separate. A report can find every person and still
describe them all wrongly; on this camera it does, because the deployed VLM
classifies one posture per *frame* and applies it to every stationary person in
it. This framework exists to quantify that and to test fixes against ground
truth rather than promote them on intuition.

## Why a new fixture shape

The existing activity eval (`training/activity-mlp/` +
`films-ground-truth.json`) scores a **single person** as an interval timeline
(`start_s, end_s, activity`) — one activity at a time. A hall has ~5 people per
frame each doing different things, so activity here must be **per-person,
per-frame**, keyed to detection boxes. That is the shape under
`magazyn-hall-v1/`.

## The framework, per camera

Each camera fixture is stamped from its detection fixture. The steps are the same
for any camera:

1. **Detection ground truth** — a `pose-resolution`-shaped fixture with
   human-confirmed `persons[].bbox` (for `magazyn`, this is #99).
2. **Generate the annotation package**:

   ```bash
   uv run benchmarks/activity/tools/build_activity_annotation.py \
     --detection-manifest <detection manifest.json> \
     --frames-dir <frames dir> \
     --out benchmarks/activity/<camera-fixture>
   ```

   Produces per-person native crops, an `index.html`, and a
   `manifest.scaffold.json` with `activity: null` per person and a readability
   prior. Crops and `index.html` regenerate on demand and are not committed.
3. **Human labeling pass** — fill every `activity`, confirm by eye, no classifier
   involved (see the fixture's `METHODOLOGY.md`). Commit as `manifest.json`.
4. **Score classifier arms** — run each candidate classifier against the fixture;
   match detections to labeled boxes by bbox IoU; report a per-class confusion
   matrix and sitting-vs-standing accuracy on the posture-readable subset.

The generator and (forthcoming) scorer are fixture-driven, so a second camera
needs annotation, not code.

## Layout

```
benchmarks/activity/
  README.md                         this file
  tools/build_activity_annotation.py  detection fixture -> annotation package
  <camera-fixture>/
    manifest.scaffold.json          generated; activity=null, to be labeled
    manifest.json                   committed after the human pass (the fixture)
    METHODOLOGY.md                  how this fixture was labeled + its limits
    crops/ , index.html             generated review artifacts (gitignored)
```

## The second shape: process-activity INTERVALS

`hala-prawe-v1` measures something the per-frame shape cannot express. A
chronometraż records **how long each activity lasted**, so its unit is
`(zone_id, activity_id, start_s, end_s)` — an interval, not a per-frame label.
A 20-minute clip is tens of rows rather than 1200 frames × N people.

**The unit is the zone, not a person track.** Assumption A5, confirmed by the
client 2026-08-28: the measurement unit is the station, never an identified
person — `brak na stanowisku` means the zone is empty, whoever they are. So the
tool samples one station ROI over time and asks what is happening *there*. No
tracking, no re-ID, no faces. That is also what makes the hand pass tractable:
tracking every person through a 4K fisheye hall is a much larger job the
work-study does not need.

```bash
uv run benchmarks/activity/tools/build_interval_annotation.py \
  --manifest benchmarks/activity/hala-prawe-v1/manifest.source.json \
  --slot W1
```

Produces station-ROI crops at a fixed stride, an `*.intervals.scaffold.json`
with `activity: null` per sample, and a keyboard-driven `*.timeline.html` that
folds consecutive same-label samples into intervals and exports them. All three
regenerate on demand and are gitignored; the committed fixture is the exported
`*.intervals.json`.

Three properties worth knowing before using it:

- **The crop comes from the native frame.** Cropping a 1280×736 downscale would
  throw away the ~3× that makes station framing work at all (#86: station crop
  32.4% recall vs full-frame 640 at 0.0%).
- **Timing is PTS-derived, never frame-indexed.** `hala-prawe-v1`'s W1 reports
  `r_frame_rate=120/1` while the true rate is ~20 fps; anything counting frames
  mis-maps every W1 timestamp. Boundaries land at the midpoint between two
  differently-labelled samples, so error is ≤ stride/2 — ±1 s at the default
  stride of 2 s, which is the resolution the client accepted (A3).
- **Arc-flash hints are suggestions, never labels.** The seeding threshold is
  computed per clip, because the metric is clip-relative (W1 and W2 differ ~3×
  at the median on the identical crop). It detects an *arc*, not welding *work*,
  and cannot separate `ukladanie_pretow` from `postoj`. A suggestion the
  annotator never confirms never becomes a label.

## Scoring an arm (`hala-prawe-v1` / C.0)

Once a spike arm has produced predictions, score every arm through the same tool
so no two arms are graded by slightly different rules:

```bash
uv run benchmarks/activity/tools/evaluate_arms.py \
  --manifest benchmarks/activity/hala-prawe-v1/manifest.source.json \
  --predictions runs/vlm/W1.json runs/vlm/W2.json \
  --out benchmarks/activity/hala-prawe-v1/C0-report.md \
  --json-out benchmarks/activity/hala-prawe-v1/C0-report.json
```

It emits all five things `#117` asks of each arm — per-activity confusion matrix
on the held-out union, boundary timing error, GPU-seconds per video-hour, the
single-card hardware verdict, and the arc-flash comparison. Run with no
`--predictions` to regenerate just the baseline section.

Four rules worth knowing before reading its output:

- **The folds come from the manifest**, never from the tool. It refuses to run
  against a manifest with no `split` block, because a split chosen at scoring
  time is exactly what the acceptance criterion forbids.
- **So does the delivered vocabulary.** `manifest.source.json` →
  `delivery_vocabulary` names one bucket and the activities it merges, and the
  tool scores that vocabulary in its own section on top of the per-activity
  one. `--collapse pozostale=sciaganie_elementu,inna_czynnosc` overrides the
  block for exploration; the report header prints which of the two it used,
  because an ad-hoc collapse and a declared one do not carry the same weight.
  Three guards apply wherever the mapping came from: `nierozpoznane` cannot be
  a member (it is neither work nor downtime, so bucketing it would convert
  unknown time into measured time), a member that is not an activity in the
  manifest is refused rather than silently dropped, and the bucket may not
  reuse an activity's name. The two vocabularies never share a table — a merge
  cannot be undone by reading harder.
- **An unanswered sample is an error, not a smaller denominator.** An arm that
  declines to predict does not get an easier score for it.
- **Recall is the client's bar, and the bar alone is not enough.** The free
  arc-flash baseline, swept for best F1, hits 99.4% recall on `spawanie` by
  reporting 2.18× the welding time that happened. So every class also reports
  `time_ratio` (predicted seconds ÷ true seconds), and a class whose recall
  passes above 1.25× is marked **gamed** rather than passed. Over-reporting
  productive time is the direction of error a work-study client actually pays
  for.

## Shipping a station head (`hala-prawe-v1` / #122)

The classifier is two pieces with very different lifecycles. The **frozen DINOv2
backbone** is identical at every station, ships once inside the container image,
and is pinned in `setup-models.sh` (`dinov2-base-v1.0`). What is
station-specific is a **1.8 MB temporal head**. Onboarding a station is
"annotate twenty minutes, train a head" — no new large model, no engine
redeploy. Keep that separation; it is the economics of the offer.

```bash
# 1. the delivered vocabulary trained directly, for the comparison (GPU)
python benchmarks/activity/tools/run_delivered_vocabulary_arm.py \
  --manifest .../manifest.source.json --crops-root .../crops \
  --out-dir .../predictions-delivered --cache-dir runs/tcn-pixel/cache \
  --box cctv-vps --gpu-index 1

# 2. score it into its own report (laptop) - its own, because a three-class arm
#    predicts none of the four merged activities and would read as four
#    catastrophic zeroes in a seven-category table
uv run benchmarks/activity/tools/evaluate_arms.py \
  --manifest .../manifest.source.json --predictions .../predictions-delivered/*.json \
  --out .../C0-delivered-report.md --json-out .../C0-delivered-report.json

# 3. train on ALL windows, export the head, generate the card (GPU)
python benchmarks/activity/tools/train_station_head.py \
  --manifest .../manifest.source.json --crops-root .../crops \
  --cache-dir runs/tcn-pixel/cache \
  --report .../C0-report.json --direct-report .../C0-delivered-report.json \
  --version 1.0.0 --out-dir models
```

The ONNX exports need `onnxscript`, which the GPU image does not carry: prefix
the container command with
`pip install --target /app/.venv/lib/python3.12/site-packages onnxscript`.

Four things worth knowing about what comes out:

- **The shipped weights cannot be scored.** They train on every annotated window
  and have no held-out material — they reach 100% on all three windows because
  all three were in training. Every figure in the card is read from
  `C0-report.json`, measured on the cross-validated folds, on models that are
  not this file. The card says so next to the numbers, and refuses to build with
  a blank where a measurement belongs.
- **The head emits seven classes, not three.** The collapse happens after
  argmax; the card carries both lists and the mapping. Training three directly
  was measured and lost — see METHODOLOGY.
- **Both artefacts are single self-contained files.** torch writes weights to a
  sibling `.onnx.data` by default, which would leave a graph whose sha256
  verifies nothing; `assert_single_file` refuses that.
- **Preprocessing is resize 518 → centre-crop 224.** Both numbers, always. See
  METHODOLOGY for why quoting only the first is a mistake this fixture already
  made.

## Status

- `magazyn-hall-v1` — scaffold generated (296 people, 264 posture-readable prior).
  Human labeling pass pending. Tracked as the ground-truth sub-issue of the
  activity-accuracy epic.
- `hala-prawe-v1` — **three windows annotated** (2026-09-01/02): 599/599, 600/600
  and 600/600 samples at a 2 s stride. W3 is second-shift material with a
  different operator, recovered from the appliance buffer after the C.0 report
  named the missing afternoon footage as the open question. Split is **3-fold**,
  amended 2026-09-02 — that amendment came *after* three arms had been measured
  on 2-fold, so the two sets of figures are not comparable and every arm was
  rerun rather than rescored.
  **The folds disagree, and that is the result:** `spawanie` scores 97.8% and
  91.2% where the held-out window is morning material, and **27.5%** where it is
  the evening window no fold trained on. Pose detection on W3 is normal (86.5%)
  and the VLM arm, which trains on nothing, gets 97.6% there, so the footage
  reads fine and it is the trained model that fails to transfer. Read the
  finding section in `hala-prawe-v1/METHODOLOGY.md` before quoting any single
  number from this fixture, including the 72% union (`#117`).
  **Most of that gap closed on 2026-09-02** (`#120` rung 2): the same TCN fed
  frozen DINOv2 embeddings of the station crop instead of pose geometry scores
  76.5% on the unseen window and **89.0% / 88.2% on the union**, the first arm
  here to clear the bar on two activities with an honest time ratio, at 11× less
  GPU cost. The two hard hand-work classes did not move: their apparent gains are
  flagged `inflated` (1.8x to 3.2x the real duration).
