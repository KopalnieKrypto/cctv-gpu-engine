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

## Status

- `magazyn-hall-v1` — scaffold generated (296 people, 264 posture-readable prior).
  Human labeling pass pending. Tracked as the ground-truth sub-issue of the
  activity-accuracy epic.
