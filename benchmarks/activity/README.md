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

## Status

- `magazyn-hall-v1` — scaffold generated (296 people, 264 posture-readable prior).
  Human labeling pass pending. Tracked as the ground-truth sub-issue of the
  activity-accuracy epic.
- `hala-prawe-v1` — clips verified, station ROI fixed, interval scaffolds
  generated for both windows (599 + 600 samples at 2 s). **Human labeling pass
  pending** — it is the binding input on the C.0 feasibility gate
  (`#117`).
