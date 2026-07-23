# magazyn-hall-v1 activity annotation methodology

> **STATUS: PROPOSED, AWAITING HUMAN CONFIRMATION (2026-07-23).** 296 people
> carry a *proposed* activity in `manifest.proposed.json` (real `activity` still
> null, `review_status: pending`). The proposals are 130 from the human detection
> notes and 134 from an agent crop read; 32 inherit `unresolved` from the size
> prior. A human confirms/overrides before this becomes `manifest.json`. See
> "Preliminary proposals".

## Preliminary proposals — and what they already reveal

`manifest.proposed.json` was drafted from human-authored signal only (the #99
detection `note` fields) plus an agent crop read where notes were silent, with
provenance recorded per person (`proposal_source` ∈ note / crop / size_prior) and
17 genuinely ambiguous people flagged `contested`. No classifier was used.

The proposed distribution is **243 standing, 20 walking, 33 unresolved, 0
sitting, 0 running**. That is a load-bearing finding, not a footnote:

- The frame-level defect this whole fixture targets is **sitting-vs-standing**.
  If the confirmed ground truth also has ~0 sitting, this scene *cannot exercise
  the defect*: the whole-frame "standing" verdict will be accidentally correct
  almost everywhere, so the whole-frame VLM will score high here for the wrong
  reason, and the per-person-crop fix will show little improvement to measure.
- magazyn was the right scene for the **detection** problem (#97/#101) and is a
  poor scene for the **activity** fix, precisely because a fabrication hall has no
  posture diversity. Proving the crop fix needs a scene with real sitters — an
  office, control room, or break area. This reshapes the epic's go/no-go: the
  second-camera step should be chosen for posture diversity, likely pulled
  forward, not left as a "reproduce the win" afterthought.

Two false-positive classes were caught and removed during the proposal QA, and
are guarded against in `tools/propose_activity_labels.py` so they do not recur
per camera: `"edge sits on the tile boundary"` (box geometry, not posture — all 7
were standing workers) and two spurious `running` matches on stationary welders.

## Why this fixture exists

`magazyn-hall-v1` (the detection fixture under `benchmarks/pose-resolution/`)
answers *were the people found*. This fixture answers a different and, for the
product, more important question: **is what each person is described as doing
correct.**

The two are not the same axis and must not be conflated:

- Detection recall (#97/#101) is measured against per-person **boxes**.
- Activity accuracy is measured against per-person **labels** — one of
  `sitting` / `standing` / `walking` / `running` for each detected person.

The deployed classifier has a specific, measurable defect this fixture exists to
quantify. Per the engine SPEC, the VLM "classifies the **frame-level** stationary
posture," and per-detection displacement can override an individual to `walking`.
So in any frame with more than one stationary person, **every non-moving person
receives the same sitting-or-standing label** — the whole frame is called
"standing" or "sitting" as a unit. On this camera that is a mean of ~5 people per
frame sharing one posture verdict. The sitting-vs-standing axis is therefore the
fixture's **primary** measurement target, because it is the axis the current
pipeline gets structurally wrong.

## What it is

A per-person activity label attached to each of the 296 human-confirmed boxes in
`benchmarks/pose-resolution/magazyn-hall-v1/manifest.json`. Same frames, same
boxes, same camera — only the question changes. Reusing #99's boxes means the
activity fixture inherits its config-derived, detector-independent sampling; no
new frames are drawn and no detector is consulted at any point.

Vocabulary: the four #33 classes plus an `unresolved` sentinel for people too
small or occluded to read a posture from. `unresolved` is a property of *this
fixture's evidence*, not a fifth class the classifier emits — it marks people
whose activity the ground truth cannot support, so the fixture never asserts a
posture it did not actually see.

## The sampling limitation, stated up front

The frames are `magazyn-hall-v1`'s: 20 per window at `fps=1/3`, i.e. **3 seconds
apart**. That interval is fine for reading a *stationary posture* off a single
still — sitting versus standing is legible — but it is too sparse for a human to
judge `walking` versus `running`, or to distinguish a walker from someone paused
mid-step, from adjacent frames.

Consequences, applied honestly:

- **Sitting / standing** — labeled from the single crop. This is the primary axis
  and it is reliable.
- **Walking** — labeled only when the still itself is unambiguous (a clear
  mid-stride leg split, motion blur on the limbs). Otherwise the visible
  stationary posture is recorded, not an inferred one.
- **Running** — essentially unjudgeable from a 3-second-spaced still and rare in
  this scene; expect approximately none, and require an unambiguous still.
- When genuinely torn between `standing` and `walking`, the label is the
  **visible** posture, never a guess about motion the fixture cannot see.

A fixture that needs `walking`/`running` measured to the same standard would have
to re-sample this footage densely (e.g. short high-fps bursts). That is out of
scope here and noted as the known ceiling of this fixture.

## Readability floor

`build_activity_annotation.py` marks a `posture_prior` of `readable` at or above
**80 px native bbox height** and `unresolved` below it. This is a floor on what a
*human* can label from a still — 264 of 296 people (89%) clear it — **not** a
claim about what the classifier can resolve. The classifier's own floor is higher
and unknown; discovering it is the benchmark's job, not the annotation's. The
prior is a hint the reviewer overrides per crop, exactly as #99's boxes were
confirmed by eye rather than trusted from the pass.

## Annotation pass

1. Generate the package (already done, reproducible):

   ```bash
   uv run benchmarks/activity/tools/build_activity_annotation.py \
     --detection-manifest benchmarks/pose-resolution/magazyn-hall-v1/manifest.json \
     --frames-dir benchmarks/pose-resolution/magazyn-hall-v1/frames \
     --out benchmarks/activity/magazyn-hall-v1
   ```

   Emits `crops/` (one native-resolution crop per person, small ones upscaled for
   viewing only), `index.html` (crops grouped by frame, largest first, each with
   the #99 detection note beside it), and `manifest.scaffold.json`.

2. A human labels every crop in `manifest.scaffold.json`, setting `activity` and
   `posture_readable`, and flips `review_status` to `confirmed`. The detection
   `note` is context, **not** a label — postures are read off the crop.

3. The confirmed file is committed as `manifest.json`, replacing the scaffold, in
   the two-commit pattern #99 used (scaffold → annotations).

**No classifier participates.** Auto-labeling with any candidate arm would make
that arm its own ground truth — the same circularity #99 guards against — and is
forbidden here for the same reason.

## Scoring intent

Downstream, the activity benchmark matches a classifier's detections to these
boxes by bbox IoU on the same frame and compares labels on the `posture_readable`
subset. The headline metric is per-person sitting-vs-standing accuracy among
stationary people — the axis the whole-frame classifier gets wrong — reported as
a confusion matrix, not a single number, so a classifier that collapses everyone
to one posture is visible as such.

## Generic across cameras

Nothing here is specific to `magazyn`. The generator takes any #99-shaped
detection manifest, so an activity fixture for a second camera is: annotate that
camera's detection boxes, run the generator, run the labeling pass. The framework
recipe lives in `benchmarks/activity/README.md`.
