# `hala-prawe-v1` — process-activity ground truth

Gates **C.0** in `gpu-exchange/docs/prd/slice-chronometraz.md`. Measures whether the
client's seven process activities are separable on **their actual camera**, which is a
whole-hall 4K fisheye and not the station-framed view the PRD originally assumed.

## Why a third fixture shape

There are now three annotation shapes in this repo, and they are not interchangeable:

| Fixture | Shape | Answers |
|---|---|---|
| `training/activity-mlp/` + `films-ground-truth.json` | one person, interval timeline | when did *the* worker change activity |
| `magazyn-hall-v1/` | many people, **per-person per-frame** posture | is each detected person's posture right |
| **`hala-prawe-v1/`** (this one) | many people, **per-track interval timeline** | when did *each* worker start and stop each process activity |

This fixture is the intersection of the other two, and neither existing tool produces
it. A hall has several workers doing different things at once, so a single timeline is
wrong; but a chronometraż records *intervals*, not per-frame labels, so a per-frame
matrix is the wrong unit and vastly more expensive to annotate.

The unit is therefore **`(track_id, activity_id, start_s, end_s)`**. A 20-minute clip is
perhaps 30–50 such intervals across all workers — tractable by hand, unlike 1200 frames
× N people.

## What the annotator does

For each clip, for each person track:

1. Mark the interval boundaries where the activity changes.
2. Label each interval with one of the seven ids in `manifest.source.json`.
3. Use **`nierozpoznane`** whenever the footage genuinely does not show enough to tell.
   This is not failure — it is the honest label, and a fixture with zero `nierozpoznane`
   on a fisheye hall view is a fixture someone guessed on.

Two rules that decide whether the resulting number means anything:

- **`postoj` requires all three conditions**: in the zone, not moving, nothing in the
  hands. A worker standing still holding a rod is `ukladanie_pretow` or
  `inna_czynnosc`, never `postoj`.
- **`inna_czynnosc` is real work.** Carrying wire, grinding, measuring, adjusting the
  bender. It exists so that the five client-named activities do not have to absorb
  everything else, which would inflate them. It is not a synonym for `nierozpoznane`.

## The bar

**≥85% correct per activity**, on material held out of the training set — the client's
own number and definition (assumption A11, 2026-08-28). Per activity, not overall: a
rare class that is always wrong cannot hide behind an average.

**Detection recall caps this.** An undetected person cannot be classified, so a run that
finds half the people cannot exceed ~50% recall on any activity no matter how good the
classifier is. That is why the clips are captured at `1280x736` and not at the shipped
`640x640` default — see `manifest.source.json` → `camera.pose_rationale`.

## Provenance and retention

Source clips live in R2 and are **not committed** (see `.gitignore`) — same convention as
`magazyn-hall-v1/`, where `crops/`, `index.html` and `review.html` are regenerated
artifacts.

The clips were captured as normal platform tasks, so their original keys sit under
`tenants/{tenant}/appliance-uploads/`, which the C-5 retention reaper deletes on task
completion. Each clip therefore has an `r2_key_preserved` copy outside that prefix, with
no `uploads` row pointing at it, which is what makes it invisible to
`findReapableTaskInputs`. **If the reaper reaches production before the copies exist, the
source material is gone permanently.**

## Status

- **W1** (09:00–09:20) and **W2** (10:20–10:40) captured 2026-08-28. Both pre-break.
- **W3 (afternoon) is DROPPED** — decision 2026-09-01, proceed on two windows. This
  fixture therefore represents **only the first half of a shift**, permanently, and
  every accuracy figure derived from it inherits that bias. Quote it with the caveat
  attached or do not quote it.
- **Content verified 2026-09-01.** The welding station is manned in both windows by the
  same operator, and both contain real arc time: **W1 ≈ 140 s (11.7%)**, **W2 ≈ 310 s
  (25.9%)**, measured at 1 fps by saturated-pixel fraction (`Y>235`) inside
  `station_roi`. W2 is the denser of the two. Person counts: W1 avg 3.18 / peak 6,
  W2 avg 4.28 / peak 7, `recall_risk: normal` on both.
- **No clip is annotated yet.** `annotated: false` throughout — this is the one
  remaining C.0 input with no shortcut.

## What the fixture already proves

Two things worth stating before anyone annotates a frame:

1. **The vocabulary gap is reproducible on demand.** The operator mid-arc, sparks
   visible, is classified `standing` by the shipped four-pose model — and `standing`
   maps to `Praca` in this camera's `activity_label_map`. Welding and standing idle
   are the same label today. That is the whole reason this slice exists, and it is now
   a screenshot rather than an argument.
2. **A2 does not bind at the station.** The client refused a *physical* reframe, which
   was read as killing station-framed optics. It does not: the welding operator stands
   ~700 px tall in the native 4K frame, and median detected person height is 332 px
   (W1) / 319 px (W2) against a 180 px resolvable floor. A *software* crop of the
   station ROI recovers station framing from footage we already receive. The crop must
   come from the native frame — cropping the 1280x736 downscale throws away the 3x.

## Arc flash as a weak label — and its limits

The saturated-pixel scan is cheap enough to run over any clip and pre-seeds `spawanie`
candidate intervals for the annotator. Two limits keep it honest:

- **It is clip-relative, not a constant.** W1 and W2 differ roughly threefold at the
  median (0.28 vs 0.62) on the identical crop, so ambient light drift alone would move
  a fixed threshold. Normalise per clip.
- **It detects arc, not work.** It cannot separate `ukladanie_pretow` from `postoj`,
  which is most of the vocabulary. Treat it as a floor for the classification spike:
  a model that costs a GPU and cannot beat a brightness threshold on `spawanie` has
  not earned its place.

## The source material is irreplaceable

Both originals under `tenants/{tenant}/appliance-uploads/` **were deleted** by the C-5
retention reaper — verified 2026-09-01, the keys are gone and both tasks read
`inputUploadIds: []`. The `benchmarks/hala-prawe-v1/` copies survived only because
someone made them by hand at 11:04 CEST on 2026-08-28, about 6.5 hours before the
reaper reached production at 17:34 CEST.

There is no second copy, and the appliance buffer is 3 h deep, so nothing can reproduce
this footage. Note what that implies: **the preserve step is still manual**, nothing in
either repo automates it, and the platform has no retention-hold flag. The next fixture
captured without that hand-copy will be lost.
