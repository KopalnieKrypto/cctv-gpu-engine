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

## How to run the pass

```bash
# clips must be present locally (see "The source material is irreplaceable")
uv run benchmarks/activity/tools/build_interval_annotation.py \
  --manifest benchmarks/activity/hala-prawe-v1/manifest.source.json --slot W1
open benchmarks/activity/hala-prawe-v1/W1.timeline.html
```

Keys `1`–`7` label the current sample and advance; `←`/`→` navigate, `0` clears,
`s` accepts an arc-flash suggestion. Progress is kept in the browser, so a
partial pass survives a reload. "Eksportuj interwały" writes
`W1.intervals.json` — **that** file is the fixture and gets committed. The
crops, the scaffold and the HTML are regenerated artifacts and are gitignored.

At the default 2 s stride each window is ~600 samples, so a full pass is roughly
600 keystrokes per clip. The export carries both the folded `intervals` and the
raw per-sample `samples`, so the fold can be re-derived and audited rather than
taken on trust.

Two things the tool will not do for you. It never guesses an activity — every
sample starts `null`. And its arc hints cover only `spawanie`; the other six
activities are entirely hand work, which is most of the vocabulary and all of
the hard part.

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
- **W1 is annotated** (2026-09-01, `W1.intervals.json`): 599/599 samples at a 2 s
  stride, 85 intervals, no unlabelled gap on the timeline. **W2 is not** — it is the
  one remaining C.0 input with no shortcut.
- **The split is declared** (`manifest.source.json` → `split`): W1 train/dev, W2
  held out, recorded before any spike arm ran.

### W1 annotation, cross-checked against the arc signal

The arc-flash timeline is an independent machine measurement, so it audits the hand
labels for free. Of the 196 arc-seconds proposed on W1, **82.7% (162 s) fall inside
hand-labelled `spawanie`**, the rest in `ukladanie_pretow` (28 s) and
`inna_czynnosc` (6 s) — consistent with ±1 s boundary slop and spatter visible while
the operator handles rods. That agreement also confirms the annotation is indexed by
`pts_time`: had W1's bogus `r_frame_rate=120/1` been trusted anywhere in the chain,
the two timelines would not line up at all.

Two numbers fall out of it that the C.0 report needs:

- **The arc-flash baseline's recall ceiling on `spawanie` is 36.2%.** Hand-labelled
  welding *work* totals 447 s on W1; arc fires on 162 s of it. The gap is real
  welding — positioning, tacking, chipping slag, the pauses between beads. So the
  free baseline cannot reach the 85% bar on `spawanie` **by construction**, not by
  being badly tuned. That converts "the baseline is not a solution" from an argument
  into a measurement.
- **The station runs a regular ~82 s production cycle.** Long welding blocks start at
  219, 303, 391, 477 … 1139 s — gaps of 84, 88, 86, 80, 78, 86, 82, 78 s, with one
  258 s outlier that is exactly the 132 s `brak_na_stanowisku` absence at 539–671 s.
  A temporal model therefore has real periodic structure to exploit here, which is an
  argument for the sequence-segmentation arm over per-frame classification.

**Zero `nierozpoznane` on W1**, which the rule above says to distrust. Here it is
defensible and the report must say why rather than leave it implied: the labels were
made inside the 900×800 native-pixel station crop, where the operator stands ~500 px
tall and helmet, gloves, torch and rebar jig are individually legible — not on the
fisheye hall view the rule was written for. The honest-uncertainty check still has to
be re-applied to W2 independently; if W2 also comes back with zero, that is worth a
second look at whether the crop is simply easy or the annotator settled.

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
