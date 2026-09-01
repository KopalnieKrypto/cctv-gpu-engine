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
- **Both windows are annotated** (2026-09-01): `W1.intervals.json` 599/599 samples /
  85 intervals, `W2.intervals.json` 600/600 / 92 intervals, both at a 2 s stride with
  no unlabelled gap on the timeline. The first C.0 acceptance criterion is met.
- **The split is declared** (`manifest.source.json` → `split`): 2-fold
  cross-validation, recorded before any spike arm ran. See "Why 2-fold" below.

### Why 2-fold, and not the single split declared first

The split was first written as W1 train / W2 held out, on the reasoning that W2 is the
denser window. Annotating W2 falsified that reasoning — **not** by producing a result,
but by revealing the class support, which is why the amendment is legitimate and is
recorded rather than quietly applied.

The bar is **per activity**. On W2 alone, three classes cannot carry it:

| Class | W1 | W2 | Problem on a W2-only held-out set |
|---|---|---|---|
| `brak_na_stanowisku` | 66 | **2** | one error moves the score 50 pp — unmeasurable |
| `postoj` | 42 | **16** | 6.2 pp per error; 85% is coarser than the granularity |
| `nierozpoznane` | **0** | 58 | a trained arm would never see an example of it |

Under 2-fold every class clears 52 held-out samples — `spawanie` 497,
`ukladanie_pretow` 359, `inna_czynnosc` 107, `brak_na_stanowisku` 68, `nierozpoznane`
58, `postoj` 58, `sciaganie_elementu` 52 — and every labelled sample is predicted
exactly once by a model that never saw it. With two folds the variance on a 52-sample
class is still wide; report it as indicative, not settled.

### W2 annotation, cross-checked against the arc signal

Same audit as W1, and it lands differently in a way the report has to carry:

| | W1 | W2 |
|---|---|---|
| arc-seconds proposed | 196 | 430 |
| of which hand-labelled `spawanie` | 162 (**82.7%**) | 242 (**56.3%**) |
| leaked into `ukladanie_pretow` | 28 | **159** |
| hand-labelled `spawanie` total | 447 s | 546 s |
| arc-flash recall on `spawanie`, conservative cut-off | **36.2%** | **44.3%** |

**The arc threshold's precision collapses from 82.7% to 56.3% between two windows of
the same station on the same day** — 159 s of W2's arc time sits inside hand-labelled
`ukladanie_pretow`. This is the clip-relative caveat in `manifest.source.json`
demonstrated rather than asserted: the threshold was tuned on W1's brightness and does
not port to W2 an hour and twenty minutes later. Any arm compared against this baseline
must be compared against a **per-clip re-tuned** baseline, or the comparison flatters
the model.

### Correction: the "recall ceiling" was an artefact of one threshold

An earlier version of this file concluded that arc-flash **cannot clear the 85% bar on
`spawanie` by construction**. That is wrong, and the threshold sweep in
`tools/evaluate_arms.py` is what falsified it. The 36.2% / 44.3% figures above are
recall at *one conservative cut-off*, not a ceiling. Swept for best F1 per clip, the
same signal reaches **99.4% recall on `spawanie`** — it clears the client's bar
comfortably.

It does so by calling **2.18× as much time `spawanie` as actually happened**, at 45.6%
precision. The two operating points are the same signal read differently:

| Operating point | Recall | Precision | Time reported |
|---|---:|---:|---:|
| conservative (annotation-hint cut-off) | 40.4% | 93.1% | 0.43× |
| oracle F1 (in-sample, hindsight) | **99.4%** | 45.6% | **2.18×** |

The real finding is therefore **about the bar, not about the baseline**: *a
recall-only bar is gameable by any arm willing to over-call the common class.* A free
brightness threshold clears 85% on `spawanie` while being useless, so no arm may be
promoted on per-activity recall alone. `evaluate_arms.py` now reports `time_ratio`
next to every class and marks a class **gamed** rather than passed above 1.25×.

What survives unchanged is the reason arc-flash is not a candidate, and it never
depended on the threshold: **it cannot separate `ukladanie_pretow` from `postoj` at
all** — five of the seven activities, and all of the hard part. It remains the cost
floor.

**`nierozpoznane` on W2: 58 samples (9.7%, 116 s)** against zero on W1. The honest-
uncertainty check that W1 could only argue, W2 now demonstrates.

### W1 annotation, cross-checked against the arc signal

The arc-flash timeline is an independent machine measurement, so it audits the hand
labels for free. Of the 196 arc-seconds proposed on W1, **82.7% (162 s) fall inside
hand-labelled `spawanie`**, the rest in `ukladanie_pretow` (28 s) and
`inna_czynnosc` (6 s) — consistent with ±1 s boundary slop and spatter visible while
the operator handles rods. That agreement also confirms the annotation is indexed by
`pts_time`: had W1's bogus `r_frame_rate=120/1` been trusted anywhere in the chain,
the two timelines would not line up at all.

Two numbers fall out of it that the C.0 report needs:

- **At the conservative cut-off, arc-flash recall on `spawanie` is 36.2%.**
  Hand-labelled welding *work* totals 447 s on W1; arc fires on 162 s of it. The gap
  is real welding — positioning, tacking, chipping slag, the pauses between beads.
  This is a property of that operating point, **not a ceiling** — see the correction
  above, where a swept threshold reaches 99.4% recall at 2.18× the true time.
- **The station runs a regular ~82 s production cycle.** Long welding blocks start at
  219, 303, 391, 477 … 1139 s — gaps of 84, 88, 86, 80, 78, 86, 82, 78 s, with one
  258 s outlier that is exactly the 132 s `brak_na_stanowisku` absence at 539–671 s.
  A temporal model therefore has real periodic structure to exploit here, which is an
  argument for the sequence-segmentation arm over per-frame classification.

**Zero `nierozpoznane` on W1**, which the rule above says to distrust. Here it is
defensible and the report must say why rather than leave it implied: the labels were
made inside the 900×800 native-pixel station crop, where the operator stands ~500 px
tall and helmet, gloves, torch and rebar jig are individually legible — not on the
fisheye hall view the rule was written for. **Resolved by W2**, which came back with 58
`nierozpoznane` samples from the same annotator on the same crop: the label was
available and was used where the footage warranted it, so W1's zero reflects an easy
window rather than an annotator who settled.

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

### ⚠️ The preserved R2 copies could not be found (2026-09-01)

Checked while staging the clips onto `cctv-vps` for the spike, and it did not go the
way this document assumed. Using the GPU stack's own credentials (`.env.gpu`,
bucket `surveillance-data`):

- `benchmarks/hala-prawe-v1/W1-2026-08-28T0700Z.mp4` → **HeadObject 404**
- the `benchmarks/` prefix → **0 objects**
- the only top-level prefix in the bucket is `surveillance-jobs/`; there is no
  `tenants/` prefix either
- `list_buckets` → `AccessDenied`, so **other buckets could not be checked**

So this is not proof the copies are gone — they may sit in the platform's own bucket,
which these credentials cannot enumerate. It *is* proof that `r2_key_preserved` in
`manifest.source.json` does not resolve with the credentials this repo has, and that
nobody has actually verified the copies since they were made.

**What is confirmed to exist right now:** the working copy on the dev Mac, and a copy
staged onto `cctv-vps` at
`/home/mvp/cctv-gpu-engine/benchmarks/activity/hala-prawe-v1/` (pushed over scp,
because the clips are gitignored and cannot travel by `git pull` — the repo's
"never rsync/scp" rule governs source, not fixture data).

Two copies on two machines, neither of them object storage, for material this document
calls irreplaceable. **Someone should confirm the R2 copies exist and record which
bucket holds them**, or re-upload from the Mac and note the bucket in the manifest.
Until then, do not treat `r2_key_preserved` as a backup that has been checked.
