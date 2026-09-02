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

A window not yet in `arc-timeline.csv` gets **measured** rather than silently skipped —
saturated-pixel fraction (`Y>235`) inside `station_roi` at 1 fps, appended to the CSV.
Before that, the timeline was hand-made for W1/W2 and unreproducible, so a new window
got no hints and the tool presented that as "no arc here" rather than "nobody measured
this". `--no-arc` opts out.

Two things the tool will not do for you. It never guesses an activity — every
sample starts `null`. And its arc hints cover only `spawanie`; the other six
activities are entirely hand work, which is most of the vocabulary and all of
the hard part. **Treat the hints as weaker on some windows than others**: their
precision against hand labels runs 82.7% on W1, 56.3% on W2 and 32.2% on W3.

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
- **W3 was dropped, then recovered.** Dropped 2026-09-01 (afternoon window not
  captured); recovered the same evening from the appliance's rolling buffer once the
  C.0 report named the missing second-half-of-shift material as the open question.
  **18:25–18:45 local, second shift, a different operator.** The original decision's
  cost is not erased by the recovery: W1 and W2 are still both pre-break, so any
  figure computed over those two alone still carries that bias, and two of the three
  windows still do.
- **Content verified 2026-09-01.** The welding station is manned in both windows by the
  same operator, and both contain real arc time: **W1 ≈ 140 s (11.7%)**, **W2 ≈ 310 s
  (25.9%)**, measured at 1 fps by saturated-pixel fraction (`Y>235`) inside
  `station_roi`. W2 is the denser of the two. Person counts: W1 avg 3.18 / peak 6,
  W2 avg 4.28 / peak 7, `recall_risk: normal` on both.
- **All three windows are annotated**: `W1.intervals.json` 599/599 samples / 85
  intervals and `W2.intervals.json` 600/600 / 92 intervals (2026-09-01),
  `W3.intervals.json` 600/600 / 75 intervals (2026-09-02). All at a 2 s stride with
  no unlabelled gap on the timeline.
- **The split is 3-fold** (`manifest.source.json` → `split`), amended 2026-09-02.
  Read the integrity note below before quoting any figure produced under it.

### The split, and the two amendments behind it

It has been rewritten twice, and the two rewrites are not equivalent. That
distinction is the point of writing any of this down.

**First amendment (2026-09-01), before any arm ran.** The split started as W1 train /
W2 held out, on the reasoning that W2 is the denser window. Annotating W2 falsified
that — not with a result, but by revealing class support, which only existed once the
labels did. On W2 alone `brak_na_stanowisku` had 2 samples (one error moves the score
50 pp), `postoj` 16, and `nierozpoznane` 0 on W1, so a trained arm would never see one.
2-fold fixed the granularity but not the disjointness.

**Second amendment (2026-09-02), after three arms had been measured.** W3 was recovered
and annotated, so the fixture moved to 3-fold. This is the one to be careful with.

> **Integrity note.** The manifest records
> `declared_before_any_model_run: false` for the current protocol. That flag was
> **true** for 2-fold and is **false** now, and it is set honestly rather than carried
> over. A figure produced under 3-fold is **not comparable** to the arm results already
> published under 2-fold, and the two must never appear in the same table. Anything
> measured from here is a fresh measurement of a re-scoped fixture, not an improvement
> on an old number.

What the third window actually fixes:

| Class | W1 | W2 | W3 | Total | Effect |
|---|---:|---:|---:|---:|---|
| `spawanie` | 224 | 273 | 247 | 744 | already fine |
| `ukladanie_pretow` | 180 | 179 | 193 | 552 | already fine |
| `inna_czynnosc` | 65 | 42 | 70 | 177 | more headroom |
| `sciaganie_elementu` | 22 | 30 | **58** | **110** | doubles |
| `brak_na_stanowisku` | 66 | 2 | **24** | 92 | **no longer disjoint** |
| `nierozpoznane` | 0 | 58 | 4 | 62 | **no longer disjoint** |
| `postoj` | 42 | 16 | **4** | **62** | **not fixed** |

The two disjoint classes are the real win: under 2-fold each model was asked for a
class it had never seen, and both scored a meaningless 0.0%. Every fold now trains on
at least some of both.

**`postoj` is not fixed, and the reason matters.** W3 contributes four samples — eight
seconds in twenty minutes. Across an hour of footage the class totals 62 samples. The
C.0 report asked whether the failing classes were limited by the camera or by data
volume; for `postoj` the answer is neither. **The activity is rare at this station**,
so more comparable footage cannot rescue a class that barely happens. Collecting
another three windows would add perhaps a dozen samples.

`nierozpoznane` also stays thin in one direction: the fold holding out W2 trains on
only 4 examples of it.

### Two confounds are now entangled

W3 differs from W1 and W2 in **both** shift and operator. If an arm scores worse on
W3, this fixture cannot say which caused it, and "the evening is harder" would be an
unsupported claim. Separating them needs an evening window with the morning operator,
or vice versa, which nothing currently on disk provides.

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

### W3 annotation, cross-checked against the arc signal

Same audit again, and the trend it started continues to its conclusion:

| | W1 | W2 | **W3** |
|---|---:|---:|---:|
| arc metric p50 | 0.28 | 0.62 | **2.83** |
| arc metric max | 4.91 | 9.79 | 8.44 |
| **max ÷ p50** | **17.5×** | **15.8×** | **3.0×** |
| precision against hand labels | 82.7% | 56.3% | **32.2%** |
| coverage of hand-labelled `spawanie` | 36.2% | 44.3% | **33.8%** |

**On W3 the arc hint is barely better than noise.** Of the 518 seconds above the
clip-relative threshold, only 167 fall in hand-labelled `spawanie` while 200 fall in
`ukladanie_pretow`. The peak is comparable to W2's, but the *baseline* is ten times
higher, so the arc no longer stands out from the scene the way it did in the morning
windows.

**A correction, because I got this wrong in passing.** On first inspection I sampled
three frames — the minimum showed no arc, the median showed a small one, the maximum a
full arc with sparks — and concluded the elevated median was signal rather than ambient
light, so the 276 suggested samples were "plausibly right". The full cross-check says
otherwise. Three frames were not a sample. The metric does carry *some* signal, but the
threshold placement on this window is poor and the suggestions were unreliable.

The annotation itself is not contaminated by them: hand-labelled `spawanie` is 247
samples against 276 suggested, and the two overlap on only a third of the welding.
That is what independent labelling looks like, rather than someone accepting hints.

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

### W3 has a different provenance, and the same fragility

W1 and W2 were captured as platform tasks. **W3 was not.** It was cut by hand from the
appliance's rolling buffer on `cctv-vps-camera` — 20 consecutive one-minute chunks
concatenated with stream copy, so the frames are the recorder's own and were never
re-encoded. Its container metadata is honest (`r_frame_rate=20/1`, 23 985 frames over
1200.04 s), so W1's `120/1` trap does not recur.

That buffer is **three hours deep**, which is the whole reason this material exists at
all: the recovery happened at 21:20 local, mid-shift, with the buffer holding
18:21–21:21. **The first 85 minutes of that shift were already gone.** A window from a
past day cannot be recovered this way at all.

The full three hours it was cut from are preserved at
`~/preserve-2026-09-01-zmiana2` on `cctv-vps-camera`, outside the rotating directory,
so further windows can be cut without re-capturing. **That directory is on one machine
and is not backed up anywhere**, and neither is `W3-2026-09-01T1625Z.mp4` beyond the
dev Mac. Same gap as above, one day newer.

**If a future window is wanted, schedule the capture.** Reconstructing one after the
fact only works inside a three-hour tail, and only by luck.
