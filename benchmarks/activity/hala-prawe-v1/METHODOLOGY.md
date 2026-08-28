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
- **W3** (afternoon) not yet captured. Until it is, this fixture represents only the
  first half of a shift and any accuracy figure from it carries that caveat.
- No clip is annotated yet. `annotated: false` throughout.
