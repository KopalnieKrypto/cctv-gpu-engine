# Pipeline

MP4 → YOLO11-pose → person tracking → activity classification → `result.json`.

```
iter_frames (ffmpeg @1fps)
  → pose_detector      detect people + COCO keypoints
  → tracker            who is this? → track_id           (issue #32)
  → activity_classifier / vlm_classifier   what are they doing?
  → track_filter       has this identity earned the right to count?
  → aggregator         person-minutes, timeline, keyframes
  → report_json        canonical result.json artifact
```

`detections_dump` taps the stream *before* filtering, so `detections.jsonl`
records every raw detection — including ones the filter later rejected. That is
what makes it an audit trail (see "why was this counted?" below).

## Tracking (issue #32)

Each detection gets a `track_id` that is stable for as long as the same person
stays in view. Two consequences worth knowing:

- **Association is by appearance, not position.** At 1 fps a person crosses
  most of the frame between samples, so bbox overlap (IoU) carries almost no
  signal. Identity comes from OSNet Re-ID cosine similarity instead.
- **A track must be seen 3 times within 5 frames before it counts.** Sporadic
  false positives — the bench-on-wheels that Film 1 reported as a second
  person — never get there. Nothing is thrown away on confidence alone: a
  faint box can extend a track it matches, it just cannot found a new one.

  The advisory specified "3 *consecutive* frames". Real detections are not
  consecutive: measured on cctv-vps, a genuinely-present person was detected in
  frames 0, 2, 3, 5, 6 — eight sightings, never three in a row. A strict
  consecutive rule discarded that person entirely (and a second one with seven
  sightings), which swaps the client's over-count for a worse under-count. The
  window keeps the artifact filter while letting real, flickering people
  through.

Because a track can take its whole window to prove itself, `track_filter` is a
delay line: frames reach the aggregator four steps late. Frames are always
emitted, even when everyone in them was rejected — the frame happened, so
duration and frame totals stay honest.

### Open: the match threshold is untuned

`MATCH_THRESHOLD = 0.5` is provisional and **not yet validated on real
footage**. OSNet similarities occupy a narrow high band — on a real frame two
unrelated crops already score ~0.49 — so 0.5 sits at the floor and leans toward
merging two people into one track. Tuning it needs same-person vs
different-person crops from the pilot camera; until then, treat multi-person
person-minutes as provisional.

### Known failure modes

**Body-only Re-ID decays with time.** OSNet matches on clothing and body shape,
which drift with lighting, pose and occlusion. Roughly: ~95% re-match at a 30 s
gap, ~80% at 2 min, ~60–70% at 5 min, under 50% beyond. `max_track_age_s`
defaults to 120 s and retires anything older rather than trust a stale match.

**Longer gaps, and any identity question that spans videos, cameras or days,
need face recognition — which is deliberately out of scope.** Do not raise
`max_track_age_s` to chase returners across long absences: the failure it buys
is two different people merged into one identity, which silently corrupts
person-minutes. A split shows up as an absence gap, which is visible and
recoverable. Split over merge, always.

**A person who returns after 120 s starts a new track.** For the bending-station
pilot this is accepted: we report the absence gap and do not claim same-identity
across it. Distinguishing a replacement worker from the same worker returning
("swap detection") was deferred by the client.

## Why was this counted?

Run with `--dump-detections out.jsonl`. Each line is one frame; each person
carries `track_id`. A detection with `"track_id": null` was never given an
identity (too faint to start a track, matched nothing). A `track_id` seen
fewer than 3 times within any 5-frame span was rejected by the filter and
contributed nothing to `result.json`.

`--no-tracker` disables tracking entirely and reproduces pre-#32 numbers, where
every detection counted. It exists for baseline comparison and as a rollback —
not for production.

## Focused inference ROI (issue #86)

The optional top-level `inference_roi` in `--zones zones.json` focuses the one
pose call on the bounding rectangle of an existing semantic zone plus an
explicit fixed pixel margin:

```json
{
  "inference_roi": {"zone_id": "bending-1", "margin_px": 160},
  "zones": [
    {
      "id": "bending-1",
      "name": "Giętarka 1",
      "polygon": [[1200, 500], [2600, 500], [2600, 1900], [1200, 1900]]
    }
  ]
}
```

The crop is clipped to each frame and letterboxed to the model's declared
fixed square input. Bboxes and keypoints are translated back into full-frame
pixels before semantic zone assignment, tracking, annotation, or reporting.
The margin must be explicit, finite, and non-negative. A missing field keeps
full-frame inference; malformed, zero-area, or fully off-frame ROIs fail
visibly. Geometry above is illustrative only—pilot geometry must come from the
versioned annotation fixture.

The ROI path is available for issue #86's comparison but is not automatically
the production default. See [the benchmark runbook](../docs/POSE_RESOLUTION_BENCHMARK.md).
