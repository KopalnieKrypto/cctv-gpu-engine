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
- **A track must persist 3 consecutive frames before it counts.** Sporadic
  false positives — the bench-on-wheels that Film 1 reported as a second
  person — do not survive that. Nothing is thrown away on confidence alone:
  a faint box can extend a track it matches, it just cannot found a new one.

Because proof arrives on a track's 3rd frame, `track_filter` is a delay line:
frames reach the aggregator two steps late. Frames are always emitted, even
when everyone in them was rejected — the frame happened, so duration and frame
totals stay honest.

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
identity (too faint to start a track, matched nothing). A `track_id` that
appears in fewer than 3 consecutive lines was rejected by the filter and
contributed nothing to `result.json`.

`--no-tracker` disables tracking entirely and reproduces pre-#32 numbers, where
every detection counted. It exists for baseline comparison and as a rollback —
not for production.
