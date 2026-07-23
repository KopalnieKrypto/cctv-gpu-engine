# Magazyn hall v1 annotation methodology

> **STATUS: UNANNOTATED.** Every `persons` array in `manifest.json` is empty.
> The frames, clips, hashes and manifest are prepared; the annotation pass
> described below has **not** been performed. The fixture cannot score anything
> until it has been. See "Annotation pass — still to do".

## Why this fixture exists

`bending-pilot-v1` and this fixture come from **the same physical camera**:
`a7b76f41-a106-4019-811d-db6f9576d0dc`. They differ in *scope*, not in hardware.

- `bending-pilot-v1` annotates only inside the recovered station rectangle
  `[1280, 0]–[2560, 720]` — the upper-middle band of the frame, which is the
  **far field**: the smallest, hardest targets in the scene.
- This fixture annotates the **whole 3840×2160 frame**, which is what a
  production `result.json` actually reports on.

That distinction matters for reading #86. Its `0 TP / 74 FN` full-frame-640 result
was measured *inside the station rectangle*, so it characterises the hardest
region of the frame rather than the scene as a whole. Production reports mix that
far field with a near field where detection demonstrably works — in
cctv-gpu-engine#97, the near-camera worker was detected reliably while mid- and
far-field workers were missed. Neither number is wrong; they answer different
questions. This fixture exists to answer the production one.

## Provenance

- Source: production task `ce475156-f85a-4264-a23d-052105a50ec1`, tenant
  `client-production@antra.one`, camera `magazyn`.
- Source object: `chunk_000.mp4`, 2 396 924 949 bytes, HEVC 3840×2160,
  3540.436 s, from the platform's R2 under the task's `appliance-uploads/` prefix.
- Recording anchor: video `t=0` corresponds to wall clock **12:44:06** on
  2026-07-22. Derived from the burned-in overlay on two independent
  `result.json` keyframes — `t=755` reads 12:56:41 and `t=3538` reads 13:43:04,
  and both give the same `t=0`. Every frame carries this overlay, so wall clock
  is independently checkable on each image.
- Native frame size: 3840×2160. Frames are extracted at full resolution with no
  rescaling, so annotation coordinates are native pixels.

## Window selection

The task's shift config gates 13:00–13:30 as a break, which maps to video
`t = 954 … 2754`. Windows were placed inside the two **shift-active** stretches
either side of it, at fixed round offsets:

| Window | Source offset | First frame (overlay) | Last frame (overlay) |
|---|---:|---|---|
| `window-1` | `t = 300 s` | 12:49:07 | 12:50:04 |
| `window-2` | `t = 2820 s` | 13:31:07 | 13:32:04 |
| `window-3` | `t = 3420 s` | 13:41:07 | 13:42:04 |

**Selection is config-derived, never detector-derived.** The only input was the
shift schedule (which stretches of footage the platform was asked to analyse).
No timeline, `person_minutes`, keyframe list or other detector output influenced
which windows or which frames were taken — the `result.json` keyframe set in
particular was *not* used, since the engine selects those precisely because
detection succeeded, which would bias the fixture toward the arm under test.

Each clip is cut with `-c copy`, so ffmpeg snaps to the nearest preceding
keyframe; all three land 1 s after the requested offset, consistently. The cut
leaves dangling references in the first GOP, which ffmpeg reports and recovers
from — all 60 extracted frames were verified to decode cleanly, to be 3840×2160,
and to fall in a 1.09–1.23 MB band with no outliers.

## Sampling

20 frames per 60-second window at `fps=1/3` (offsets 0, 3, …, 57 s), JPEG
quality 12 — identical parameters to `bending-pilot-v1`, so the two fixtures are
directly comparable. 60 frames total. Per-frame SHA-256 is locked in
`manifest.json` and `assets.sha256`.

## Scoring zone

`zones.json` defines a single zone `hall-1` covering the exact frame rectangle
`[0,0]–[3840,2160]`, with `inference_roi.margin_px = 0`.

The benchmark requires an `inference_roi` to derive its scoring zone, and it
verifies that every annotated person's foot point falls inside that zone. A
whole-frame zone makes "in zone" mean "in frame", which is the question this
fixture asks. On-boundary points count as inside (`Zone.contains`), so a person
cut off at the frame edge is still valid.

## Annotation pass — still to do

**Rules**

- Boxes are full-frame `xyxy` in native 3840×2160 pixels.
- Box **every visible person in the frame** — there is no semantic sub-region
  here, unlike `bending-pilot-v1`. The people the current detector misses are the
  entire point of this fixture; omitting them would measure nothing.
- Occluded people: box the visible extent plus the directly inferable continuous
  extent behind thin rebar/rack occlusion. Fully hidden people are not annotated.
- An empty `persons` array is a legitimate annotation, but it must be the result
  of looking and finding nobody — record which frames those were.

**The one rule that cannot be relaxed**

Do **not** prelabel with any candidate arm (shipped 640, 640×384, 1280×736, or
square 1280). Doing so makes the arm its own ground truth and the measurement
becomes circular — the failure `bending-pilot-v1` guarded against with its
contact-sheet review pass.

If a drawing aid is needed, it must be a configuration that is **not** a
candidate — e.g. a tiled or heavily-cropped pass used for annotation only — and
every aided frame must still be reviewed by eye at full resolution. Record any
aid used and any manual correction made, as `bending-pilot-v1`'s methodology does.

**Expected effort**

60 frames at 4K, whole-frame, with deliberate attention to small mid- and
far-field figures. `bending-pilot-v1` produced 74 boxes over 60 frames within a
narrow rectangle; this fixture covers roughly six times the area, so expect
substantially more boxes per frame and a correspondingly longer pass.

**On completion**

Fill `persons` in `manifest.json`, record the annotator, date, per-window box
counts and any corrections in this file, then flip the STATUS banner at the top.
The benchmark loader independently verifies hashes, image bounds, unique IDs,
window count, frame count, and in-zone foot points before CUDA loads.

## Distribution

`frames/` and `clips/` are gitignored and distributed as an immutable release
asset — see `ASSET.md`. Git tracks this file, `manifest.json`, `zones.json` and
`assets.sha256`.
