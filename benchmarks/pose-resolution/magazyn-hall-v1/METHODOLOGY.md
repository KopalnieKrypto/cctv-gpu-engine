# Magazyn hall v1 annotation methodology

> **STATUS: ANNOTATED AND HUMAN-CONFIRMED (2026-07-23).** 296 people across 60
> frames, every frame reviewed by eye and every count confirmed. See
> "Annotation pass — as performed".

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

## Annotation pass — as performed

**Annotator:** Claude Opus 4.8, systematic 12-tile pass.
**Reviewer:** Tomasz Kowalczyk — full visual review of all 60 frames.
**Date:** 2026-07-23.

### Why tiles, and why every tile

An earlier ad-hoc pass cropped only the regions where people had appeared in a
previous frame. It missed a welder at `[925, 62, 974, 142]` in
`window-1-frame-002` entirely, because that region was never magnified — the
crop had been chosen from `frame-001`'s occupancy, which is a selection bias on
the very population the fixture measures.

The replacement protocol is exhaustive and identical for every frame: a 4x3 grid
of 960x720 tiles, each upscaled 2x to 1920x1440 (`review/tiles/`), and **every
tile opened for every frame** — including tiles that look like empty floor. At
2x a 31 px far-field worker renders at 62 px, which is what makes them findable.
One agent per frame, each reporting full-frame coordinates via
`frame_x = 960*C + tx/2`, `frame_y = 720*R + ty/2`.

No detector, pose model, or YOLO was run at any point. That prohibition was
stated in every agent prompt with its reason: this fixture measures those
detectors, so using one would make a detector its own ground truth.

### Human review

The annotation pass produced 343 candidate boxes. Every frame was then reviewed
by eye at full resolution, frame by frame, with a per-frame count and per-box
adjudication. The review:

- **removed 48 boxes** — overwhelmingly the two recurring non-person objects below
- **added 1 box** the pass missed entirely — the bench worker in
  `window-1-frame-011` at `~[2690,1335,2850,1540]`, corroborated by his presence
  in `frame-010` and `frame-012`
- **corrected 2 counts** the reviewer initially mis-stated and revised on
  re-inspection (`frame-005`, `frame-008`)

Final: **296 people**, mean 4.93/frame — window-1 114, window-2 105, window-3 77.

Every box in `manifest.json` is human-confirmed. Agent confidence ratings were an
input to that judgement and are deliberately **not** carried into the manifest:
retaining them would imply the fixture is unsure about boxes a human confirmed.

### Recurring shapes, and what the review settled

Three shapes recur across many frames and account for most of the disagreement.
Recording them because they will resurface in any future annotation of this
camera:

| Shape | Location | Frames | Verdict |
|---|---|---:|---|
| A | `~[825,0,860,78]`, top edge | 20 | **A person exactly when the pass rated it `high`.** Rejected in w2 f001–f012 (rated `medium`/`low`), confirmed in f013–f019 (rated `high`). Someone moves into that spot mid-window; before that the shape there is not a person. |
| B | `~[506,250,530,308]`, tall thin turquoise | 19 | **Never a person.** Zero confirmations across all appearances. Equipment. |
| C | `~[595,64,633,165]`, far field | 5 | **Always a person.** Confirmed in w3 f001–f005 regardless of whether the pass rated it `low` or `medium` — a genuine far-field worker the pass consistently under-rated. |

B and C are mirror images and together are the clearest statement of what this
fixture is for: at far-field scale the annotation pass both over-reported
equipment and under-rated real workers, and only human review separated them.

A bright arc-weld flare is **not** a person unless a body is visible beside it
(`window-1-frame-008`, `[1410,145,1465,235]` rejected). The confirmed welder in
`window-1-frame-002` has a discernible dark silhouette next to the flare; that is
the distinction.

### Rules applied

- Boxes are full-frame `xyxy` in native 3840x2160 pixels.
- Every visible person is boxed, anywhere in the frame. **No minimum size** — the
  far-field figures are the entire point, and a size floor would delete the
  finding the fixture exists to measure.
- Occlusion: visible extent plus the directly inferable continuous body behind
  thin rebar/rack occlusion. Fully hidden people are not annotated.
- Frame edge: the visible portion is boxed; on-boundary points count as in-zone.
- Two boxes were dropped in validation for failing person-plausible geometry
  (18x16 and 22x18 px — too small to be a person even at far-field scale).

### Known limit

Box precision on the smallest far-field figures is approximately +/-3 native
pixels. At the benchmark's IoU 0.5 threshold this is not material for figures of
~31x84 px, but it is the reason a future arm should not read fine-grained
localisation differences on that population as signal.

## Distribution

`frames/` and `clips/` are gitignored and distributed as an immutable release
asset — see `ASSET.md`. Git tracks this file, `manifest.json`, `zones.json` and
`assets.sha256`.
