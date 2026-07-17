# Bending pilot v1 annotation methodology

## Provenance and station boundary

- Annotation date: 2026-07-16.
- Annotator: Codex, with the user confirming that the supplied factory view is
  the bending station. The exact camera used here is the evidence-backed pilot
  camera from issues #83/#86: `a7b76f41-a106-4019-811d-db6f9576d0dc`.
- Native frame size: 3840×2160.
- The surviving #83 artifacts were checksummed before reuse. OpenCV template
  matching placed the earlier 1280×720 four-worker reference tile at exactly
  `[1280, 0]–[2560, 720]` in the native frame (normalized correlation
  `0.943313`). This recovered rectangle, rather than newly guessed geometry,
  is the semantic `bending-1` zone in `zones.json`.
- The focused inference crop expands that rectangle by a fixed 160 pixels and
  clips to the frame, producing `[1120, 0]–[2720, 880]`. The margin is fixed
  before scoring, matches the previously documented project example, and
  retains workers entering at the zone edge without changing semantic
  membership.
- Reference artifact SHA-256 values:
  - native 3840×2160 frame: `0a23e024a0044511bddcbe3bcd12a40e28e7f390a2caf1f02a4226ec24e73d25`
  - annotated 1280×720 tile: `f8a479e5bf5b18dc3fb3424ade9e066e67eada9967519f8c550d0dc9d1d61970`
  - letterboxed 1280×720 tile: `33dbe04205c2d1402d6689aaa2ee752d5c27cce0fd7ec34c7be7871b9c4967f4`

## Recording windows and sampling

Three one-minute rolling-buffer chunks were copied before overwrite. Their
midpoint overlays show 2026-07-16 21:18:46, 21:38:49, and 21:58:49,
respectively. The approximately 20-minute spacing covers the available
occupancy variation under the same night lighting: one worker with table
occlusion, two workers with rack occlusion, and a worker crossing the zone
boundary.

| Window | Source clip SHA-256 | Selected frames |
|---|---|---:|
| `window-1` | `65f70eea1d204c62b19e586107aacda1933b7bd458029ad168073ef725fc1b41` | 20 |
| `window-2` | `7e0772cf7d00223367adb066ccb23f3c2b5288fb6b852c4fb722a74f64b87e35` | 20 |
| `window-3` | `0b0ef1584f2ff9b778391fa2754d5439b5e5afe541504ed3e100a2811b73fae0` | 20 |

Frames were sampled at offsets `0, 3, …, 57` seconds with ffmpeg's `fps=1/3`
filter. Every arm reads the same 60 full-resolution JPEGs. The materialized
JPEGs were re-encoded at ffmpeg quality 12 to reduce the immutable fixture
asset size; dimensions and pixel coordinate space remain 3840×2160. Per-frame
SHA-256 values are locked in `manifest.json`.

## Annotation rules and review

- Boxes are full-frame `xyxy` coordinates around each visible person.
- A person is included only when the midpoint of the box's bottom edge is
  inside or on the recovered semantic rectangle. A person visible in the
  160-pixel inference margin but standing outside the semantic rectangle is
  intentionally excluded.
- Occluded people are boxed to the visible body extent plus the directly
  inferable continuous extent behind thin rebar/rack occlusion. Fully hidden
  people are not annotated.
- Empty `persons` arrays in window 3 were manually checked; they represent
  genuine out-of-zone foot points during entry/exit, not assumed negatives.

Three prelabel views were used as drawing aids: full-frame fixed 1280,
focused-ROI fixed 640, and focused-ROI fixed 1280. The last is annotation-only
and is not a candidate arm. To avoid making any candidate its own truth, every
selected frame was then visually reviewed at the expanded ROI resolution in
15 four-frame contact sheets. The review retained one in-zone person in every
window-1 frame, two in every window-2 frame, and 14 total in-zone instances in
window 3 (74 boxes overall).

Manual corrections recorded during review:

- `window-1-frame-020`: the focused-1280 aid missed the visible worker; the
  reviewed focused-640 box was retained.
- `window-2-frame-005`: two nested boxes described one occluded worker; the
  larger continuous-body box was retained and the duplicate removed.
- `window-2-frame-012` and `window-2-frame-014`: the high-visibility worker
  behind the rack was manually added; both foot points remain inside the zone.
- Window 3 boundary frames were reviewed individually against the yellow zone
  edge; detections with bottom midpoint `x < 1280` or `y > 720` were excluded.

The benchmark loader independently verifies all hashes, image bounds, unique
IDs, window count, frame count, and in-zone foot points before loading CUDA.

## Distribution

The large `frames/` and `clips/` payload is distributed as an immutable GitHub
release asset. Git tracks this methodology, `manifest.json`, `zones.json`, and
the asset checksum/download instructions; the binary directories are ignored.
