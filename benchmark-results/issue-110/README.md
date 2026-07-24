# Issue #110 measured result — native-resolution tiling

Tiling reaches the band #101 proved no full-frame arm can. On the same 296-person
`magazyn-hall-v1` ground truth, the **80–120 px native band — 90 of 296 people,
the largest bucket — went from 0.0% at every full-frame arm to 23.3%**, and it
did so on **710 MiB** of VRAM against the shipped 1280×736 arm's 8,122 MiB,
because every tile is model-input-sized no matter how many tiles a frame needs.

The cost is real: 16 tiles per 4K frame is 3,358 ms of pose per frame (202
min/h, pose-only), 4.7× the shipped resolution arm. And the grid-only arm
*regresses* the ≥260 px band (73.5% → 49.0%) because a large near-field person is
split across tile seams and no single tile frames them whole — exactly the case
#110 flagged for an optional full-frame pass. This measures the lever; it does
not yet productionize it.

## Arms

- **`tiled_1280x736` (without zones)** — tile the whole frame (16 tiles). ✅
- **`tiled_zones_1280x736` (with zones)** — tile only inside camera `a7b76f41`'s
  four authored ROI polygons (`strefa-1..4`, mirrored verbatim from the platform
  `analysis_config`; 12 of 16 tiles). ✅ **Identical quality to the grid arm at
  75% of the pose cost** — the four skipped tiles held no detected person, so
  zone-focusing bought a 25% compute saving at zero recall/precision loss.
- **`tiled_fullframe_1280x736` (hybrid)** — the 16-tile grid plus one whole-frame
  1280×736 pass, merged (17 pose calls/frame). ✅ **Pareto-dominates the grid
  arm** — better on every band and every aggregate metric, for +6% cost — because
  the full-frame pass frames the large near-field people the grid splits, and the
  IoS merge keeps the whole-body box over the seam partials. The best arm measured.

## Aggregate — `tiled_1280x736` vs #101's full-frame arms

All arms scored on `benchmarks/pose-resolution/magazyn-hall-v1` — 60 manually
annotated 3840×2160 frames across 3 windows, **296 human-confirmed people**,
whole-frame scoring zone, confidence 0.25, NMS IoU 0.45, TP IoU 0.5. Tiling adds
overlap 0.2, Intersection-over-Smaller merge threshold 0.6.

| Arm | TP/FP/FN | Precision | Recall | F1 | Pose p50/p95 | Peak VRAM | Detections |
|---|---|---:|---:|---:|---:|---:|---:|
| full frame → 640 (shipped) | 21/12/275 | 63.6% | 7.1% | 12.8% | 115/116 ms | 7,868 MiB | 33 |
| full frame → 1280×736 | 105/13/191 | 89.0% | 35.5% | 50.7% | 232/234 ms | 8,122 MiB | 118 |
| **tiled 1280×736 (grid)** | **134/55/162** | **70.9%** | **45.3%** | **55.3%** | **3358/3363 ms** | **710 MiB** | **214** |
| **tiled 1280×736 (zones)** | **134/55/162** | **70.9%** | **45.3%** | **55.3%** | **2523/2530 ms** | **710 MiB** | **214** |
| **tiled + full-frame (hybrid)** | **145/53/151** | **73.2%** | **49.0%** | **58.7%** | **3568/3577 ms** | **710 MiB** | **223** |

Tiling has the highest recall and F1 of any arm measured, at the lowest VRAM and
by far the highest pose cost. Precision drops to 70.9% (from 89.0%): more tiles
in cluttered regions raise boundary false positives, and IoS-0.6 dedup does not
catch every cross-tile duplicate — the precision risk #110 named.

The two tiling arms produce **bit-identical detections** (214, same TP/FP/FN,
same per-band recall). The with-zones arm tiled 12 of the 16 tiles — only those
intersecting the four authored `strefa` bboxes — and the four it skipped held no
detected person, so it reached the same numbers on 25% less pose (2523 vs 3358
ms/frame). On this camera the authored zones already bound where people are, so
zone-focusing is a free compute saving, not a recall trade. On a camera whose
zones covered less of the occupied floor it would trade recall for tiles; here it
does not.

## Recall by ground-truth person height — the headline

| GT height (native px) | people | `baseline_640` | `full_frame_1280x736` | `tiled` (grid/zones) | **`tiled_fullframe`** |
|---|---:|---:|---:|---:|---:|
| < 80 | 32 | 0.0% | 0.0% | 3.1% | **3.1%** |
| **80–120** | **90** | **0.0%** | **0.0%** | **23.3%** | **23.3%** |
| 120–180 | 34 | 0.0% | 11.8% | 35.3% | **38.2%** |
| 180–260 | 91 | 1.1% | 71.4% | 83.5% | **86.8%** |
| ≥ 260 | 49 | 40.8% | 73.5% | 49.0% | **63.3%** |

The hybrid keeps the tiles' small-band wins (`< 180 px` unchanged or better) and
buys back the large-band loss: `≥ 260 px` goes 49.0% → 63.3%, and `180–260 px`
83.5% → 86.8%. It does not fully reach the full-frame arm's 73.5% at `≥ 260`
(some whole-body boxes are still out-competed by a higher-confidence tile partial
in the merge) — a dedup-tuning lever, not a ceiling.

Two things the numbers say:

- **The 80–120 px band is no longer dead.** A native 1280×736 tile letterboxes at
  scale 1.0, so a 100 px person stays 100 px at model input — well above the
  ~60 px floor #101 found — and tiling recovers 21 of the 90 people every
  full-frame arm missed. The 120–180 band roughly triples (11.8% → 35.3%) and
  180–260 climbs 71.4% → 83.5% for the same reason.
- **The grid-only arm loses big people.** ≥260 px drops 73.5% → 49.0%: a person
  that tall near the camera is cut by tile seams, and the seam-clipped partials
  never reach IoU 0.5 with the whole-body GT box. A full-frame pass merged into
  the tile detections would recover them — the #110 "optionally add a full-frame
  pass" note, now measured as necessary, not optional, for near-field scenes.

## Cost — detector-isolated (#110 methodology)

`classifier=heuristic`-equivalent: the tiling command runs pose only, no VLM. The
per-hour figure is `mean(pose_wallclock_s) * fps * 3600 / 60` at 1 fps, and each
`pose_wallclock_s` sample already sums the frame's 16 tiles.

| Arm | Tiles/frame | Pose/frame | Pose min/h | Peak VRAM | Detections (drives VLM) |
|---|---:|---:|---:|---:|---:|
| full frame → 1280×736 | 1 | 232 ms | ~14 (pose only) | 8,122 MiB | 118 |
| tiled 1280×736 (grid) | 16 | 3358 ms | 202 | 710 MiB | 214 |
| tiled 1280×736 (zones) | 12 | 2523 ms | 152 | 710 MiB | 214 |
| tiled + full-frame (hybrid) | 17 | 3568 ms | 215 | 710 MiB | 223 |

VRAM is the surprise win — 11× smaller than the full-frame 1280×736 arm — because
tiling never holds a >1280 px input tensor; it holds twelve-to-sixteen 1280×736
ones in sequence. The cost that bites is wall-clock: 152–202 min of pose per
video hour makes the tiling arms ~11–14× the shipped detector's pose budget. The
with-zones arm is the cheaper of the two at identical quality (152 vs 202 min/h),
because the authored zones drop four empty tiles per frame. The VLM adds on top,
scaling with the 214 detections found (read separately, not run here).

## Reproducibility evidence

- Host `cctv-vps` (`mvp-serwer`), NVIDIA GeForce RTX 5070 **GPU 1**, 12,227 MiB,
  driver 595.45.04. GPU 0 excluded (SGLang staging LLM, 9,222 MiB) — the same
  exclusion #86/#101 made. Run on the host `uv` GPU env (onnxruntime-gpu,
  CUDAExecutionProvider), not the container; per-PID VRAM from `nvidia-smi
  --query-compute-apps` matches the host process PID directly.
- Model `models/yolo11s-pose-1280x736.onnx`, SHA-256
  `7ee0fcd86efdf953678a92649a682b4369ee5372ebf2623876602286c37731e9` — byte-
  identical to the arm #101 measured and the export baked into the image
  (`gpu-service/Dockerfile`), so this is the same detector at tile scale.
- Fixture `magazyn-hall-v1` at `89a8287`; `manifest.json` re-verifies every frame
  SHA-256 before CUDA loads.
- Tile grid: 4 columns × 4 rows = 16 native 1280×736 tiles per 3840×2160 frame at
  overlap 0.2 (columns step 1024 px, rows 589 px, last tile flush to the far
  edge). Merge is greedy highest-confidence with IoS ≥ 0.6 suppression. The
  with-zones arm keeps the 12 tiles that intersect a `strefa` bbox.
- With-zones ROI: `benchmarks/pose-resolution/magazyn-hall-v1/zones-roi.json` —
  the four authored polygons (`strefa-1..4`) of camera `a7b76f41`, copied verbatim
  from the platform `analysis_config` (`restrict_to_zones: true`, `pose 1280x736`)
  on 2026-07-24. Scoring stays whole-frame (`zones.json` `hall-1`), so an
  out-of-zone person is an honest miss, not silently excluded.
- The hybrid arm (`tiled_fullframe_1280x736`) adds one whole-frame 1280×736 pose
  call (offset 0,0, already full-frame coords) to the 16-tile pool before the
  merge — 17 pose calls/frame. Same model, same fixture, same whole-frame scoring.
- `tiled_1280x736.json`, `tiled_zones_1280x736.json`, and
  `tiled_fullframe_1280x736.json` retain raw per-frame detections, per-frame pose
  timing, `recall_by_height`, `detector_cost`, and peak VRAM. Runbook:
  `docs/POSE_RESOLUTION_BENCHMARK.md` § "Native-resolution tiling arms".

## What this redirects the work toward

1. **Tiling is the band-80–120 lever, confirmed.** No full-frame arm on 12 GB
   reaches it; tiling does, at 23.3% and rising with the larger bands.
2. **The hybrid (tiles + full-frame pass) is the arm to carry forward.** It
   Pareto-dominates grid-only — every band and every aggregate metric better for
   +6% cost — and fixes the near-field regression (≥260 px 49% → 63%). Grid-only
   is obsoleted by it.
3. **The real production choice is two modes, not four.** `full_frame_1280x736`
   (cheap, ~14 min/h, high precision, but blind below 180 px) versus `hybrid`
   (sees the small bands and the large ones, ~15× the pose cost, ~73% precision).
   Zones is an orthogonal compute optimization that layers on either. Grid-only
   and zones-only need not be exposed.
4. **Precision and cost are the remaining gates.** 73.2% precision (cross-tile
   dedup) and 215 min/h pose. Next levers: better dedup to lift precision and to
   close the ≥260 gap to full-frame's 73.5%; a coarser overlap or larger tile to
   cut the tile count. Then wire the winning mode end-to-end (the pipeline does
   not tile yet) behind a per-camera config field, like #109 did for resolution.
