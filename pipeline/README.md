# Analysis Pipeline

```text
MP4 chunks
  → ffmpeg frames at 1 fps
  → YOLO11-pose
  → OSNet person tracker
  → minimum-track filter
  → activity classifier
  → optional zones / shift / workstation modes
  → result.json schema 6
```

The canonical full-video output is structured JSON for platform rendering. Standalone HTML remains available through `--format html` for local debugging only.

## CLI

```bash
uv run python -m pipeline.analyze input.mp4 \
  --output result.json \
  --classifier vlm \
  --dump-detections detections.jsonl
```

Useful switches:

| Option | Meaning |
|---|---|
| `--classifier heuristic|vlm|mlp|station` | Activity mode; CLI default is heuristic, Docker default is VLM. `station` is the per-zone chronometraż (#123) and needs `--zones` |
| `--station-head PATH` | Station head ONNX (station mode only) |
| `--station-card PATH` | Its generated model card — the rectangle the run gates on, and the time ratios it quotes |
| `--backbone PATH` | Frozen DINOv2 backbone, shared by every station |
| `--model PATH` | YOLO11-pose ONNX model (square or non-square, e.g. 640×640 or 1280×736) |
| `--reid-model PATH` | OSNet model used by default tracking |
| `--no-tracker` | Reproduce pre-#32 behavior; not for production |
| `--max-track-age SECONDS` | Retirement window; default 120 |
| `--zones zones.json` | Zone, shift, rules, and optional inference-ROI config |
| `--dump-detections PATH.jsonl` | Archive raw per-frame evidence |
| `--format html` | Write legacy local HTML instead of canonical JSON |

Single-frame CUDA smoke:

```bash
uv run python -m pipeline.analyze input.mp4 \
  --timestamp 12.5 \
  --model models/yolo11s-pose.onnx
```

## Pose inference

The default deployed model is 640×640 YOLO11s-pose; a non-square `1280x736` export is a per-camera option (#100/#109). The loader accepts both square and non-square YOLO11 pose exports (letterboxed, aspect-preserving) and validates the standard `[1,56,N]` output.

- bbox: rows 0–3;
- confidence: row 4;
- 17 COCO keypoints × `(x,y,visibility)`: rows 5–55.

Frames are letterboxed to the model input. Bboxes and keypoints are mapped back into original-frame pixels before tracking, zone assignment, annotation, or reporting.

Inference is CUDA-only. The loader preloads NVIDIA libraries, creates the ONNX Runtime session, and verifies the session actually uses `CUDAExecutionProvider`.

## Tracking and count filtering

Each detection receives a stable `track_id` while the same person's appearance remains matchable.

- Association uses OSNet appearance embeddings rather than bbox IoU. At 1 fps, position overlap is too weak for reliable identity.
- A track must appear at least three times within five processed frames before it contributes to `result.json`.
- The filter delays frames while a track proves persistence, but it still emits empty confirmed frames so duration remains honest.
- `detections.jsonl` taps the stream before filtering and therefore contains both counted and rejected detections.

`max_track_age_s` defaults to 120 seconds. A return after retirement receives a new ID.

### Known identity limitation

OSNet body appearance is not enrolled identity. No representative benchmark calibrated long-gap re-match and false-merge rates, and issue #89 was closed as not planned. Do not quote estimated percentages, increase track age to chase returners, or silently merge tentative matches.

The safety rule is split over merge: a visible identity split creates an absence gap; merging two workers silently corrupts time totals.

Face recognition and cross-video/cross-camera identity remain out of scope.

## Classifiers

### VLM

The deployed Docker default uses Qwen2.5-VL-3B for stationary posture and bbox displacement for walking. The VLM label is computed once per frame and shared by non-moving detections; walking remains a per-detection decision.

### Heuristic

The supported baseline uses geometric keypoint rules and displacement smoothing. It remains available for comparisons and rollback.

### Experimental MLP

The MLP classifies each detection independently from a frozen feature schema, then smooths by `track_id`. The loader verifies artifact checksum, metadata, feature schema, class order, and CUDA provider.

### Station (issue #123)

`--classifier station` answers a different question from the other three: not who is in the hall and what they are doing, but how much of the session at one workstation went on each category of work. The client accepted the **zone** as the unit of measurement rather than a person, so the path is the person pipeline with three components removed rather than a fourth added:

```
zone crop from the native frame at a fixed stride
  → frozen DINOv2 backbone
  → ~1 MB station-specific temporal head
  → per-sample category → intervals → totals
```

No pose model, no OSNet, no VLM — asserted by test, because a stray import costs the VRAM and load time the mode exists to avoid rather than failing anything visible.

Requires `--zones` naming exactly one zone with `rules.type: "station"`. The zone polygon's bounding box **is** the crop, taken from the native frame and never from a downscaled copy, and it must equal the card's `station.zone_native_px` — a head fed a rectangle it was not fitted on returns confident logits over pixels it has never seen, and nothing downstream would report that.

Three artefacts, all pinned in `setup-models.sh` and baked into the image: the backbone (identical at every station, shipped once), the head, and its generated model card. The card is load-bearing, not documentation: it carries the rectangle above, the sampling stride, the collapse from the head's seven classes to the four delivered ones, and the **measured time ratio** printed beside every total. A card whose time ratios have a hole in them stops the run rather than yielding a total that looks exact.

Selected per camera by `classifier: "station"` in the mounted `zones.json`, resolved at container start exactly like `pose.mode`.

It is not production-approved. The frozen #33 test measured 62.67% for MLP versus 93.33% for VLM, with regressions on both held-out geometries and Film 1. See [the full evaluation](../docs/mlp-classifier-eval.md).

## Zone configuration

```json
{
  "recording_start": "2026-07-16T06:00:00+02:00",
  "shift": {
    "timezone": "Europe/Warsaw",
    "windows": [["07:00", "15:00"]],
    "breaks": [["11:00", "11:20"]]
  },
  "zones": [
    {
      "id": "bending-1",
      "name": "Giętarka 1",
      "polygon": [[1200, 500], [2600, 500], [2600, 1900], [1200, 1900]],
      "rules": {
        "type": "bending",
        "work": {"min_move_px": 40},
        "conversation": {"proximity_px": 150},
        "absence": {"flag_after_s": 180}
      }
    }
  ]
}
```

### Assignment

A detection belongs to a zone when the midpoint of its bbox bottom edge lies inside the polygon. Edge/vertex points count as inside. The first matching zone wins; outside detections keep `zone_id: null`.

### Shift gating

`recording_start` maps video time to wall clock. Windows and breaks are recurring half-open intervals. Only frames inside a working window and outside every break contribute to aggregation. Use an IANA timezone when the recording could cross a DST transition.

### Bending modes

`rules.type` defaults to `bending`; it is currently the only implemented ruleset.

- `presence`: the longest-dwelling in-zone track becomes the anchored worker;
- `absent`: gaps between anchored-worker presence runs, flagged past `flag_after_s`;
- `work`: anchored-worker foot-point motion above `min_move_px`;
- `conversation`: at least two close, stable, low-movement tracks.

The report retains the exact intervals and totals. Thresholds are station-specific configuration, not universal defaults to promote without validation.

## Focused inference ROI

An optional top-level block focuses the single pose call on one configured zone:

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

The crop is clipped to the frame and all outputs are translated back into full-frame pixels. The margin must be explicit, finite, and non-negative.

This path is experimental. Issue #86 found no eligible fixed-640, fixed-1280, or focused-ROI software arm for the bending **pilot** camera; #88 (closed, deferred) needs a client-provided station-framed stream before that pilot resumes. For deep-hall cameras, per-camera 1280×736 input and `hybrid` tiling are the shipped detection levers (#100/#109/#110/#111); 640×640 full-frame stays the default. See [the benchmark result](../benchmark-results/issue-86/README.md).

## `result.json` schema 6

Top-level fields:

- `schema_version`;
- video duration/frame/person summary;
- all four activity `person_minutes` buckets;
- one-minute `timeline` bins;
- annotated base64-JPEG `keyframes`;
- `zones[]` with posture totals, presence/work/absence, and conversation;
- `shift` windows/breaks/excluded duration or `null`;
- classifier/model `diagnostics`;
- `station_activity` — **present only in station mode** (issue #123).

Presentation strings and layout are deliberately absent; the platform owns rendering.

### `station_activity`

Additive: the key is absent, not `null`, in every other mode, and a golden-file test holds those modes byte-identical. `schema_version` stays at 6 for the same reason — the key can only appear from a mode that did not exist at 6, so nothing an existing consumer reads changes, and the platform renders the section on its presence rather than on a version (gpu-exchange#210).

Per zone: the `intervals`, a `categories[]` entry per delivered category, `coverage`, `session_s`, the `stride_s` and `boundary` convention, the `abstention` name, and the `model` block naming the station, the version, the head's sha256 and the recordings it was trained on.

Three things make it hard to read the totals as more exact than they are, and none are optional:

- **`time_ratio` sits inside each category object**, beside its `total_s`. Not a parallel map: a renderer can drop a key from a map without noticing, and this fixture already produced a baseline that met a 99.4% recall bar while reporting 2.18× the real welding time.
- **`coverage` is samples predicted over samples possible**, and every `share` divides by what the session *could* have produced. A gap that silently leaves the denominator flatters every number above it.
- **`nierozpoznane` keeps its own category row** at whatever value it has, including zero. It is neither work nor downtime; folding it into the collective bucket would convert unknown time into measured time.

Interval boundaries land at the **sample midpoint**, the convention `benchmarks/activity/tools/evaluate_arms.py` folds the annotation under — checked against that implementation by test, because every quoted `time_ratio` was measured against intervals folded that way.

## Detection audit archive

Each JSONL line represents one processed frame. Persons carry bbox, confidence, keypoints, activity, `track_id`, and zone assignment where applicable.

Use it to answer:

- what YOLO detected before the persistence filter;
- why a detection was not counted;
- which track/zone/activity reached aggregation;
- whether a later report change came from detection, tracking, classification, or presentation.

The archive cannot be reconstructed from the bounded keyframe buffer after the run, so enable it during any validation whose claims require per-frame evidence.

## Measured references

- [#86 pose-resolution benchmark](../docs/POSE_RESOLUTION_BENCHMARK.md) and [measured no-winner result](../benchmark-results/issue-86/README.md)
- [#34 activity MLP frozen evaluation](../docs/mlp-classifier-eval.md)

These artifacts state hardware, image/model hashes, raw timing, VRAM samples, assumptions, and gate outcomes. Do not turn them into unscoped performance promises for different videos or camera geometries.
