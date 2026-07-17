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
| `--classifier heuristic|vlm|mlp` | Activity mode; CLI default is heuristic, Docker default is VLM |
| `--model PATH` | Fixed-square YOLO11-pose ONNX model |
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

The deployed model is fixed-640 YOLO11s-pose. The loader accepts fixed square YOLO11 pose exports and validates the standard `[1,56,N]` output.

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

This path is experimental. Issue #86 found no eligible fixed-640, fixed-1280, or focused-ROI software arm for the bending camera. Production remains fixed-640 while #88 waits for a station-framed camera stream. See [the benchmark result](../benchmark-results/issue-86/README.md).

## `result.json` schema 6

Top-level fields:

- `schema_version`;
- video duration/frame/person summary;
- all four activity `person_minutes` buckets;
- one-minute `timeline` bins;
- annotated base64-JPEG `keyframes`;
- `zones[]` with posture totals, presence/work/absence, and conversation;
- `shift` windows/breaks/excluded duration or `null`;
- classifier/model `diagnostics`.

Presentation strings and layout are deliberately absent; the platform owns rendering.

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
