# Surveillance Video Activity Analysis — As-Built Specification

Status: current as of 2026-07-24
Scope: batch CCTV analysis, bare-metal client appliance, standalone R2 worker, and GPU Exchange REST integration

This document describes the implemented system. Historical design intent remains in `IMPLEMENTATION_PLAN.md` and `plans/`; those files are not the current runtime contract.

## 1. Product boundary

The system processes recorded MP4 chunks after capture. It does not perform live alerting.

Implemented outcomes:

- detect people and COCO pose keypoints with YOLO11s-pose;
- track people within a recording using OSNet appearance embeddings;
- classify each detection into `sitting`, `standing`, `walking`, or `running`;
- optionally assign people to configured workstation zones and gate results to shift windows;
- derive bending-station presence, absence, work, and conversation intervals;
- produce canonical schema-versioned JSON for platform rendering;
- retain per-frame detection evidence when requested or when the standalone worker archives it.

Non-goals:

- live RTSP monitoring or alerting;
- face recognition or enrolled identity;
- automatic identity across videos, cameras, or days;
- long-gap returner merging when the tracker is uncertain;
- activity classes outside the four-class posture vocabulary;
- unmeasured promotion of higher resolution, focused ROI, tiling, or experimental classifiers.

## 2. Architecture

### 2.1 Platform path

```text
camera ──RTSP──> bare-metal client appliance
                    │
                    ├── outbound register/heartbeat/task/snapshot API
                    ▼
               GPU Exchange platform ── presigned URLs ──> R2
                    │
                    └── gpu-agent ── POST /analyze ──> gpu-service REST :5003
                                                        │
                                                        └── result.json to presigned URL
```

The appliance holds camera credentials and a platform bearer token. It does not hold R2 credentials. The platform issues presigned upload URLs. The gpu-agent owns GPU-service container lifecycle and supplies presigned input/result URLs.

### 2.2 Standalone compatibility path

```text
R2 surveillance-jobs/{job_id}/status.json
        ▲                         │
        │                         ▼
dashboard :5000 <── gpu_service.worker ──> input chunks / result.json
```

`docker compose up` runs the polling worker and investor dashboard. This path uses `.env.gpu` R2 credentials and R2 `status.json` coordination.

### 2.3 Components

| Component | Responsibility |
|---|---|
| `pipeline/` | Frame streaming, pose, tracking, classification, zones, aggregation, JSON/HTML rendering |
| `client-agent/client_agent/` | Appliance UI, discovery, recording, rolling buffer, platform/presigned clients |
| `client-appliance/` | Root and user-mode systemd packaging |
| `gpu-service/gpu_service/worker.py` | Standalone R2 poll/claim/process/upload loop and dashboard lifecycle |
| `gpu-service/gpu_service/rest_server.py` | GPU-agent REST task contract on port 5003 |
| Cloudflare R2 | Video/result object storage; standalone job coordination |
| GPU Exchange | Appliance configuration, task dispatch, tenant boundaries, presigned URL issuance, report rendering |

## 3. AI pipeline

### 3.1 Frame extraction

- Input: one or more MP4 chunks.
- Sampling: 1 fps by default.
- Processing: streamed frame-by-frame through ffmpeg; a complete video is never loaded into RAM.
- Multiple chunks share one aggregator and receive monotonically offset timestamps.
- Progress callbacks fire within chunks and at chunk boundaries so the worker can update status and telemetry.

### 3.2 Pose model

| Property | Contract |
|---|---|
| Default model | `models/yolo11s-pose.onnx` |
| Input | `[1,3,H,W]`; default 640×640, plus a supported per-camera non-square `1280x736` export (#100/#109) |
| Output | `[1,56,N]`: bbox 0–3, confidence 4, keypoints 5–55 |
| Runtime | `onnxruntime-gpu`, `CUDAExecutionProvider` required |
| Confidence | 0.25 |
| NMS IoU | 0.45 |

Preprocessing preserves aspect ratio through letterboxing, converts to RGB float32 `/255`, then CHW plus batch dimension. Postprocessing reverses letterbox coordinates into original-frame pixels before downstream consumers.

`ort.preload_dlls(cuda=True, cudnn=True)` must run before session creation. The implementation verifies the created session provider list and raises on silent CPU fallback.

### 3.3 Person tracking

Tracking is enabled by default.

1. OSNet x0_25 produces an appearance embedding for each detected person crop.
2. The tracker assigns a stable `track_id` while evidence is sufficiently similar and recent.
3. A track must be observed at least three times in a five-frame window before it contributes to aggregation.
4. Raw `detections.jsonl` is written before the minimum-track filter, preserving rejected detections for audit.

`max_track_age_s` defaults to 120 seconds. The system intentionally prefers splitting an uncertain returner into a new track over merging two people. No calibrated long-gap re-match or false-merge rates exist; issue #89 was closed as not planned.

`--no-tracker` exists only for pre-#32 baseline reproduction.

### 3.4 Activity classifiers

All modes output exactly one of `sitting`, `standing`, `walking`, `running`.

| Classifier | Contract |
|---|---|
| `vlm` | Deployed Docker default. Qwen2.5-VL-3B classifies the frame-level stationary posture; per-detection displacement can override to walking. |
| `heuristic` | Supported baseline/rollback. Geometric keypoint rules plus displacement smoothing. CLI default for lightweight local execution. |
| `mlp` | Experimental CUDA-only per-detection ONNX classifier with per-track smoothing and checksum/schema/class-order validation. Not approved for production. |

The frozen #34 evaluation measured MLP accuracy at 62.67% versus VLM 93.33% on the same 150-row #33 held-out test. The image remains `CLASSIFIER=vlm`; no MLP deployment occurred.

### 3.5 Keyframes and aggregation

The aggregator produces:

- video duration and analysed-frame count;
- peak/average person count and dominant activity;
- person-minutes for all four activities;
- one-minute timeline bins;
- bounded keyframe candidates with at least one best frame per observed activity;
- optional zone and shift summaries.

Keyframes are annotated and encoded as base64 JPEG in `result.json`.

## 4. Zones, shifts, and workstation modes

Zones are configuration, never hardcoded geometry.

```json
{
  "recording_start": "2026-07-16T06:00:00+02:00",
  "shift": {
    "timezone": "Europe/Warsaw",
    "windows": [["07:00", "15:00"]],
    "breaks": [["11:00", "11:20"]]
  },
  "restrict_to_zones": false,
  "inference_roi": {
    "zone_id": "bending-1",
    "margin_px": 160
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

### 4.1 Assignment

The midpoint of a detection's bounding-box bottom edge is its foot point. The detection belongs to the first polygon containing that point; boundary points count as inside. A detection outside all zones has `zone_id: null`.

### 4.2 Restricting the analysis to zones

`restrict_to_zones` (boolean, default `false`) is the opt-in to masking the analysis to the polygons.

- `false` or absent: zones only ADD a per-zone breakdown. The headline tallies — `peak_persons`, `avg_persons`, `person_minutes`, `timeline`, `dominant_activity` — count every detection in the frame.
- `true` with at least one zone: those headline tallies count only detections whose foot point falls in a zone. Keyframe selection follows the same population, so the report's evidence matches its numbers.
- `true` with an empty `zones` list: no restriction. Masking to no polygon would zero the whole report, which no author means.

The per-zone breakdown is membership-gated either way and does not change with the flag.

### 4.3 Shift gating

- `recording_start` anchors `timestamp_s=0` to wall clock.
- `shift.windows` and `shift.breaks` are recurring half-open `[start,end)` intervals.
- Frames outside all windows or inside a break are excluded from aggregation.
- An IANA `shift.timezone` makes elapsed-time mapping DST-safe. Without it, the offset in `recording_start` is used.

### 4.4 Bending ruleset

`rules.type` defaults to `bending`; unsupported types fail at config load.

- Anchored worker: the in-zone track with the longest cumulative dwell.
- Presence/absence: sightings split into intervals when a sampling gap is too wide.
- Work: anchored-worker foot-point movement above `min_move_px`.
- Conversation: at least two stable, sufficiently close, low-movement tracks.
- Long absence: interval duration above `flag_after_s` is marked `flagged`.

These values are configurable pixel/time thresholds, not universal semantic truth. The bending pilot remains unverified on a viable station-framed stream.

### 4.5 Inference ROI

`inference_roi` crops the single pose call to the selected zone's bounding rectangle plus an explicit margin, then translates detections back into full-frame coordinates.

It is not the production default. Issue #86 measured fixed-640, fixed-1280, and focused-ROI arms; none passed the combined quality, runtime, and VRAM gate. Issue #88 (closed, deferred) needs an optically station-framed stream from the client before bending-pilot validation can resume. Separately, for deep-hall cameras a non-square `1280x736` input and `hybrid` tiling shipped as per-camera options (#100/#109/#110/#111).

## 5. Output contracts

### 5.1 Canonical result

`result.json` is UTF-8 JSON with `schema_version: 6`.

```json
{
  "schema_version": 6,
  "video_duration_s": 0.0,
  "total_frames": 0,
  "peak_persons": 0,
  "avg_persons": 0.0,
  "dominant_activity": "none",
  "person_minutes": {
    "sitting": 0.0,
    "standing": 0.0,
    "walking": 0.0,
    "running": 0.0
  },
  "timeline": [],
  "keyframes": [],
  "zones": [],
  "shift": null,
  "diagnostics": {
    "classifier": "vlm",
    "activity_model": null,
    "model_path": "models/yolo11s-pose.onnx",
    "model_sha256": "e3b0c442…",
    "input_size": [640, 640],
    "pose_mode": "full_frame",
    "conf_threshold": 0.25,
    "nms_threshold": 0.45,
    "source_frame": [3840, 2160],
    "detection_scale": {
      "input_scale": 0.1667,
      "floor_input_px": 60,
      "resolvable_height_native_px": 360.0,
      "resolvable_height_frac": 0.1667,
      "median_detected_height_native_px": 247.0,
      "detections_measured": 33,
      "recall_risk": "high"
    }
  }
}
```

Each zone contains posture person-minutes plus `presence` and `conversation` blocks. MLP runs populate version/checksum/feature-schema diagnostics.

The detection half of `diagnostics` records the configuration that produced the result, so any artifact can be attributed to it (issue #98). Every value is read back from what actually ran: `model_path` and `model_sha256` come from the weights file the session was created from — never from `MODEL_PATH`'s default or the Dockerfile ARG, because `docker-compose.yml` bind-mounts `./models` over the baked weights — and `input_size` is the ONNX's own declared `[w, h]`, not `IMG_SIZE`. `source_frame` is the `[w, h]` of the first analysed frame. `pose_mode` is the resolved detector mode (`full_frame` or `hybrid`, #111). `detection_scale` (#113) is a survivorship-bias-free recall-risk signal derived from scene geometry and the ~60 px model-input detection floor: `input_scale`, the smallest resolvable person height (`resolvable_height_native_px`/`_frac`), a supporting `median_detected_height_native_px`, and a `recall_risk` verdict (`high`|`normal`) the platform renders as a client-facing caveat so a low-recall run is never presented as a work-time measurement. It is `null` only when no frame was analysed. `model_sha256` is `null` only when the weights file cannot be read; diagnostics never fail a job.

Presentation, branding, localization, and interactive layout belong to the platform. The CLI's `--format html` output is a retained local debugging artifact, not the worker/platform contract.

### 5.2 Detection archive

`detections.jsonl` contains one JSON object per processed frame. Each person includes bbox, confidence, keypoints, activity, `track_id`, and optional `zone_id`.

The standalone worker uploads it beside `result.json` when enabled. Local validation uses `--dump-detections PATH.jsonl`. Because it taps the stream before minimum-track filtering, it is the audit source for explaining what was detected versus what was counted.

## 6. Standalone R2 protocol

Bucket default: `surveillance-data`.

```text
surveillance-jobs/{job_id}/
  status.json
  input/chunk_001.mp4
  input/chunk_002.mp4
  output/result.json
  output/detections.jsonl
```

The polling worker lists pending statuses, claims one job, downloads chunks, processes them, uploads result/evidence, and writes a terminal status. R2 methods retry three times with exponential backoff. Status-list reads use an ETag cache.

The standalone claim protocol remains low-concurrency last-writer-wins. Platform task dispatch does not use this claim protocol.

## 7. GPU-agent REST protocol

Entrypoint: `python -m gpu_service.rest_server`
Default port: `5003`

| Route | Contract |
|---|---|
| `GET /healthz` | `200` only after model/CUDA warm-up; otherwise `503` |
| `POST /analyze` | Validates task ID, non-empty input presigned URLs, result presigned URL, and tenant prefix; returns `202` |
| `GET /status/<task_id>` | Returns queued/running progress/completed/failed; `404` if unknown |

The task runner:

1. loads `ZONES_CONFIG_PATH` or `/config/zones.json` when present;
2. downloads every input URL;
3. concatenates multiple chunks with ffmpeg;
4. runs the canonical JSON pipeline;
5. uploads bytes to `result_presigned_url`;
6. records a terminal in-memory state.

One container per task is the intended deployment model. Repeated submission of the same queued/running task ID is idempotent.

## 8. Client appliance

### 8.1 Packaging

The only client deployment target is bare-metal `client_agent.appliance` through waitress on `:8080`.

- Root layout: `/opt/cctv-client`, `/etc/cctv-client`, system unit.
- User layout: `~/.local/share/cctv-client`, `~/.config/cctv-client`, user unit.
- User-mode install enables and verifies systemd linger.
- The retired client Docker image and direct R2 client must not return.

### 8.2 Standalone mode

When either `PLATFORM_URL` or `APPLIANCE_TOKEN` is missing, the appliance runs the local Flask UI, discovery, snapshots, and buffer-only recorder without platform callbacks.

Legacy `/upload`, `/start`, `/jobs`, and `/report` R2 workflows return `503`.

### 8.3 Platform mode

With both platform credentials set, the appliance:

- registers and pushes discovered cameras;
- heartbeats desired/actual camera state;
- starts/stops per-camera rolling recorders;
- trims retention continuously;
- polls and uploads requested time windows through presigned URLs;
- claims platform snapshot requests and uploads JPEGs through presigned URLs.

All platform communication is outbound HTTPS; no customer-LAN inbound route is required.

### 8.4 Runtime configuration

Environment cold-start fallbacks:

| Environment | Built-in default | Platform wire key |
|---|---:|---|
| `BUFFER_HOURS` | 1 | `buffer_hours` |
| `POLLING_INTERVAL_SECONDS` | 5 | `polling_interval_seconds` |
| `HEARTBEAT_INTERVAL_SECONDS` | 30 | `heartbeat_interval_seconds` |
| `UPLOAD_CHUNK_BYTES` | 52,428,800 | `upload_chunk_bytes` |

All values must be positive integers. Register/heartbeat settings override valid environment fallbacks. Later changes apply live to the buffer, poller, heartbeat loop, and uploader without restarting the appliance.

### 8.5 Snapshot variants

Platform claims support:

- `thumbnail`: RTSP frames scale to at most 640 px wide and use JPEG qscale 4;
- `detail`: RTSP frames retain decoded stream dimensions and use qscale 2.

Both RTSP variants decode for a one-second settle period before capture. Missing or unknown variants fall back to `thumbnail`. Vendor HTTP snapshot bytes pass through unchanged for both variants.

## 9. Models and dependencies

`setup-models.sh` checksum-pins YOLO11s-pose, OSNet x0_25, and the experimental activity MLP plus metadata.

The GPU image:

- uses `nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04`;
- installs pinned PyTorch cu128, transformers, accelerate, and qwen-vl-utils;
- bakes default YOLO, OSNet, and MLP artifacts;
- declares `CLASSIFIER=vlm`;
- fails fast when no usable GPU or insufficient configured VRAM is available.

The standalone compose bind-mount overlays `/app/models`; local model files must therefore be materialized before starting compose.

## 10. Measured evidence and promotion state

Measurements are valid only for their named fixture, model, image, and hardware.

| Evidence | Verified result |
|---|---|
| #86 fixed-640 VLM, 181.995 s bending fixture, RTX 5070 | 76.267 s wallclock; linear extrapolation 25.14 min/h; 7,866 MiB peak; failed detection-quality gate |
| #86 fixed-1280 | 126.393 s; 41.67 min/h extrapolated; 8,122 MiB; failed gate |
| #86 focused ROI | 96.435 s; 31.79 min/h extrapolated; 7,866 MiB; failed gate |
| Film 1 same-image resource comparison | VLM 129.660 s / 7,808 MiB; heuristic 73.898 s / 508 MiB; MLP 71.325 s / 540 MiB |
| #33 frozen test | VLM 93.33%; MLP 62.67%; heuristic 33.33% |

Sources: `benchmark-results/issue-86/README.md` and `docs/mlp-classifier-eval.md`.

Promotion state:

- VLM remains the deployed default.
- MLP is published only for reproducibility and is closed as not planned after failing quality gates.
- Focused ROI remains experimental after #86's no-winner result. A non-square `1280x736` per-camera input and `hybrid` tiling, by contrast, shipped and were measured on `magazyn-hall-v1` (#100/#101/#109/#110/#111): ~35% recall at 1280×736 and ~49% with tiling, vs ~7% at 640. No arm clears the recall gate alone, so results carry the `detection_scale` recall-risk caveat.
- Zone implementation is complete. Bending-pilot acceptance remains deferred (#88, closed) pending a client-provided station-framed stream plus a manually reviewed real shift.

## 11. Required invariants

- CUDA-only pose/Re-ID/MLP inference; no CPU fallback.
- Frame streaming; no full-video in-memory load.
- No face recognition or automatic long-gap identity.
- No client-side R2 credentials.
- No Docker client deployment.
- No promotion based on training success alone; every numeric gate needs measured evidence.
- Long-running operator commands must expose flushed progress or heartbeats.
- Completion claims must distinguish verified, failing, not-run, and external-input rows.

## 12. Repository map

```text
pipeline/                  core analysis implementation and tests
gpu-service/               GPU worker/REST image and tests
client-agent/              shared appliance package and tests
client-appliance/          systemd/install packaging and contract tests
datasets/activity-classifier/ #33 reviewed release metadata
training/activity-mlp/     #34 reproducible experiment and raw results
benchmarks/pose-resolution/ versioned #86 fixtures/methodology
benchmark-results/issue-86/ measured no-winner artifacts
docs/                      setup, benchmark, and evaluation documentation
```
