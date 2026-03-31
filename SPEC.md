# Surveillance Video Activity Analysis — Specification

Status: Draft v1
Scope: Standalone prototype (pre-platform integration)

## 1. Problem Statement

Clients with CCTV surveillance systems need automated activity analysis of recorded footage. They want answers to "what were people doing and for how long" without manual review of hours of video.

This system solves three problems:

- It turns raw surveillance footage into structured activity reports (person-minutes per activity class) without human review.
- It leverages idle GPU infrastructure (mining farms, dual-boot GRUB) for batch video processing, creating a new revenue stream for GPU investors.
- It validates a new `problem_type` for the ML Compute Exchange platform before full integration.

Important boundary:

- This is a **batch processing** system, not live monitoring. Client submits a recording → system processes → returns a report.
- The prototype operates standalone (R2-coordinated, no database). Platform integration (job dispatch, billing, tenant isolation) is Phase 4.
- Per-person tracking is deferred. Reports aggregate in **person-minutes**, not per-individual breakdown.

## 2. Goals and Non-Goals

### 2.1 Goals

- Process MP4 video files on GPU infrastructure and produce standalone HTML activity reports.
- Classify detected persons into 4 activity classes: sitting, standing, walking, running.
- Achieve ≥80% activity classification accuracy on standard-angle surveillance footage.
- Process at ≤1:1 ratio (1h video ≈ ≤1h processing) on RTX 5070.
- Enable zero-network-config video delivery via client-agent (outbound HTTPS only).
- Coordinate jobs via R2 key conventions (no database dependency).
- Produce reports viewable in any browser without additional software.
- Validate the AI pipeline before platform integration.

### 2.2 Non-Goals

- Live/real-time RTSP monitoring or streaming analysis.
- Per-person tracking or re-identification (ByteTrack/DeepSORT — deferred).
- Complex activity recognition beyond sit/stand/walk/run.
- RODO/GDPR compliance (deferred to production).
- Platform integration (problem_type, billing, tenant isolation — Phase 4).
- Multi-camera correlation or cross-camera tracking.
- Face recognition or person identification.

## 3. System Overview

### 3.1 Components

```
┌─ Client LAN ──────────────┐        ┌─ R2 Bucket ─────────────┐        ┌─ GPU Server ─────────────┐
│                            │        │ surveillance-jobs/       │        │                          │
│  client-agent (Docker)     │        │   {job_id}/              │        │  surveillance-serve      │
│  Flask UI :8080            │───────>│     status.json          │<───────│  (Docker, NVIDIA)        │
│  ffmpeg RTSP → MP4         │ upload │     input/chunk_*.mp4    │  poll  │                          │
│  boto3 → R2                │        │     output/report.html   │        │  worker.py (poll loop)   │
│                            │<───────│                          │───────>│  analyze.py pipeline     │
│  Shows report in browser   │  poll  │                          │ upload │  YOLO-pose + heuristics  │
└────────────────────────────┘        └──────────────────────────┘        └──────────────────────────┘
```

1. **Pipeline** (`pipeline/`)
   - Core AI: frame extraction → pose detection → activity classification → report generation.
   - CLI interface: `python analyze.py input.mp4 --output report.html`
   - Runs inside Docker with NVIDIA GPU access.

2. **Client Agent** (`client-agent/`)
   - Lightweight Docker daemon on client's LAN.
   - Flask web UI on `:8080` for RTSP configuration and job management.
   - Records from camera via ffmpeg, uploads to R2.
   - Polls for report completion.

3. **GPU Service** (`gpu-service/`)
   - Worker daemon on GPU server, polls R2 for pending jobs.
   - Downloads video → runs pipeline → uploads report.
   - Extends pipeline Docker image.

4. **R2 Coordinator**
   - No database. Job state tracked via `status.json` in R2.
   - Key convention: `surveillance-jobs/{job_id}/status.json`

### 3.2 External Dependencies

- Cloudflare R2 (S3-compatible) — video storage + job coordination.
- NVIDIA GPU with CUDA 12+ and cuDNN — inference.
- ffmpeg — frame extraction (pipeline) and RTSP recording (client-agent).
- Client's RTSP-capable surveillance camera.

## 4. Domain Model

### 4.1 Job

```
{
  job_id:        string      // UUID v4
  status:        enum        // pending | processing | completed | failed
  worker_id:     string?     // server identifier that claimed the job
  created_at:    ISO 8601
  updated_at:    ISO 8601
  progress_pct:  number      // 0-100
  input_chunks:  string[]    // R2 keys of uploaded MP4 chunks
  error:         string?     // failure reason
  duration_s:    number?     // video duration in seconds
  report_key:    string?     // R2 key of generated report
}
```

### 4.2 Frame Analysis Result

```
{
  frame_index:    number
  timestamp_s:    number      // seconds from video start
  person_count:   number
  persons: [
    {
      bbox:       [x1, y1, x2, y2]     // pixel coords in original image
      confidence: number                // 0-1
      keypoints:  [[x,y,vis], ×17]     // COCO 17 keypoints
      activity:   enum                  // sitting | standing | walking | running
    }
  ]
}
```

### 4.3 Activity Report

```
{
  summary: {
    video_duration_s:  number
    total_frames:      number
    peak_persons:      number
    avg_persons:       number
    dominant_activity: enum
  }
  person_minutes: {
    sitting:   number
    standing:  number
    walking:   number
    running:   number
  }
  timeline: [                           // 1-minute bins
    { minute: number, sitting: n, standing: n, walking: n, running: n }
  ]
  keyframes: [                          // 5 annotated frames
    { timestamp_s: number, image_base64: string, persons: [...] }
  ]
}
```

## 5. AI Pipeline Specification

### 5.1 Model

| Property | Value |
|----------|-------|
| Model | YOLOv11n-pose |
| Format | ONNX |
| Input | `[1, 3, 640, 640]` float32 (NCHW, normalized 0-1) |
| Output | `[1, 56, N]` — 4 bbox + 1 conf + 51 keypoints (17×3) |
| Runtime | onnxruntime-gpu, CUDAExecutionProvider only |
| VRAM | ~2-4 GB |
| Latency | ~100ms/frame on RTX 5070 (validated 2026-03-31) |

**No CPU fallback.** If CUDA is unavailable, the pipeline must fail with a clear error. CPU inference at ~10x slower (1s/frame) makes 1:1 processing ratio impossible.

### 5.2 Frame Extraction

```
ffmpeg -i input.mp4 -vf fps=1 -q:v 2 frames/frame_%06d.jpg
```

- Sampling rate: **1 fps** (3600 frames/hour)
- Frame size: ~50-100KB at 1080p JPEG quality 2
- Memory: frame-by-frame processing, never full video in RAM

### 5.3 Preprocessing

Identical to validated `infra/ai-test/yolo-serve/app.py`:

1. Load image via PIL → RGB
2. Resize to 640×640 (bilinear)
3. Convert to float32, normalize `/255.0`
4. Transpose HWC → CHW
5. Add batch dimension → `[1, 3, 640, 640]`

### 5.4 Postprocessing (Pose)

YOLO-pose output `[1, 56, N]` (transposed format):

```
Row 0-3:   cx, cy, w, h (bbox center + size, 640-space)
Row 4:     confidence
Row 5-55:  17 keypoints × 3 (x, y, visibility) in 640-space
```

Per detection:
1. Filter by confidence ≥ `CONFIDENCE_THRESHOLD` (default 0.25)
2. NMS with IoU threshold 0.45
3. Scale bbox + keypoints to original image coordinates
4. Extract 17 COCO keypoints

### 5.5 COCO 17 Keypoints

```
 0: nose           1: left_eye      2: right_eye
 3: left_ear       4: right_ear
 5: left_shoulder  6: right_shoulder
 7: left_elbow     8: right_elbow
 9: left_wrist    10: right_wrist
11: left_hip      12: right_hip
13: left_knee     14: right_knee
15: left_ankle    16: right_ankle
```

Heuristics use: **5,6** (shoulders), **11,12** (hips), **13,14** (knees), **15,16** (ankles).

### 5.6 Activity Classification

Rule-based decision tree per detected person. All thresholds are tunable constants.

```
Input: 17 keypoints with visibility scores

Step 1: Keypoint visibility check
  Required: hips (11,12), knees (13,14), ankles (15,16) with visibility > 0.5
  If insufficient → FALLBACK: bbox aspect ratio
    bbox height/width < 1.5 → SITTING
    else → STANDING

Step 2: Compute geometric features
  knee_angle = average angle(hip, knee, ankle) for both legs
  hip_height_ratio = (bbox_bottom - avg_hip_y) / bbox_height
  stride_ratio = |ankle_L_x - ankle_R_x| / |hip_L_x - hip_R_x|
  torso_lean = angle_from_vertical(avg_shoulders → avg_hips)

Step 3: Classification
  IF knee_angle < 120° AND hip_height_ratio < 0.40 → SITTING
  IF stride_ratio > 2.0 AND torso_lean > 15°       → RUNNING
  IF stride_ratio > 1.3                             → WALKING
  ELSE                                              → STANDING
```

**Default thresholds:**

| Threshold | Value | Rationale |
|-----------|-------|-----------|
| `KNEE_ANGLE_SIT` | 120° | Seated posture has bent knees |
| `HIP_HEIGHT_RATIO_SIT` | 0.40 | Seated hips are lower in bbox |
| `STRIDE_RATIO_WALK` | 1.3 | Walking spreads ankles beyond hip width |
| `STRIDE_RATIO_RUN` | 2.0 | Running has wider stance |
| `TORSO_LEAN_RUN` | 15° | Forward lean while running |
| `KEYPOINT_VIS_MIN` | 0.5 | Minimum visibility confidence |

Target accuracy: **≥80%** on surveillance footage from standard angles (camera at 2-3m height, 30-60° tilt).

### 5.7 Aggregation

Per-frame: `{ timestamp_s, person_count, activities: { Activity → count } }`

Summary: sum person-frames per activity ÷ sampling_fps = **person-seconds** ÷ 60 = **person-minutes**

Timeline: group into 1-minute bins, sum per activity.

### 5.8 Report Generation

Standalone HTML file (Jinja2 template + vendored Chart.js inline):

1. **Summary table** — video duration, frames analyzed, peak/avg person count, dominant activity
2. **Pie chart** — % person-time per activity
   - Colors: green=standing, blue=sitting, orange=walking, red=running
3. **Timeline chart** — stacked bar, X=time (1-min bins), Y=person count, colored by activity
4. **5 annotated keyframes** — selected by max person count with ≥2min spacing
   - Rendered: bounding boxes, 17-keypoint skeleton overlay, activity label per person
   - Encoded: base64 PNG inline

Expected report file size: ~5-10MB (dominated by base64 keyframes).

## 6. R2 Job Coordination Protocol

### 6.1 Bucket

Separate bucket: `surveillance-data` (isolated from main platform R2).

### 6.2 Key Convention

```
surveillance-jobs/{job_id}/
  status.json              # Job state (source of truth)
  input/chunk_001.mp4      # Uploaded video chunks
  input/chunk_002.mp4
  output/report.html       # Generated report
```

### 6.3 Job Lifecycle

```
                 client-agent uploads
                 video + status.json
          ┌──────────────────────────────┐
          ▼                              │
      ┌────────┐    worker claims    ┌───┴───────┐
      │PENDING │───────────────────>│PROCESSING  │
      └────────┘                    └─────┬──────┘
                                          │
                            ┌─────────────┼─────────────┐
                            ▼                             ▼
                     ┌───────────┐                 ┌──────────┐
                     │ COMPLETED │                 │  FAILED  │
                     └───────────┘                 └──────────┘
```

### 6.4 status.json Schema

```json
{
  "job_id": "uuid",
  "status": "pending",
  "created_at": "2026-03-31T10:00:00Z",
  "updated_at": "2026-03-31T10:00:00Z",
  "input_chunks": ["input/chunk_001.mp4"],
  "worker_id": null,
  "progress_pct": 0,
  "error": null,
  "duration_s": null,
  "report_key": null
}
```

### 6.5 Claim Protocol (MVP)

Last-writer-wins. Worker reads `status.json`, checks `status == "pending"`, writes `status: "processing"` + own `worker_id`. At low concurrency (1-2 workers) this is acceptable. Post-MVP: platform's `SELECT FOR UPDATE SKIP LOCKED`.

### 6.6 Progress Updates

Worker updates `status.json` periodically during processing with `progress_pct` (0-100). Client-agent polls every 15 seconds.

## 7. Client Agent Specification

### 7.1 Purpose

Lightweight daemon on client's LAN. Records from RTSP camera, uploads to R2, polls for results.

### 7.2 Web UI (Flask, :8080)

| Route | Method | Function |
|-------|--------|----------|
| `/` | GET | Recording form: RTSP URL + duration selector |
| `/test-connection` | POST | ffmpeg probe of RTSP URL → success/fail |
| `/start` | POST | Begin recording + upload flow |
| `/jobs` | GET | Job list with status badges, auto-refresh 10s |
| `/jobs/<id>/report` | GET | Serve downloaded HTML report |

### 7.3 Recording

```bash
ffmpeg -rtsp_transport tcp -i {rtsp_url} -c copy -t {duration_s} output.mp4
```

- Stream copy (no re-encoding) — minimal CPU.
- Long recordings: `-f segment -segment_time 3600` → 1h chunks.
- MVP: "record for next Xh" (1/2/4/8h options). Scheduled recording deferred.

### 7.4 Upload

- boto3 S3-compatible client with R2 credentials (scoped API token).
- `upload_file()` handles multipart automatically (R2: 5GB single, 5TB multipart).
- After upload: write `status.json` with `status: "pending"`.

### 7.5 Docker

```yaml
services:
  surveillance-agent:
    build: ./client-agent
    image: python:3.11-slim + ffmpeg
    ports: ["8080:8080"]
    volumes: ["./recordings:/recordings"]
    env_file: .env
    restart: unless-stopped
```

No GPU required. Runs on any machine with Docker on client's network.

### 7.6 Installation Flow

1. Client receives: agent zip/git link + `.env` with R2 credentials + video tutorial.
2. Client installs Docker Desktop (Windows) or Docker Engine (Linux).
3. `docker compose -f docker-compose.client.yml up -d`
4. Opens `http://localhost:8080`.

## 8. GPU Service Specification

### 8.1 Worker Loop

```python
while True:
    job = poll_for_pending()     # ListObjects → find status.json with status=pending
    if job:
        claim(job)               # Write status=processing + worker_id
        download_video(job)      # R2 → local /tmp
        report = run_pipeline()  # analyze.py as library
        upload_report(job)       # HTML → R2 output/
        complete(job)            # Write status=completed + report_key
    else:
        sleep(10)                # Poll interval
```

### 8.2 Error Handling

- Pipeline crash → set `status: "failed"` + error message in status.json.
- Download failure → retry 3× with exponential backoff, then fail.
- Upload failure → retry 3×, then fail (report preserved locally).
- Worker crash mid-processing → status stays "processing". No automatic recovery in MVP — operator re-sets to "pending" manually. Post-MVP: heartbeat + timeout.

### 8.3 Docker

```yaml
services:
  surveillance-serve:
    build: ./pipeline
    runtime: nvidia
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["${SURVEILLANCE_GPU:-0}"]
              capabilities: [gpu]
    volumes:
      - ${MODELS_DIR:-./models}:/models:ro
    env_file: .env
    command: ["python3", "worker.py"]
```

Pattern identical to `infra/ai-test/docker-compose.yml`.

## 9. Performance Model

### 9.1 Processing Budget (1h video, 1080p, RTX 5070)

| Stage | Per-frame | Total (3600 frames) |
|-------|-----------|---------------------|
| Frame extraction (ffmpeg) | ~10ms | ~36s |
| YOLO-pose inference (GPU) | ~100ms | ~360s (6min) |
| Activity classification | ~1ms | ~3.6s |
| Report generation | — | ~5s (one-time) |
| **Total** | ~111ms | **~7min** |

Processing ratio: **~8:1** (1h video in ~7min). Well within 1:1 target.

### 9.2 VRAM Budget

| Component | VRAM |
|-----------|------|
| YOLO11n-pose ONNX model | ~200MB |
| CUDA context + inference buffers | ~400MB |
| **Total** | **~600MB** |

12GB available per GPU — can run alongside other models.

### 9.3 Network Budget (upload)

| Video length | File size (1080p H.264) | Upload at 10 Mbps | Upload at 50 Mbps |
|-------------|-------------------------|--------------------|--------------------|
| 1h | ~2-4 GB | ~30-55min | ~6-11min |
| 8h | ~16-32 GB | ~4-7h | ~45-85min |

Upload time may exceed processing time on slow connections. Chunked upload with resume is essential.

## 10. Configuration

### 10.1 Environment Variables

```bash
# R2 credentials (shared by client-agent and gpu-service)
R2_ENDPOINT=https://<account>.r2.cloudflarestorage.com
R2_ACCESS_KEY_ID=<key>
R2_SECRET_ACCESS_KEY=<secret>
R2_BUCKET=surveillance-data

# GPU service
SURVEILLANCE_GPU=0
MODELS_DIR=./models
POLL_INTERVAL_S=10

# Client agent
RTSP_DEFAULT_URL=                    # optional pre-fill
MAX_RECORDING_HOURS=8

# Pipeline
CONFIDENCE_THRESHOLD=0.25
SAMPLING_FPS=1
KEYFRAME_COUNT=5
KEYFRAME_MIN_SPACING_S=120
```

## 11. Report Format

### 11.1 Structure

```html
<!DOCTYPE html>
<html>
<head>
  <title>Surveillance Activity Report — {job_id}</title>
  <script>/* vendored Chart.js ~200KB */</script>
  <style>/* inline CSS */</style>
</head>
<body>
  <h1>Activity Analysis Report</h1>
  <section id="summary">
    <!-- Summary table: duration, frames, persons, dominant activity -->
  </section>
  <section id="activity-breakdown">
    <!-- Pie chart: person-minutes per activity -->
  </section>
  <section id="timeline">
    <!-- Stacked bar chart: 1-min bins, activity breakdown -->
  </section>
  <section id="keyframes">
    <!-- 5 annotated frames: bbox + skeleton + activity label, base64 PNG -->
  </section>
  <footer>
    Generated by ML Compute Exchange — {timestamp}
  </footer>
</body>
</html>
```

### 11.2 Keyframe Selection

1. Score each frame by person count.
2. Sort descending.
3. Greedily select top frames with ≥`KEYFRAME_MIN_SPACING_S` between them.
4. Take first `KEYFRAME_COUNT` (default 5).

### 11.3 Keyframe Annotation

Per person in keyframe:
- Bounding box (2px solid, color by activity)
- 17-keypoint skeleton overlay (COCO skeleton connections)
- Activity label above bbox

## 12. Integration Path (Phase 4, future)

When prototype is validated, integrate into the main ML Compute Exchange platform:

| Aspect | Prototype (now) | Integrated (future) |
|--------|----------------|---------------------|
| Job coordination | R2 key conventions | Platform job dispatch (`SELECT FOR UPDATE SKIP LOCKED`) |
| Job claim | Last-writer-wins | Atomic DB claim |
| Billing | None | GPU-seconds × surveillance rate (new `billing_rates` entry) |
| Tenant isolation | Single R2 bucket | `tenants/{tenantId}/` prefix |
| Task routing | Hardcoded | `task_routing_rules` entry: `problem_type: 'surveillance_analysis'` |
| Model registry | Local | `models` table: `yolo11n-pose`, `docker_image: surveillance-serve` |
| Docker | Standalone compose | Added to `gpu-agent` docker-compose (port 5003) |
| Video delivery | Client-agent only | Client-agent OR dashboard upload |
| Status updates | R2 polling | SSE via data-service (existing pattern) |
| Tracking | Per-frame aggregation | ByteTrack per-person tracking |
| RODO | Ignored | Umowa powierzenia, DPIA, optional face blur |

## 13. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| ONNX Runtime on Blackwell (sm_120) | Pipeline doesn't load on GPU | **Validated 2026-03-31**: onnxruntime-gpu works on RTX 5070 with CUDA EP. yolo11n object detection at 104ms. Pose variant needs verification |
| Heuristic accuracy <80% | Unreliable reports | Tune thresholds on real footage. Fallback: lightweight MLP classifier on keypoints |
| Camera angle (top-down/extreme tilt) | Keypoints poorly visible | Fallback to bbox aspect ratio. Document requirement: camera at 2-3m, 30-60° angle |
| Upload time > processing time | Poor UX on slow connections | Chunked multipart upload with resume. Consider H.265 re-encode on agent |
| Multiple overlapping persons | Occlusion, missed detections | YOLO handles partial occlusions. Accuracy degrades >10 persons |
| Client can't install Docker | Blocked from using system | Video tutorial. Post-MVP: standalone binary (Go) |
| RODO | Legal risk in production | Deferred. Before production: powierzenie danych, DPIA, optional auto-blur |
| Worker crash mid-processing | Job stuck in "processing" | MVP: manual reset. Post-MVP: heartbeat + timeout auto-recovery |
| onnxruntime-node has no CUDA | Can't use Node.js for inference | **Confirmed 2026-03-31**: npm onnxruntime-node lacks CUDA EP. Always use Python onnxruntime-gpu |

## 14. Validation Checklist

### Phase 1: Core AI Pipeline

- [ ] `yolo11n-pose.onnx` loads on RTX 5070 with CUDAExecutionProvider
- [ ] Frame extraction via ffmpeg at 1fps produces correct frame count
- [ ] Pose detection returns 17 keypoints per person with reasonable confidence
- [ ] Activity heuristics classify ≥80% of activities correctly on test footage
- [ ] Report HTML opens in browser with charts rendered and keyframes visible
- [ ] Processing ratio ≤1:1 (1h video in ≤1h)
- [ ] `test_heuristics.py` passes with synthetic keypoint arrays

### Phase 2: Client Agent

- [ ] Flask UI accessible at :8080
- [ ] RTSP connection test succeeds with real camera
- [ ] Recording produces valid MP4 via ffmpeg stream copy
- [ ] Upload to R2 succeeds with progress indication
- [ ] status.json created correctly in R2

### Phase 3: GPU Service E2E

- [ ] Worker detects pending job via R2 polling
- [ ] Worker claims job (status → processing)
- [ ] Worker downloads video, runs pipeline, uploads report
- [ ] Worker updates status → completed with report_key
- [ ] Client-agent detects completion and displays report
- [ ] Failed job correctly sets status → failed with error

### Phase 4: Integration (future)

- [ ] New `problem_type: 'surveillance_analysis'` in task_routing_rules
- [ ] Docker service added to gpu-agent compose on port 5003
- [ ] Billing rate configured and usage records created
- [ ] Tenant isolation via R2 prefix
- [ ] SSE status updates replace R2 polling

## 15. File Structure

```
infra/video-test/
├── SPEC.md                            # This document
├── DECISION_LOG.md                    # All 8 design decisions + rationale
├── IMPLEMENTATION_PLAN.md             # 4-phase implementation roadmap
├── RTX5070_CONSTRAINTS.md             # Hardware compatibility analysis
├── docker-compose.yml                 # GPU service
├── docker-compose.client.yml          # Client agent
├── .env.example
├── setup-models.sh                    # Download yolo11n-pose.onnx
├── benchmark.sh
├── pipeline/                          # Phase 1: Core AI
│   ├── Dockerfile                     # nvidia/cuda + python + ffmpeg
│   ├── requirements.txt               # onnxruntime-gpu, opencv, jinja2, etc.
│   ├── analyze.py                     # CLI: python analyze.py input.mp4
│   ├── frame_extractor.py
│   ├── pose_detector.py
│   ├── activity_classifier.py
│   ├── report_generator.py
│   ├── report_template.html           # Jinja2 + Chart.js
│   └── test_heuristics.py
├── gpu-service/                       # Phase 3: R2 worker
│   ├── Dockerfile
│   ├── worker.py
│   └── r2_client.py
├── client-agent/                      # Phase 2: Client daemon
│   ├── Dockerfile
│   ├── agent.py                       # Flask app
│   ├── recorder.py                    # ffmpeg RTSP
│   ├── uploader.py                    # boto3 → R2
│   └── templates/index.html
├── test/                              # Validation scripts
│   ├── test_video_pose.py             # End-to-end pose test on sample video
│   ├── run_test.sh
│   └── requirements.txt
├── models/                            # .gitkeep (yolo11n-pose.onnx)
└── test-data/                         # .gitkeep (sample MP4s)
```

## 16. Technology Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Pose model | YOLOv11n-pose (ONNX) | Proven YOLO infra, nano for speed, pose for keypoints |
| Inference runtime | onnxruntime-gpu | **Validated on RTX 5070.** Python package has CUDA EP (npm package does not) |
| Frame extraction | ffmpeg (system) | Industry standard, handles all codecs |
| Image processing | OpenCV + Pillow | Annotation overlays + base64 encoding |
| Activity classification | Geometric heuristics | No training data needed, interpretable, tunable |
| Report template | Jinja2 + Chart.js | Standalone HTML, no server needed to view |
| Client UI | Flask | Minimal, sufficient for form + job list |
| R2 client | boto3 (S3-compat) | Proven pattern from platform design |
| Container base | nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04 | Validated on test VPS 2026-03-31 |
| Python runner | uv (preferred), pip fallback | Fast, clean dependency management |

## 17. Key Lessons (from ai-test validation 2026-03-31)

These findings directly inform this specification:

1. **onnxruntime-node (npm) has no CUDA support** — all inference must use Python `onnxruntime-gpu`. This is why the pipeline and gpu-service are Python, not TypeScript.

2. **nvidia/cuda base image required** — `python:3.11-slim` lacks CUDA libraries. Must use `nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04` and install Python on top.

3. **RTX 5070 YOLO inference at ~100ms** — validated with yolo11n object detection. Pose variant expected similar (same backbone, slightly larger output tensor).

4. **574MB VRAM for YOLO nano** — leaves 11.5GB headroom on 12GB GPU. No VRAM concerns for pose model.

5. **GPU-only, no CPU fallback** — CPU inference (~1s/frame) breaks the 1:1 processing ratio. Fail fast if no GPU.

6. **Docker caching gotcha** — `scp` doesn't delete old files. Always clean target directory before copying new Docker context.

7. **uv over pip** — faster, cleaner Python dependency management. Scripts should auto-detect uv → pip fallback.
