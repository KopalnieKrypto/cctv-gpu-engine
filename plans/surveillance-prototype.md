# Plan: Surveillance Video Activity Analysis — Batch Prototype

> Source PRD: [GitHub Issue #1](https://github.com/KopalnieKrypto/cctv-gpu-engine/issues/1) | Technical spec: `SPEC.md`

## Architectural decisions

Durable decisions that apply across all phases:

- **Architecture style**: 2 Docker images (client-agent, gpu-service) coupled only through R2 bucket. No database, no direct communication.
- **Data model**: Job state via `status.json` in R2 (`surveillance-jobs/{job_id}/`). Statuses: pending → processing → completed | failed.
- **AI runtime**: Python `onnxruntime-gpu` with CUDAExecutionProvider only. No CPU fallback. Model: YOLOv11n-pose ONNX.
- **Target GPUs**: RTX 5070 (primary), RTX 4090.
- **Docker base**: `nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04` for GPU workloads, `python:3.11-slim` for client-agent.
- **Report format**: Standalone HTML — vendored Chart.js, base64 images, zero external deps.
- **R2 bucket**: `surveillance-data`, Cloudflare R2 (S3-compatible), scoped API tokens via boto3.
- **Activity classes**: sitting, standing, walking, running — no others.
- **Frame sampling**: ffmpeg at 1 fps, frame-by-frame processing, never full video in RAM.

---

## Phase 1: Single-frame pose pipeline

**User stories**: 20, 21

### What to build

Minimal end-to-end proof: take an MP4, extract a single frame with ffmpeg, run YOLO-pose inference on GPU, parse the output tensor into keypoints, classify activity via heuristics, print structured result to stdout.

This validates the entire AI stack works — ONNX model loading, CUDA inference, output parsing, and heuristic classification — before building anything around it.

### Acceptance criteria

- [ ] `yolo11n-pose.onnx` loads with CUDAExecutionProvider (fails with clear error if no CUDA)
- [ ] ffmpeg extracts a frame from test MP4 at correct timestamp
- [ ] Preprocessing produces `[1, 3, 640, 640]` float32 tensor
- [ ] Postprocessing parses `[1, 56, N]` output into bbox + 17 keypoints per person
- [ ] NMS filters overlapping detections (IoU 0.45)
- [ ] Heuristics return one of: sitting, standing, walking, running
- [ ] Keypoint coordinates scale correctly to original image dimensions
- [ ] CLI outputs JSON with person count, keypoints, and activity per detection

### Testing focus

- [ ] **CUDA availability check**: run on machine without GPU → pipeline exits with clear error message, not a cryptic ONNX crash
- [ ] **Model format validation**: load model, verify input/output tensor shapes match expected `[1,3,640,640]` / `[1,56,N]`
- [ ] **Heuristic unit tests**: synthetic keypoint arrays for each activity class (sitting with bent knees, standing upright, walking stride, running pose) → verify correct classification
- [ ] **Fallback path**: keypoints with low visibility (<0.5) → verify bbox aspect ratio fallback triggers
- [ ] **Edge cases**: frame with 0 persons → empty result. Frame with 10+ persons → all detected and classified. Extremely dark/blurry frame → low confidence detections filtered out.

---

## Phase 2: Full video analysis + HTML report

**User stories**: 9, 10, 11, 12, 13, 20

### What to build

Extend Phase 1 to process entire videos: extract all frames at 1 fps, run pose detection on each, aggregate results into person-minutes, generate timeline bins, select keyframes, render standalone HTML report with charts and annotated keyframes.

`python analyze.py input.mp4 --output report.html` — the complete local pipeline.

### Acceptance criteria

- [ ] ffmpeg extracts frames at 1 fps — frame count matches `video_duration_s` (±1)
- [ ] Frame-by-frame processing: VRAM stable, no memory leak over long videos
- [ ] Aggregation: person-minutes per activity calculated correctly (person-frames ÷ fps ÷ 60)
- [ ] Timeline: 1-minute bins with per-activity person counts
- [ ] Keyframe selection: 5 frames with max person count, ≥120s spacing between them
- [ ] Keyframe annotation: bounding boxes, COCO skeleton overlay, activity label per person
- [ ] Report HTML: opens in browser, pie chart + stacked bar chart render (Chart.js), keyframe images display
- [ ] Report has zero external requests (fully self-contained)
- [ ] Report file size reasonable (~5-10MB for 1h video)
- [ ] Processing time ≤1:1 ratio on RTX 5070

### Testing focus

- [ ] **Known-length video**: use a video with known duration → verify frame count, timeline bin count, and person-minutes math
- [ ] **Report integrity**: open report HTML in headless browser (or just check for `<canvas>`, Chart.js init, base64 `<img>` tags) → all sections present
- [ ] **Memory stability**: process a 30+ minute video → monitor VRAM usage, ensure no growth over time
- [ ] **Empty video**: video with no people → report generates with zero person-minutes, no crash
- [ ] **Short video**: <1 minute video → report still generates (fewer than 5 keyframes is fine)
- [ ] **Corrupted frames**: if ffmpeg produces a bad frame → pipeline skips it gracefully, doesn't abort

---

## Phase 3: R2 job coordination

**User stories**: 16, 22, 23, 24

### What to build

R2 client module + worker loop: poll R2 for `status.json` with `status: pending`, claim job (verify still pending, write `processing` + `worker_id`), download video chunks, run pipeline, upload report HTML, update status to `completed` or `failed`.

Tested by manually placing a test job (MP4 + status.json) in R2.

### Acceptance criteria

- [ ] Worker discovers pending job via R2 `ListObjects` within poll interval
- [ ] Claim protection: worker reads status, verifies `pending`, then writes `processing`. If status changed between read and write → worker skips job
- [ ] Video download: all `input/chunk_*.mp4` files downloaded to local temp dir
- [ ] Pipeline runs on downloaded video, produces report
- [ ] Report uploaded to `output/report.html` in job's R2 prefix
- [ ] `status.json` updated to `completed` with `report_key` and `progress_pct: 100`
- [ ] On pipeline failure: `status.json` set to `failed` with error message, worker stays alive
- [ ] Progress updates: `status.json` updated periodically during processing with `progress_pct`

### Testing focus

- [ ] **R2 connectivity**: invalid credentials → clear error on startup, not silent failure
- [ ] **Claim race condition**: create 2 workers, submit 1 job → only 1 worker processes it (the other skips after seeing `processing` status)
- [ ] **Download failure**: delete video from R2 after job is claimed → worker handles missing file, sets `failed` status
- [ ] **Upload failure**: simulate R2 write error (e.g., wrong bucket) → worker retries 3×, then fails job, keeps report locally
- [ ] **Corrupted video in R2**: upload a truncated MP4 → worker fails gracefully, sets error in status.json
- [ ] **Empty bucket**: no pending jobs → worker polls quietly, no errors in logs
- [ ] **Status.json format**: missing fields or malformed JSON → worker logs warning, skips job

---

## Phase 4: GPU service Docker + investor dashboard

**User stories**: 14, 15, 17, 18, 19, 25

### What to build

Dockerize the worker (Phase 3) with NVIDIA runtime. Add a simple HTTP dashboard page (served from the same container) showing job history: job ID, timestamp, status, duration, error. The investor sees this page to confirm their GPU is working.

`docker compose up` on GPU server → worker starts polling, dashboard accessible.

### Acceptance criteria

- [ ] Dockerfile builds on `nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04` base
- [ ] `docker compose up` starts worker + dashboard with NVIDIA GPU access
- [ ] Worker processes jobs inside container (CUDA inference works)
- [ ] Dashboard page accessible via HTTP (e.g., `:5000/dashboard`)
- [ ] Dashboard shows: job ID, timestamp, status, video duration, error (if failed)
- [ ] Dashboard auto-refreshes or has manual refresh
- [ ] Container handles bad input (corrupted video, empty file) without crashing
- [ ] Works on RTX 5070 and RTX 4090
- [ ] Container restarts cleanly (`restart: unless-stopped`)

### Testing focus

- [ ] **GPU passthrough**: `docker compose up` on machine with NVIDIA GPU → CUDA EP loads inside container. On machine without GPU → container exits with clear error
- [ ] **Image size**: verify Docker image is reasonable (<5GB). Large images slow deployment to investor machines
- [ ] **Container resilience**: kill container mid-processing → restart → worker resumes polling (stuck job stays `processing` — acceptable for MVP)
- [ ] **Dashboard under load**: 100+ completed jobs in history → page still loads quickly
- [ ] **Env vars**: missing R2 credentials → container logs clear error on startup, doesn't silently poll nothing
- [ ] **Model download**: verify `setup-models.sh` or model mount works inside container (models not baked into image)

---

## Phase 5: Client agent — upload + status

**User stories**: 1, 2, 5, 6, 7, 8

### What to build

Flask web app in Docker: upload MP4 form, boto3 upload to R2 with `status.json` creation, job list page with auto-refresh (polls R2 for status updates), report viewer (downloads report HTML from R2 and serves it), direct report download endpoint.

`docker compose -f docker-compose.client.yml up -d` → UI at `:8080`.

### Acceptance criteria

- [ ] Flask app accessible at `:8080`
- [ ] Upload form accepts MP4 file
- [ ] Upload creates `surveillance-jobs/{job_id}/input/chunk_001.mp4` + `status.json` (pending) in R2
- [ ] Job list page shows all submitted jobs with status badges
- [ ] Job list auto-refreshes every 10 seconds
- [ ] Completed jobs show "View Report" link → renders report HTML in browser
- [ ] Direct download link for report HTML file
- [ ] Docker container runs on `python:3.11-slim` (no GPU needed)
- [ ] `docker compose -f docker-compose.client.yml up -d` starts cleanly

### Testing focus

- [ ] **Large file upload**: upload a 2GB+ MP4 → upload completes without timeout or memory issues (multipart)
- [ ] **R2 connectivity from client**: invalid R2 credentials → UI shows clear error, not 500
- [ ] **Job status transitions**: manually set status.json to each state (pending, processing with progress_pct, completed, failed) → UI reflects correctly
- [ ] **Report rendering**: completed report loads in browser via client-agent proxy without broken assets
- [ ] **Concurrent uploads**: two uploads at same time → both create separate jobs correctly
- [ ] **No GPU dependency**: container runs on any machine with Docker — no CUDA, no NVIDIA runtime

---

## Phase 6: Client agent — RTSP recording

**User stories**: 3, 4

### What to build

Add RTSP support to client-agent: connection test endpoint (ffmpeg probe), recording form with duration selector (1/2/4/8h), ffmpeg stream copy to MP4, then feed into Phase 5 upload flow. Long recordings segmented into 1h chunks.

### Acceptance criteria

- [ ] RTSP connection test: POST with URL → success/fail response within timeout
- [ ] Recording: ffmpeg stream copy from RTSP URL for selected duration
- [ ] Long recordings (>1h): segmented into 1h chunks, all uploaded as `chunk_001.mp4`, `chunk_002.mp4`, etc.
- [ ] Recording produces valid MP4 (playable, correct duration)
- [ ] After recording completes: auto-uploads to R2 and creates job (reuses Phase 5 flow)
- [ ] UI shows recording progress (or at minimum: "recording in progress" state)

### Testing focus

- [ ] **Invalid RTSP URL**: test with garbage URL → clear "connection failed" message, not hung ffmpeg process
- [ ] **RTSP timeout**: camera goes offline mid-recording → ffmpeg exits, partial MP4 is still valid and uploaded
- [ ] **Duration accuracy**: request 1h recording → resulting MP4 is ~3600s (±5s)
- [ ] **Disk space**: 8h recording at 1080p ~32GB → verify enough temp space, cleanup after upload
- [ ] **ffmpeg process management**: if user starts a second recording while one is active → reject or queue, don't spawn unbounded ffmpeg processes
