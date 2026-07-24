# Pose resolution selection benchmark (issue #86)

This runbook compares the same `yolo11s-pose` weights in three modes:

1. full frame → fixed 640 letterbox (`baseline_640`);
2. full frame → fixed 1280 letterbox (`full_frame_1280`);
3. configured station inference ROI + fixed margin → fixed 640 letterbox
   (`focused_roi_640`).

The existing 3×3/640 result is reference evidence only and is never run as a
production arm.

## Versioned fixture

Store the fixture manifest, annotations, zones config, and methodology under a
versioned directory such as `benchmarks/pose-resolution/bending-pilot-v1/`.
Large JPEG/MP4 payloads may live in an immutable GitHub release asset; the
manifest must retain their relative materialized paths and SHA-256 values.

The pilot manifest schema is:

```json
{
  "schema_version": 1,
  "fixture_id": "bending-pilot-v1",
  "annotation_methodology": "METHODOLOGY.md",
  "frames": [
    {
      "id": "window-1-frame-000",
      "window_id": "window-1",
      "path": "frames/window-1-frame-000.jpg",
      "sha256": "<64 lowercase hex characters>",
      "persons": [{"bbox": [x1, y1, x2, y2]}]
    }
  ]
}
```

Requirements enforced before CUDA loads:

- at least 60 frames from at least three distinct recording windows;
- unique frame IDs, relative paths contained within the fixture, and matching
  SHA-256 values;
- finite, positive-area full-frame `xyxy` boxes inside the image;
- every annotated person's bbox-bottom midpoint lies inside the configured
  semantic station polygon.

Annotate every in-zone person manually. Empty `persons` is valid only when a
human verified that the frame contains no in-zone person. Record annotator,
date, source clip checksums, frame-sampling rule, ROI/margin rationale, and any
ambiguous/occluded cases in `METHODOLOGY.md`. Never seed the final truth from
one benchmark arm without manual review; doing so would bias selection toward
that arm.

Films 1 and 2 use the same manifest shape but may contain one recording window;
they are whole-frame detection-recall regression fixtures.

## Run each arm in an isolated process

Run on `cctv-vps` only after commit → push → VPS pull. Each arm is a separate
process so per-process VRAM and model allocator state cannot leak between arms.
Use the same ordered `--throughput-clip` arguments for all three commands.

The host development environment does not include the VLM stack. Run inside
the pinned GPU-service image while bind-mounting only the current source and
fixtures; mounting the whole repository over `/app` would hide the image's
`.venv`. Use `--pid=host` so the process PID reported by Python matches the PID
reported by `nvidia-smi` for the per-process VRAM sampler. Select an idle GPU
before the run; the measured issue #86 run used identical RTX 5070 GPU 1.

```bash
export DOCKER_HOST=unix:///var/run/docker.sock
IMAGE=ghcr.io/kopalniekrypto/cctv-gpu-engine/gpu-service@sha256:332b8b1cda232519b19341d9f894c8b0a73c68d026964320f041e8120ef2ef81

docker run --rm --gpus device=1 --pid=host --ipc=host \
  --entrypoint /app/.venv/bin/python \
  -v "$PWD/pipeline:/app/pipeline:ro" \
  -v "$PWD/benchmarks:/app/benchmarks:ro" \
  -v "$PWD/models:/app/models:ro" \
  -v "$PWD/benchmark-results:/app/benchmark-results" \
  -v cctv-hf-cache:/root/.cache/huggingface \
  "$IMAGE" -u -m pipeline.pose_benchmark run-arm \
  --arm baseline_640 \
  --fixture benchmarks/pose-resolution/bending-pilot-v1/manifest.json \
  --zones benchmarks/pose-resolution/bending-pilot-v1/zones.json \
  --model models/yolo11s-pose-640-issue86.onnx \
  --throughput-clip benchmarks/pose-resolution/bending-pilot-v1/clips/window-1.mp4 \
  --throughput-clip benchmarks/pose-resolution/bending-pilot-v1/clips/window-2.mp4 \
  --throughput-clip benchmarks/pose-resolution/bending-pilot-v1/clips/window-3.mp4 \
  --film-1-fixture benchmarks/pose-resolution/films-v1/film-1/manifest.json \
  --film-2-fixture benchmarks/pose-resolution/films-v1/film-2/manifest.json \
  --output benchmark-results/baseline_640.json
```

Repeat in fresh containers with
`--arm full_frame_1280 --model models/yolo11s-pose-1280-issue86.onnx` and with
`--arm focused_roi_640 --model models/yolo11s-pose-640-issue86.onnx`.

Every long stage emits `BENCHMARK_HEARTBEAT` with flushed output at least once
per 60 seconds. Detection partial JSON is checkpointed after every frame;
end-to-end partial JSON is checkpointed on every pipeline progress callback.
VRAM is sampled from `nvidia-smi --query-compute-apps=pid,used_gpu_memory` for
the benchmark PID, not inferred from model sizes or total GPU occupancy.

## Select and assert

```bash
uv run python -m pipeline.pose_benchmark select \
  --fixture benchmarks/pose-resolution/bending-pilot-v1/manifest.json \
  --arm-result benchmark-results/baseline_640.json \
  --arm-result benchmark-results/full_frame_1280.json \
  --arm-result benchmark-results/focused_roi_640.json \
  --output benchmark-results/issue-86/selection.json

POSE_BENCHMARK_RESULTS=benchmark-results/issue-86/selection.json \
  uv run pytest -m perf pipeline/pose_benchmark_perf_test.py
```

The result records raw per-frame detections, TP/FP/FN, precision/recall/F1,
pose p50/p95, measured VLM wallclock, processed frame count, measured clip
duration, the explicit linear extrapolation formula/assumption, per-process
peak VRAM, Films 1+2 recall, and every eligibility check.

If no arm qualifies, file the required narrow follow-up first and rerun
`select` with `--follow-up-issue <URL>`. No production default changes. If a
non-baseline arm wins, promote only that arm, rerun the worker-path validation,
then regenerate the final artifact with `--production-default-changed`.

## Native-resolution tiling arms (issue #110)

#101 proved the effective detection floor is ~60 px of person height at model
input, and that the 80–120 px native band (90 of 296 people in `magazyn-hall-v1`)
is 0% at every full-frame arm because a 3840 px frame downscaled to a 1280 px
input shrinks those people below the floor. Tiling is the only lever left: crop
the frame into overlapping **native-resolution** 1280×736 tiles (a native tile
letterboxes at scale 1.0, so an 80–120 px person keeps their pixels), detect each
tile, translate back to full-frame coordinates, and merge across the overlaps
with Intersection-over-Smaller dedup.

Two arms, same fixture and whole-frame scoring zone as #101:

- `tiled_1280x736` — tile the whole frame (grid). Reaches every band, pays for
  the most tiles.
- `tiled_zones_1280x736` — tile only inside the authored zone bounding boxes
  (`--roi-zones`), the production-shaped variant. Scoring stays whole-frame, so
  a person outside the tiled zones is an honest miss.

The subcommand is **detector-isolated** (#110): it scores whole-frame recall,
recall-by-native-height, per-process peak VRAM, and the pose-only per-hour cost
(`mean(pose_wallclock_s) * fps * 3600 / 60` — N tiles per frame are summed into
each sample). No end-to-end VLM run: the tiling pipeline is not wired end-to-end
yet (a follow-up), and the VLM cost scales with `total_detections`, read apart.
The reused 1280×736 model is already baked into the image at
`/app/models/yolo11s-pose-1280x736.onnx` (sha `7ee0fcd8…`).

Run each arm in a fresh container on an idle GPU (same pattern as above,
`--pid=host` for the per-PID VRAM sampler):

```bash
export DOCKER_HOST=unix:///var/run/docker.sock
IMAGE=ghcr.io/kopalniekrypto/cctv-gpu-engine/gpu-service:latest

docker run --rm --gpus device=1 --pid=host --ipc=host \
  --entrypoint /app/.venv/bin/python \
  -v "$PWD/pipeline:/app/pipeline:ro" \
  -v "$PWD/benchmarks:/app/benchmarks:ro" \
  -v "$PWD/benchmark-results:/app/benchmark-results" \
  "$IMAGE" -u -m pipeline.pose_benchmark run-tiling-arm \
  --arm tiled_1280x736 \
  --fixture benchmarks/pose-resolution/magazyn-hall-v1/manifest.json \
  --zones benchmarks/pose-resolution/magazyn-hall-v1/zones.json \
  --model /app/models/yolo11s-pose-1280x736.onnx \
  --overlap 0.2 \
  --output benchmark-results/issue-110/tiled_1280x736.json
```

For the with-zones arm add `--arm tiled_zones_1280x736` and
`--roi-zones benchmarks/pose-resolution/magazyn-hall-v1/zones-roi.json` (the
authored sub-frame zones mirrored from the production camera). Each arm's JSON
records `recall_by_height`, `quality`, `detector_cost`, `peak_process_gpu_vram_mb`,
and raw per-frame detections. There is no `select` step — the tiling arms are a
measurement against #101's rows, not a production-winner election.
