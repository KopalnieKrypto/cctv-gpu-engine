# Issue #123 — station classifier, measured cost

`cctv-vps` (RTX 5070, GPU 1, 12227 MiB), image
`ghcr.io/kopalniekrypto/cctv-gpu-engine/gpu-service:latest`, engine at `ef1e694`,
2026-09-02. Clip: `W1-2026-08-28T0700Z.mp4`, 3840×2160, 1198 s, 599 samples at
the card's 2 s stride. Peak VRAM by the #86 method — per-PID `nvidia-smi
--query-compute-apps`, sampled every 0.5 s — because the ONNX sessions' footprint
is the whole cost here and an allocator-level figure would miss it.

## The figures

| | |
|---|---|
| **GPU-seconds per video-hour** | **521.3** |
| **Peak VRAM** | **754 MiB** |
| Wall clock | 173.5 s for 1198 s of video (6.9:1) |
| Coverage | 599 / 599 samples |

754 MiB puts this beside the heuristic arm's ~710 MiB and an order of magnitude
below the VLM's 7,866 MiB, which is the expected shape: two ONNX sessions, no
pose session, no OSNet, no VLM.

## The issue quotes 49 GPU-seconds per video-hour. Both numbers are right.

They measure different things, and the difference is worth stating plainly rather
than letting one figure quietly replace the other.

| Stage | s / video-hour | How |
|---|---|---|
| ffmpeg decode | **386** | measured |
| preprocess (crop → resize → centre-crop → normalise) | 27 | measured |
| backbone, batch 1 | 12 | measured |
| remainder (head sliding window, array handling) | ~96 | by difference |
| **total** | **521** | measured end to end |

The 49 is the benchmark arm's **embedding pass over pre-extracted JPEG crops**,
which never opens a video. The model work here is 27 + 12 = **39 s/video-hour**,
consistent with it. What the benchmark had no reason to include is decoding
twenty minutes of 4K, and that is **386 s/video-hour — 74% of the run**.

Two things follow, and both were measured rather than assumed:

- **Sampling less does not decode less.** Decoding at 1 fps costs 386 s/video-hour
  and decoding at the 2 s stride costs 381. ffmpeg's `fps` filter drops frames
  *after* decoding them, so the stride buys nothing.
- **Batching the backbone is not the lever.** Batch 64 takes it from 12 to 10
  s/video-hour. The backbone is 2% of the run; optimising it would be work spent
  on the smallest term.

The decode cost is paid by every classifier on this footage, so it is not a
property of this arm. Cutting it means hardware-accelerated decode (NVDEC) in
`pipeline/video_frames.py`, which changes every mode and belongs in its own issue
rather than in the one that happened to measure it.

## What the run produced, and what it is not evidence of

| category | total | share | measured time ratio |
|---|---|---|---|
| `spawanie` | 479 s | 39.98% | 1.062 |
| `ukladanie_pretow` | 363 s | 30.30% | 1.100 |
| `pozostale` | 356 s | 29.72% | 0.841 |
| `nierozpoznane` | 0 s | 0.00% | 0.500 |

56 intervals, boundaries at the sample midpoint, coverage 1.0.

**W1 is one of the three windows the shipped head trained on.** These totals are
in-sample. They are evidence that the path runs end to end and emits a
well-formed section — nothing more. Every accuracy figure belongs to the
cross-validated folds and is quoted, with its provenance, in the model card; the
`time_ratio` column above comes from there and not from this run.

## Files

- `cost.json` — the figures above, machine-readable
- `station-W1-result.json` — the artefact the run produced
