# CCTV GPU Engine — developer shortcuts.
#
# Two install profiles, picked at sync time so the default `uv sync` stays
# lightweight (~50MB) instead of pulling ~1.5GB of CUDA wheels every time:
#
#   make sync-dev   → CPU-only onnxruntime stub, suitable for unit tests on
#                     macOS / dev boxes (use this if you don't have an NVIDIA GPU)
#   make sync-gpu   → real onnxruntime-gpu + nvidia-cublas-cu12, Linux+CUDA only
#                     (used inside the gpu-service container and on cctv-vps)
#
# See GitHub issue #9 for the rationale.

# Default fixtures for the GPU smoke test. Override on the command line, e.g.
#   make test-gpu TEST_VIDEO=test-data/cctv-1h.mp4 TEST_TIMESTAMP=120
TEST_VIDEO ?= test-data/sample.mp4
TEST_TIMESTAMP ?= 5
MODEL ?= models/yolo11n-pose.onnx

.PHONY: sync-dev sync-gpu test test-gpu

sync-dev:
	uv sync --extra cpu-stub

sync-gpu:
	uv sync --extra gpu

test:
	uv run pytest

test-gpu:
	@test -f $(MODEL) || { echo "ERROR: $(MODEL) missing — run ./setup-models.sh"; exit 1; }
	@test -f $(TEST_VIDEO) || { echo "ERROR: $(TEST_VIDEO) missing — set TEST_VIDEO=path/to/video.mp4"; exit 1; }
	uv run python -m pipeline.analyze $(TEST_VIDEO) --timestamp $(TEST_TIMESTAMP) --model $(MODEL)
