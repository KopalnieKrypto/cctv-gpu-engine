#!/usr/bin/env bash
# Download the canonical YOLO pose ONNX model into ./models/.
#
# Default: YOLO11n-pose v1.0 (12 MB nano model, fast — fine for prototyping
# on RTX 5070). Pinned to a versioned GitHub release so every checkout of
# the repo lands the *same* weights — `yolo export` from ultralytics would
# otherwise resolve to whatever upstream tagged latest at fetch time.
#
# Want a bigger model (s / m / l / x) for higher accuracy? Don't use this
# script — see README "Using a different model size" for the
# `yolo export` one-liner. The pipeline only assumes the standard YOLO11*-pose
# output layout ([1,56,N] — 4 bbox + 1 conf + 17×3 keypoints), so any
# yolo11{n,s,m,l,x}-pose ONNX exported at imgsz=640 will work as a drop-in.

set -euo pipefail

# Pin both the release tag AND the expected sha256 so an unexpected file
# swap on the GitHub release (compromise, accidental re-upload) fails the
# script instead of silently substituting weights.
MODEL_TAG="${MODEL_TAG:-yolo11n-pose-v1.0}"
MODEL_FILE="${MODEL_FILE:-yolo11n-pose.onnx}"
MODEL_SHA256="${MODEL_SHA256:-70bd721f9cb797eb44cbc70bc65213397a0a26da38fe6fd5ccdf699016d33d3c}"
MODEL_URL="${MODEL_URL:-https://github.com/KopalnieKrypto/cctv-gpu-engine/releases/download/${MODEL_TAG}/${MODEL_FILE}}"

DEST_DIR="models"
DEST="${DEST_DIR}/${MODEL_FILE}"

mkdir -p "${DEST_DIR}"

if [[ -f "${DEST}" ]]; then
    echo "→ ${DEST} already present, verifying sha256…"
else
    echo "→ Downloading ${MODEL_URL}"
    # -f: fail on HTTP errors instead of writing the error page to disk.
    # -L: follow redirects (GH release assets always redirect to S3).
    curl -fL --progress-bar "${MODEL_URL}" -o "${DEST}"
fi

# Cross-platform sha256: macOS ships `shasum`, Linux ships `sha256sum`.
if command -v sha256sum >/dev/null 2>&1; then
    actual="$(sha256sum "${DEST}" | awk '{print $1}')"
else
    actual="$(shasum -a 256 "${DEST}" | awk '{print $1}')"
fi

if [[ "${actual}" != "${MODEL_SHA256}" ]]; then
    echo "ERROR: sha256 mismatch for ${DEST}" >&2
    echo "  expected: ${MODEL_SHA256}" >&2
    echo "  actual:   ${actual}" >&2
    echo "Delete ${DEST} and re-run to retry, or override MODEL_SHA256 if intentional." >&2
    exit 1
fi

echo "✓ ${DEST} ready ($(du -h "${DEST}" | awk '{print $1}'), sha256 verified)"
