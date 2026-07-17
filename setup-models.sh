#!/usr/bin/env bash
# Download the canonical ONNX models into ./models/.
#
# Three model families are pinned here. YOLO and OSNet are required for a
# default pipeline run; the activity MLP is fetched for the opt-in issue #34
# runtime and remains experimental because its frozen quality gate failed:
#
#   1. YOLO11s-pose v1.0 — person detection + COCO keypoints (38 MB,
#      ~150 ms/frame on RTX 5070, noticeably better detection than nano).
#   2. OSNet x0_25 (MSMT17) v1.0 — person Re-ID appearance embedding for
#      tracking (0.7 MB, issue #32). Tracking is on by default, so the
#      pipeline will not start without it. `--no-tracker` skips its use.
#
# Both are pinned to versioned GitHub releases so every checkout lands the
# *same* weights — `yolo export` from ultralytics would otherwise resolve to
# whatever upstream tagged latest at fetch time.
#
# Need the smaller `n` pose variant (12 MB, ~70 ms/frame) for low-latency runs?
# Override the env vars:
#   MODEL_TAG=yolo11n-pose-v1.0 \
#   MODEL_FILE=yolo11n-pose.onnx \
#   MODEL_SHA256=70bd721f9cb797eb44cbc70bc65213397a0a26da38fe6fd5ccdf699016d33d3c \
#   ./setup-models.sh
#
# Want an even bigger pose model (m / l / x) for higher accuracy? Don't use
# this script — see README "Using a different model size" for the
# `yolo export` one-liner. The pipeline only assumes the standard
# YOLO11*-pose output layout ([1,56,N] — 4 bbox + 1 conf + 17×3 keypoints),
# so any yolo11{n,s,m,l,x}-pose ONNX exported at imgsz=640 is a drop-in.

set -euo pipefail

# Pin both the release tag AND the expected sha256 so an unexpected file
# swap on the GitHub release (compromise, accidental re-upload) fails the
# script instead of silently substituting weights.
MODEL_TAG="${MODEL_TAG:-yolo11s-pose-v1.0}"
MODEL_FILE="${MODEL_FILE:-yolo11s-pose.onnx}"
MODEL_SHA256="${MODEL_SHA256:-469beac503fdc788ea3980331bc4bfbd2bd00de3772eb0984f4c53032740583f}"
MODEL_URL="${MODEL_URL:-https://github.com/KopalnieKrypto/cctv-gpu-engine/releases/download/${MODEL_TAG}/${MODEL_FILE}}"

# OSNet Re-ID (issue #32). Exported from the OSNet author's MSMT17 weights
# (huggingface.co/kaiyangzhou/osnet, MIT) at opset 18 with a dynamic batch
# axis — the embedder sends one crop per person per frame, so batch varies.
OSNET_TAG="${OSNET_TAG:-osnet-x0_25-v1.0}"
OSNET_FILE="${OSNET_FILE:-osnet_x0_25.onnx}"
OSNET_SHA256="${OSNET_SHA256:-86f6314fc903d6b3c7e90c8f4dc5f438fb640faf2840574f3170b749c4765ce6}"
OSNET_URL="${OSNET_URL:-https://github.com/KopalnieKrypto/cctv-gpu-engine/releases/download/${OSNET_TAG}/${OSNET_FILE}}"

# Per-person activity MLP (issue #34). This prerelease artifact is available
# for reproducible evaluation and explicit ``--classifier mlp`` runs. VLM
# remains the deployed default because the frozen held-out quality gate failed.
ACTIVITY_MLP_TAG="${ACTIVITY_MLP_TAG:-activity-mlp-v1.0.0}"
ACTIVITY_MLP_FILE="${ACTIVITY_MLP_FILE:-activity-mlp-v1.0.0.onnx}"
ACTIVITY_MLP_SHA256="${ACTIVITY_MLP_SHA256:-4835d97e368567838d2c6ba2ccaf329ee541de283cfa377e72188783ac89cd67}"
ACTIVITY_MLP_URL="${ACTIVITY_MLP_URL:-https://github.com/KopalnieKrypto/cctv-gpu-engine/releases/download/${ACTIVITY_MLP_TAG}/${ACTIVITY_MLP_FILE}}"
ACTIVITY_MLP_METADATA_FILE="${ACTIVITY_MLP_METADATA_FILE:-activity-mlp-v1.0.0.json}"
ACTIVITY_MLP_METADATA_SHA256="${ACTIVITY_MLP_METADATA_SHA256:-d387b156934d8498e3afc0554324959164327848dd9fc57e1d507da9f789d8f4}"
ACTIVITY_MLP_METADATA_URL="${ACTIVITY_MLP_METADATA_URL:-https://github.com/KopalnieKrypto/cctv-gpu-engine/releases/download/${ACTIVITY_MLP_TAG}/model-metadata.json}"

DEST_DIR="models"

mkdir -p "${DEST_DIR}"

sha256_of() {
    # Cross-platform sha256: macOS ships `shasum`, Linux ships `sha256sum`.
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}

fetch_model() {
    local url="$1" file="$2" want_sha="$3" var_name="$4"
    local dest="${DEST_DIR}/${file}"

    # Symlinks bite us inside containers: a host symlink like
    #   models/yolo11n-pose.onnx → /home/foo/video-test/yolo11n-pose.onnx
    # resolves on the host (where `-f` returns true) but the symlink target
    # doesn't exist inside the gpu-service container, so onnxruntime fails
    # with NO_SUCHFILE at session init. Replace any symlink with a real
    # copy of its target so the bind-mounted ./models/ stays self-contained.
    # Discovered during issue #8 e2e validation on cctv-vps.
    if [[ -L "${dest}" ]]; then
        local target
        target="$(readlink -f "${dest}" || true)"
        if [[ -n "${target}" && -f "${target}" ]]; then
            echo "→ ${dest} is a symlink → ${target}; replacing with a real copy"
            rm "${dest}"
            cp "${target}" "${dest}"
        else
            echo "→ ${dest} is a dangling symlink; removing so curl can re-fetch"
            rm "${dest}"
        fi
    fi

    if [[ -f "${dest}" ]]; then
        echo "→ ${dest} already present, verifying sha256…"
    else
        echo "→ Downloading ${url}"
        # -f: fail on HTTP errors instead of writing the error page to disk.
        # -L: follow redirects (GH release assets always redirect to S3).
        curl -fL --progress-bar "${url}" -o "${dest}"
    fi

    local actual
    actual="$(sha256_of "${dest}")"
    if [[ "${actual}" != "${want_sha}" ]]; then
        echo "ERROR: sha256 mismatch for ${dest}" >&2
        echo "  expected: ${want_sha}" >&2
        echo "  actual:   ${actual}" >&2
        echo "Delete ${dest} and re-run to retry, or override ${var_name} if intentional." >&2
        exit 1
    fi

    echo "✓ ${dest} ready ($(du -h "${dest}" | awk '{print $1}'), sha256 verified)"
}

fetch_model "${MODEL_URL}" "${MODEL_FILE}" "${MODEL_SHA256}" "MODEL_SHA256"
fetch_model "${OSNET_URL}" "${OSNET_FILE}" "${OSNET_SHA256}" "OSNET_SHA256"
fetch_model "${ACTIVITY_MLP_URL}" "${ACTIVITY_MLP_FILE}" "${ACTIVITY_MLP_SHA256}" "ACTIVITY_MLP_SHA256"
fetch_model "${ACTIVITY_MLP_METADATA_URL}" "${ACTIVITY_MLP_METADATA_FILE}" "${ACTIVITY_MLP_METADATA_SHA256}" "ACTIVITY_MLP_METADATA_SHA256"
