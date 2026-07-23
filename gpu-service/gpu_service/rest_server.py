"""GPU-agent REST server entry point (issue #25).

Wires Flask, the task registry, a ThreadPoolExecutor for background work,
and the SIGTERM cleanup hook. Used as the Docker ENTRYPOINT in the
gpu-exchange-side variant of this image.

Production wiring (``main``):

* warm up YOLO + Qwen2.5-VL-3B in-process, then ``readiness.mark_ready()``
* install SIGTERM handler → :func:`terminate_running_tasks` and graceful
  Flask shutdown
* serve ``create_app`` on ``:5003``

The dispatch closure (``make_dispatcher``) is split out so it can be unit
tested without spawning Flask. The executor is also injected so tests can
pass a synchronous "inline" executor.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Protocol

from waitress import serve

from gpu_service.ffmpeg_concat import ffmpeg_concat
from gpu_service.http_client import PresignedHttpClient
from gpu_service.rest_api import (
    Readiness,
    TaskRegistry,
    create_app,
    terminate_running_tasks,
)
from gpu_service.task_runner import ConcatFn, HttpClientLike, PipelineFn, run_task
from gpu_service.vram_preflight import preflight_or_exit
from pipeline.zones import ZoneConfig

logger = logging.getLogger(__name__)

# Baked pose exports a camera may select per-task via zones.json `pose.input_size`
# (#109). Values are filenames resolved NEXT TO the configured MODEL_PATH, so a
# MODEL_PATH override keeps finding its siblings and nothing hardcodes /app/models
# twice. Keep in sync with the Dockerfile bakes; the platform allowlists the same
# two keys (data-ops ZonesConfigPoseSchema).
POSE_MODEL_FILENAME_BY_INPUT_SIZE = {
    "640x640": "yolo11s-pose.onnx",
    "1280x736": "yolo11s-pose-1280x736.onnx",
}


def _read_pose_input_size(config_path: str | Path) -> str | None:
    """Read `pose.input_size` from a mounted zones.json, tolerantly.

    Returns ``None`` on a missing file, unreadable bytes, non-JSON, or any shape
    that is not a string `pose.input_size`. Model selection must not depend on
    the config being fully valid — the task runner re-parses and validates the
    whole document per task, and a shift/zones problem there must not blank the
    detector choice here.
    """
    try:
        data = json.loads(Path(config_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    pose = data.get("pose")
    if not isinstance(pose, dict):
        return None
    input_size = pose.get("input_size")
    return input_size if isinstance(input_size, str) else None


def resolve_pose_model_path(env_default: str, config_path: str | Path) -> str:
    """Pick the pose model at container startup from the task's zones.json (#109).

    A camera's config may request a baked non-default export via
    ``pose.input_size``; point ``MODEL_PATH`` at the matching weights, resolved
    beside ``env_default``. Any absent / malformed / unknown selector falls back
    to ``env_default`` — the engine treats the size as an allowlist even though
    the platform already validates it, so a hand-written config cannot select a
    shape the image does not bake.
    """
    input_size = _read_pose_input_size(config_path)
    if input_size is None:
        return env_default
    filename = POSE_MODEL_FILENAME_BY_INPUT_SIZE.get(input_size)
    if filename is None:
        logger.warning(
            "unknown pose input_size %r in zones config; using default model", input_size
        )
        return env_default
    return str(Path(env_default).parent / filename)


Payload = dict[str, Any]


class ExecutorLike(Protocol):
    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any: ...


def make_dispatcher(
    *,
    registry: TaskRegistry,
    workdir_root: Path,
    http: HttpClientLike,
    concat: ConcatFn,
    pipeline: PipelineFn,
    executor: ExecutorLike,
) -> Callable[[Payload], None]:
    """Bind the runner's deps into a single ``dispatch(payload)`` closure.

    Each task gets its own subdirectory under ``workdir_root`` so a failed
    task's partial files don't collide with the next one — and an operator
    can grep through them for forensics until the container is restarted.
    """

    def dispatch(payload: Payload) -> None:
        task_id = payload["task_id"]
        task_workdir = workdir_root / task_id
        task_workdir.mkdir(parents=True, exist_ok=True)

        executor.submit(
            run_task,
            payload=payload,
            registry=registry,
            workdir=task_workdir,
            http=http,
            concat=concat,
            pipeline=pipeline,
        )

    return dispatch


def _warm_up_pipeline(
    model_path: str,
    classifier: str,
    activity_model_path: str,
    activity_model_metadata_path: str,
) -> PipelineFn:
    """Load YOLO and return the zones-aware configured pipeline closure.

    Imported lazily so unit tests on macOS never touch onnxruntime-gpu.
    """
    # Lazy imports — these modules pull onnxruntime-gpu and torch (cu128).
    # Canonical artifact is result.json (issue #72); the gpu-agent uploads
    # these bytes to the presigned result URL for the platform to render.
    from pipeline.analyze import run_full_video_to_json

    def pipeline_fn(
        chunks: list[Path],
        progress: Callable[[int], None],
        *,
        zones: ZoneConfig | None = None,
    ) -> bytes:
        return run_full_video_to_json(
            chunks=chunks,
            progress=progress,
            model_path=model_path,
            classifier=classifier,
            activity_model_path=activity_model_path,
            activity_model_metadata_path=activity_model_metadata_path,
            zones=zones,
        )

    # Touch the detector once to warm CUDA / cuDNN — fail fast if GPU is
    # missing instead of letting the first /analyze 502 the agent.
    from pipeline.pose_detector import load_pose_model

    load_pose_model(model_path)
    return pipeline_fn


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    port = int(os.environ.get("REST_PORT", "5003"))
    # Per-camera detector selection (#109): the gpu-agent mounts the task's
    # zones.json before this container starts, so the model is resolved here,
    # once, from that config — the same one-container-per-task boundary the model
    # is already warmed at. Absent config → the baked 640 default.
    env_model_path = os.environ.get("MODEL_PATH", "/app/models/yolo11s-pose.onnx")
    zones_config_path = os.environ.get("ZONES_CONFIG_PATH") or "/config/zones.json"
    model_path = resolve_pose_model_path(env_model_path, zones_config_path)
    if model_path != env_model_path:
        logger.info(
            "pose model selected from zones config: %s (default was %s)", model_path, env_model_path
        )
    classifier = os.environ.get("CLASSIFIER", "vlm")
    activity_model_path = os.environ.get(
        "ACTIVITY_MODEL_PATH", "/app/models/activity-mlp-v1.0.0.onnx"
    )
    activity_model_metadata_path = os.environ.get(
        "ACTIVITY_MODEL_METADATA_PATH", "/app/models/activity-mlp-v1.0.0.json"
    )
    workdir_root = Path(os.environ.get("WORKDIR", "/tmp/cctv-jobs"))
    workdir_root.mkdir(parents=True, exist_ok=True)

    # Issue #43 — fail-fast on insufficient VRAM before warming any model.
    # Without this, a busy GPU produces a mid-load PyTorch OOM trace that
    # masks the real cause (another CUDA process holding the card).
    preflight_or_exit(
        classifier=classifier,
        env_override=os.environ.get("VRAM_BUDGET_MB"),
    )

    readiness = Readiness()
    registry = TaskRegistry()
    executor = ThreadPoolExecutor(max_workers=1)
    http = PresignedHttpClient()

    logger.info("warming up pipeline (model=%s classifier=%s)", model_path, classifier)
    pipeline_fn = _warm_up_pipeline(
        model_path,
        classifier,
        activity_model_path,
        activity_model_metadata_path,
    )
    readiness.mark_ready()
    logger.info("readiness flipped — accepting /analyze")

    dispatch = make_dispatcher(
        registry=registry,
        workdir_root=workdir_root,
        http=http,
        concat=ffmpeg_concat,
        pipeline=pipeline_fn,
        executor=executor,
    )

    def _on_sigterm(signum: int, frame: Any) -> None:
        logger.warning("SIGTERM received — flipping running tasks to failed")
        terminate_running_tasks(registry)
        # Graceful executor shutdown — let in-flight uploads finish if they
        # can, but don't block forever (docker stop grace = 10s by default).
        executor.shutdown(wait=False, cancel_futures=True)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _on_sigterm)
    signal.signal(signal.SIGINT, _on_sigterm)

    app = create_app(readiness=readiness, registry=registry, dispatch=dispatch)
    logger.info("serving REST contract on :%d", port)
    # Production WSGI server (waitress), not Werkzeug's dev server (issue #63).
    # The gpu-agent depends on this port; the dev server is single-process with
    # weaker slow-client/timeout handling and a noisy warning banner. waitress
    # runs in the main thread so the SIGTERM handler above still fires here —
    # sys.exit(0) unwinds serve() the same way it unwound app.run().
    serve(app, host="0.0.0.0", port=port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
