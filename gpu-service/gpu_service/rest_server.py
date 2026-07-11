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

import logging
import os
import signal
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Protocol

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

logger = logging.getLogger(__name__)

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


def _warm_up_pipeline(model_path: str, classifier: str) -> PipelineFn:
    """Load YOLO + VLM, return the pipeline closure for run_task.

    Imported lazily so unit tests on macOS never touch onnxruntime-gpu.
    """
    # Lazy imports — these modules pull onnxruntime-gpu and torch (cu128).
    # Canonical artifact is result.json (issue #72); the gpu-agent uploads
    # these bytes to the presigned result URL for the platform to render.
    from pipeline.analyze import run_full_video_to_json

    def pipeline_fn(chunks: list[Path], progress: Callable[[int], None]) -> bytes:
        return run_full_video_to_json(
            chunks=chunks,
            progress=progress,
            model_path=model_path,
            classifier=classifier,
        )

    # Touch the detector once to warm CUDA / cuDNN — fail fast if GPU is
    # missing instead of letting the first /analyze 502 the agent.
    from pipeline.pose_detector import load_pose_model

    load_pose_model(model_path)
    return pipeline_fn


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    port = int(os.environ.get("REST_PORT", "5003"))
    model_path = os.environ.get("MODEL_PATH", "/app/models/yolo11s-pose.onnx")
    classifier = os.environ.get("CLASSIFIER", "vlm")
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
    pipeline_fn = _warm_up_pipeline(model_path, classifier)
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
    app.run(host="0.0.0.0", port=port, threaded=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
