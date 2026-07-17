"""Task runner — pulls the gpu-agent request through the pipeline (issue #25).

Sequence:

1. Load the server-owned zones config from ``ZONES_CONFIG_PATH`` (default
   ``/config/zones.json``) when the gpu-agent mounted one; absence is un-gated.
2. Download each ``input_presigned_urls`` entry into ``workdir/inputs/``.
3. If more than one chunk → ffmpeg-concat into a single MP4. One chunk →
   skip concat and feed the downloaded file straight to the pipeline.
4. Run the YOLO + VLM pipeline on the concatenated chunks. Pipeline
   progress callbacks bubble through to :class:`TaskRegistry` so the
   gpu-agent's ``/status/:id`` polls see live progress.
5. PUT the canonical result.json (issue #72) to ``result_presigned_url``.
6. Flip registry to ``state: completed``.

Collaborators are injected:

* ``http`` exposes ``download(url, dest_path)`` and ``upload(url, body_bytes)``.
* ``concat(inputs: list[Path], output: Path)`` is a function — production
  wraps ffmpeg, tests pass a noop closure.
* ``pipeline(chunks, progress, *, zones=None) -> bytes`` matches
  ``pipeline.analyze.run_full_video_to_json``'s relevant signature.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from gpu_service.rest_api import TaskRegistry
from pipeline.zones import ZoneConfig

Payload = dict[str, Any]
ProgressCallback = Callable[[int], None]
ConcatFn = Callable[[list[Path], Path], None]
DEFAULT_ZONES_CONFIG_PATH = Path("/config/zones.json")


class PipelineFn(Protocol):
    def __call__(
        self,
        chunks: list[Path],
        progress: ProgressCallback,
        *,
        zones: ZoneConfig | None = None,
    ) -> bytes: ...


class HttpClientLike(Protocol):
    def download(self, url: str, dest: Path) -> None: ...
    def upload(self, url: str, body: bytes) -> None: ...


def run_task(
    *,
    payload: Payload,
    registry: TaskRegistry,
    workdir: Path,
    http: HttpClientLike,
    concat: ConcatFn,
    pipeline: PipelineFn,
    zones_config_path: Path | None = None,
) -> None:
    """Process one /analyze task end-to-end.

    Any exception in any stage (download, concat, pipeline, upload) is
    caught and translated into ``state: failed`` with the exception string
    as ``error``. The gpu-agent polls ``/status/:id`` and surfaces this
    error to the platform so the operator sees a real cause instead of a
    hung "running" forever. ``zones_config_path`` is injectable for tests;
    production resolves ``ZONES_CONFIG_PATH`` or the gpu-agent contract path.
    """
    task_id = payload["task_id"]
    input_urls: list[str] = payload["input_presigned_urls"]
    result_url: str = payload["result_presigned_url"]

    try:
        resolved_zones_path = zones_config_path
        if resolved_zones_path is None:
            resolved_zones_path = Path(
                os.environ.get("ZONES_CONFIG_PATH") or DEFAULT_ZONES_CONFIG_PATH
            )
        zones = ZoneConfig.load(resolved_zones_path) if resolved_zones_path.exists() else None
        registry.set_running(task_id, progress=0.0)

        inputs_dir = workdir / "inputs"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        downloaded: list[Path] = []
        for i, url in enumerate(input_urls, start=1):
            dest = inputs_dir / f"chunk_{i:03d}.mp4"
            http.download(url, dest)
            downloaded.append(dest)

        if len(downloaded) > 1:
            concat_output = workdir / "concat.mp4"
            concat(downloaded, concat_output)
            pipeline_inputs = [concat_output]
        else:
            pipeline_inputs = downloaded

        def progress_cb(pct: int) -> None:
            registry.set_progress(task_id, progress=pct / 100.0)

        if zones is None:
            result_bytes = pipeline(pipeline_inputs, progress_cb)
        else:
            result_bytes = pipeline(pipeline_inputs, progress_cb, zones=zones)
        http.upload(result_url, result_bytes)
        registry.set_completed(task_id)
    except Exception as e:
        registry.set_failed(task_id, error=f"{type(e).__name__}: {e}")
