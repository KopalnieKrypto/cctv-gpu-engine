"""GPU service worker — orchestrates one R2 job through the pipeline.

This module is the orchestration layer for issue #5. It is intentionally
boundary-driven: the only collaborator it talks to is an R2 client (duck-typed,
so tests can pass an in-memory fake), and the pipeline is injected as a plain
callable. That keeps the worker testable without a GPU, ffmpeg, or boto3.

The R2 client interface the worker depends on:

* ``list_pending_job_ids() -> list[str]``
* ``get_status(job_id) -> dict | None``
* ``put_status(job_id, status) -> None``
* ``download_chunks(job_id, dest: Path) -> list[Path]``
* ``upload_report(job_id, html: bytes) -> str``  (returns the R2 key)

The pipeline interface is::

    pipeline(chunks: list[Path], progress: ProgressCallback) -> bytes

where ``ProgressCallback`` accepts an int 0-100 to be persisted into
``status.json``.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int], None]
PipelineFn = Callable[[list[Path], ProgressCallback], bytes]
NowFn = Callable[[], str]


class R2ClientLike(Protocol):
    """Structural type for the R2 surface the worker depends on."""

    def list_pending_job_ids(self) -> list[str]: ...
    def get_status(self, job_id: str) -> dict[str, Any] | None: ...
    def put_status(self, job_id: str, status: dict[str, Any]) -> None: ...
    def download_chunks(self, job_id: str, dest: Path) -> list[Path]: ...
    def upload_report(self, job_id: str, html: bytes) -> str: ...


def _utc_now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _fail_job(
    client: R2ClientLike,
    job_id: str,
    fallback_status: dict[str, Any],
    error: str,
    now: NowFn,
) -> str:
    """Mark a job as failed with the given error message and return ``"failed"``.

    The latest server-side status is preferred (in case progress was written
    mid-flight) and ``fallback_status`` is used only if the read fails.
    """
    failed = client.get_status(job_id) or fallback_status
    failed["status"] = "failed"
    failed["error"] = error
    failed["updated_at"] = now()
    client.put_status(job_id, failed)
    return "failed"


def process_job(
    client: R2ClientLike,
    job_id: str,
    worker_id: str,
    pipeline: PipelineFn,
    workdir: Path,
    now: NowFn = _utc_now_iso,
) -> str:
    """Run a single job's lifecycle. Returns the final outcome.

    Outcomes:

    * ``"completed"`` — pipeline ran, report uploaded, status finalized.
    * ``"skipped"``   — status missing or no longer ``pending`` (claim race).
    * ``"failed"``    — pipeline raised; status set to ``failed`` with error.

    The worker never raises out of this function for expected errors — those
    are recorded into ``status.json`` so the polling loop can keep going.
    """
    status = client.get_status(job_id)
    if status is None or status.get("status") != "pending":
        return "skipped"

    # Claim: write `processing` + worker_id. MVP last-writer-wins per SPEC §6.5.
    status["status"] = "processing"
    status["worker_id"] = worker_id
    status["updated_at"] = now()
    client.put_status(job_id, status)

    # Download all chunks into a job-scoped subdir of workdir.
    job_dir = workdir / job_id
    input_dir = job_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    try:
        chunks = client.download_chunks(job_id, input_dir)
    except Exception as exc:  # noqa: BLE001 — translate any download error into failed
        return _fail_job(client, job_id, status, f"download error: {exc}", now)

    if not chunks:
        return _fail_job(
            client,
            job_id,
            status,
            "no input chunks downloaded from R2 (missing or empty input/ prefix)",
            now,
        )

    def report_progress(pct: int) -> None:
        current = client.get_status(job_id)
        if current is None:
            return
        current["progress_pct"] = pct
        current["updated_at"] = now()
        client.put_status(job_id, current)

    try:
        html = pipeline(chunks, report_progress)
        report_key = client.upload_report(job_id, html)
    except Exception as exc:  # noqa: BLE001 — pipeline crash → record + survive
        return _fail_job(client, job_id, status, f"{type(exc).__name__}: {exc}", now)

    final = client.get_status(job_id) or status
    final["status"] = "completed"
    final["progress_pct"] = 100
    final["report_key"] = report_key
    final["error"] = None
    final["updated_at"] = now()
    client.put_status(job_id, final)
    return "completed"


def worker_loop(
    client: R2ClientLike,
    worker_id: str,
    pipeline: PipelineFn,
    workdir: Path,
    poll_interval_s: float = 10.0,
    sleep: Callable[[float], None] = time.sleep,
    max_iterations: int | None = None,
    now: NowFn = _utc_now_iso,
) -> None:
    """Long-running poll loop: discover pending jobs, process them, sleep.

    SPEC §8.1 — the worker is a daemon. Each iteration lists pending jobs
    in R2, processes every one in turn via :func:`process_job`, then sleeps
    for ``poll_interval_s`` before polling again. Failures inside
    :func:`process_job` are already absorbed into ``status.json``, so the
    loop never raises out for routine errors and stays alive.

    ``max_iterations`` is a test-only escape hatch — production callers leave
    it ``None`` for an unbounded loop. ``sleep`` is injectable so tests can
    observe poll intervals without actually waiting.
    """
    iterations = 0
    while True:
        try:
            job_ids = client.list_pending_job_ids()
        except Exception:
            logger.exception("worker_loop: list_pending_job_ids failed")
            job_ids = []

        for job_id in job_ids:
            try:
                process_job(
                    client=client,
                    job_id=job_id,
                    worker_id=worker_id,
                    pipeline=pipeline,
                    workdir=workdir,
                    now=now,
                )
            except Exception:
                # Defense in depth — process_job already records failures into
                # status.json, but if it ever escapes we still want the loop alive.
                logger.exception("worker_loop: unexpected error processing %s", job_id)

        sleep(poll_interval_s)
        iterations += 1
        if max_iterations is not None and iterations >= max_iterations:
            return


_REQUIRED_ENV = ("R2_ENDPOINT", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY")


def main(argv: list[str] | None = None) -> int:
    """``python -m gpu_service.worker`` entry point — env-var driven daemon.

    Reads R2 + worker config from the environment (SPEC §10.1), constructs a
    real :class:`R2Client` and a pipeline closure that defers to
    :func:`pipeline.analyze.run_full_video_to_html` (lazy-imported so unit
    tests don't pull onnxruntime), then hands off to :func:`worker_loop`.

    Returns ``0`` on clean exit, ``2`` if any required env var is missing.
    """
    import os
    import sys

    from gpu_service.r2_client import R2Client

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    missing = [name for name in _REQUIRED_ENV if not os.environ.get(name)]
    if missing:
        print(
            f"error: missing required environment variable(s): {', '.join(missing)}",
            file=sys.stderr,
        )
        return 2

    bucket = os.environ.get("R2_BUCKET", "surveillance-data")
    worker_id = os.environ.get("WORKER_ID", "gpu-worker-1")
    workdir = Path(os.environ.get("WORKDIR", "/tmp/cctv-jobs"))
    poll_interval_s = float(os.environ.get("POLL_INTERVAL_S", "10"))

    client = R2Client(
        endpoint=os.environ["R2_ENDPOINT"],
        access_key=os.environ["R2_ACCESS_KEY_ID"],
        secret_key=os.environ["R2_SECRET_ACCESS_KEY"],
        bucket=bucket,
    )

    def pipeline(chunks: list[Path], progress: ProgressCallback) -> bytes:
        # Lazy import: keeps the unit-test path free of onnxruntime / GPU deps.
        from pipeline.analyze import run_full_video_to_html

        return run_full_video_to_html(chunks, progress=progress)

    workdir.mkdir(parents=True, exist_ok=True)
    logger.info("starting worker_loop worker_id=%s bucket=%s", worker_id, bucket)
    worker_loop(
        client=client,
        worker_id=worker_id,
        pipeline=pipeline,
        workdir=workdir,
        poll_interval_s=poll_interval_s,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
