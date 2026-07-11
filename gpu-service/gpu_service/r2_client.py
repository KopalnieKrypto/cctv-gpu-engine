"""Cloudflare R2 client (boto3 S3-compatible) for the gpu-service worker.

Thin translation layer over ``boto3.client("s3")`` that exposes exactly the
duck-typed surface :class:`gpu_service.worker.R2ClientLike` consumes:

* :meth:`R2Client.list_pending_job_ids`
* :meth:`R2Client.get_status`
* :meth:`R2Client.put_status`
* :meth:`R2Client.download_chunks`
* :meth:`R2Client.upload_report`

The R2 key conventions follow SPEC §6.2:

::

    surveillance-jobs/{job_id}/status.json
    surveillance-jobs/{job_id}/input/chunk_001.mp4
    surveillance-jobs/{job_id}/output/result.json  (issue #72 — was report.html)

Network methods retry transient failures 3× with exponential backoff per
SPEC §8.2 (issue #61); after the final failure the original error propagates
to the worker, which records it as ``status: failed`` and keeps polling.
Status-list walks share an ETag-keyed cache so an unchanged bucket costs zero
``get_object`` calls per poll.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import boto3

JOBS_PREFIX = "surveillance-jobs"

T = TypeVar("T")

# SPEC §8.2: "retry 3× with exponential backoff, then fail". Four attempts
# total = one initial try + three retries; backoffs precede each retry.
RETRY_ATTEMPTS = 4
RETRY_BACKOFFS = (1.0, 2.0, 4.0)


def _with_retry(op: Callable[[], T]) -> T:
    """Run ``op``, retrying transient failures per SPEC §8.2.

    Up to :data:`RETRY_ATTEMPTS` tries with exponential backoff between them.
    The *original* exception is re-raised after the final failure so callers
    that already translate boto3 errors (worker → ``status: failed``, web UI
    → 404) keep working unchanged. ``time.sleep`` is looked up on the module
    at call time so tests can patch it and assert the backoff schedule.
    """
    last_exc: BaseException | None = None
    for i in range(RETRY_ATTEMPTS):
        try:
            return op()
        except Exception as exc:  # noqa: BLE001 — network layer, retry broadly
            last_exc = exc
            if i + 1 >= RETRY_ATTEMPTS:
                break
            time.sleep(RETRY_BACKOFFS[min(i, len(RETRY_BACKOFFS) - 1)])
    assert last_exc is not None  # loop ran at least once
    raise last_exc


class R2Client:
    """boto3 S3-compatible client wired for Cloudflare R2."""

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        region: str = "auto",
    ) -> None:
        self._bucket = bucket
        # ETag-keyed status.json cache (issue #61): {key: (etag, status)}.
        # ListObjectsV2 hands us each object's ETag for free, so an unchanged
        # status.json is served from here instead of costing a GET on every
        # 10 s poll — an idle 1000-job bucket otherwise burns ~8.6M reads/day.
        self._status_cache: dict[str, tuple[str, dict[str, Any]]] = {}
        # R2 ignores region but boto3 needs *something*; "auto" is the
        # convention in Cloudflare's own docs and SDK examples.
        self._s3 = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
        )

    # ----- key helpers -----

    @staticmethod
    def _status_key(job_id: str) -> str:
        return f"{JOBS_PREFIX}/{job_id}/status.json"

    @staticmethod
    def _input_prefix(job_id: str) -> str:
        return f"{JOBS_PREFIX}/{job_id}/input/"

    @staticmethod
    def _report_key(job_id: str) -> str:
        # Canonical structured artifact (issue #72) — the platform renders
        # this JSON natively; the engine no longer emits HTML here.
        return f"{JOBS_PREFIX}/{job_id}/output/result.json"

    @staticmethod
    def _detections_key(job_id: str) -> str:
        # Per-frame raw-detections archive (issue #35) — sits beside
        # result.json so post-hoc "why did the system score X?" questions
        # never force an expensive pipeline re-run.
        return f"{JOBS_PREFIX}/{job_id}/output/detections.jsonl"

    # ----- worker-facing API -----

    def list_pending_job_ids(self) -> list[str]:
        """List every status.json under the jobs prefix and keep `pending` ones.

        Uses the ``list_objects_v2`` paginator so it works correctly above the
        1000-key default page size — production buckets will accumulate
        completed jobs and we don't want them silently truncated.
        """
        paginator = self._s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=self._bucket, Prefix=f"{JOBS_PREFIX}/")
        pending: list[str] = []
        for page in pages:
            for obj in page.get("Contents", []) or []:
                key = obj["Key"]
                if not key.endswith("/status.json"):
                    continue
                status = self._read_status_cached(key, obj.get("ETag"))
                if status is None:
                    continue
                if status.get("status") != "pending":
                    continue
                # surveillance-jobs/{job_id}/status.json
                parts = key.split("/")
                if len(parts) >= 3:
                    pending.append(parts[1])
        return pending

    def list_all_job_statuses(self) -> list[tuple[str, dict[str, Any]]]:
        """Return ``(job_id, status_dict)`` for every status.json in the bucket.

        Used by the dashboard (issue #6) which needs the full job history,
        not just `pending` jobs. Same paginator walk as
        :meth:`list_pending_job_ids` but without the status filter — keys
        whose status.json is missing or unparseable are silently skipped so
        a single corrupt entry never blanks out the dashboard.
        """
        paginator = self._s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=self._bucket, Prefix=f"{JOBS_PREFIX}/")
        out: list[tuple[str, dict[str, Any]]] = []
        for page in pages:
            for obj in page.get("Contents", []) or []:
                key = obj["Key"]
                if not key.endswith("/status.json"):
                    continue
                status = self._read_status_cached(key, obj.get("ETag"))
                if status is None:
                    continue
                parts = key.split("/")
                if len(parts) >= 3:
                    out.append((parts[1], status))
        return out

    def get_status(self, job_id: str) -> dict[str, Any] | None:
        return self._read_status_key(self._status_key(job_id))

    def put_status(self, job_id: str, status: dict[str, Any]) -> None:
        body = json.dumps(status).encode("utf-8")
        self._s3.put_object(
            Bucket=self._bucket,
            Key=self._status_key(job_id),
            Body=body,
            ContentType="application/json",
        )

    def download_chunks(self, job_id: str, dest: Path) -> list[Path]:
        dest.mkdir(parents=True, exist_ok=True)
        prefix = self._input_prefix(job_id)
        paginator = self._s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=self._bucket, Prefix=prefix)
        downloaded: list[Path] = []
        for page in pages:
            for obj in page.get("Contents", []) or []:
                key = obj["Key"]
                name = key[len(prefix) :]
                if not name:  # the prefix itself, skip
                    continue
                local = dest / name
                local.parent.mkdir(parents=True, exist_ok=True)
                _with_retry(
                    lambda key=key, local=local: self._s3.download_file(
                        Bucket=self._bucket, Key=key, Filename=str(local)
                    )
                )
                downloaded.append(local)
        return sorted(downloaded)

    def upload_input_chunk(self, job_id: str, fileobj: Any) -> str:
        """Stream an input MP4 chunk to R2 via boto3 multipart upload.

        Used by the client-agent web UI (issue #7). MUST go through
        ``upload_fileobj`` (not ``put_object``) so the request body is
        drained chunk-by-chunk — a 2 GB MP4 would otherwise be loaded into
        the agent process's RAM and OOM the container.

        SPEC §6.2 key convention:
        ``surveillance-jobs/{job_id}/input/chunk_001.mp4``. Single-chunk for
        now; multi-chunk recordings (issue #8 / RTSP segmented capture) can
        pass a different chunk name later.
        """
        key = f"{self._input_prefix(job_id)}chunk_001.mp4"
        _with_retry(lambda: self._s3.upload_fileobj(fileobj, self._bucket, key))
        return key

    def get_report(self, job_id: str) -> bytes:
        """Read the canonical result.json the worker wrote for ``job_id`` from R2.

        Returns the raw bytes for the caller to consume. Artifacts are small
        (structured data + base64 JPEG frames) so the whole-buffer read is
        fine; multipart streaming is reserved for the *upload* path where
        input files are 100x larger. Errors propagate so the caller can
        translate to 404.
        """
        obj = _with_retry(
            lambda: self._s3.get_object(Bucket=self._bucket, Key=self._report_key(job_id))
        )
        return obj["Body"].read()

    def upload_report(self, job_id: str, body: bytes) -> str:
        """Upload the canonical result.json artifact (issue #72) and return its key."""
        key = self._report_key(job_id)
        _with_retry(
            lambda: self._s3.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=body,
                ContentType="application/json",
            )
        )
        return key

    def upload_detections(self, job_id: str, body: bytes) -> str:
        """Upload the per-frame detections.jsonl archive (issue #35) and return its key."""
        key = self._detections_key(job_id)
        _with_retry(
            lambda: self._s3.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=body,
                ContentType="application/x-ndjson",
            )
        )
        return key

    # ----- internals -----

    def _read_status_cached(self, key: str, etag: str | None) -> dict[str, Any] | None:
        """Return the parsed status.json for ``key``, using the ETag cache.

        On an ETag match we skip the GET entirely. Only successful reads are
        cached — a transient GET failure (returns None) is *not* cached, so
        the next poll retries it instead of pinning a phantom None until the
        object happens to change.
        """
        if etag is not None:
            cached = self._status_cache.get(key)
            if cached is not None and cached[0] == etag:
                return cached[1]
        status = self._read_status_key(key)
        if status is not None and etag is not None:
            self._status_cache[key] = (etag, status)
        return status

    def _read_status_key(self, key: str) -> dict[str, Any] | None:
        """Read a single status.json key, returning None on any read error.

        We swallow boto3 errors here (NoSuchKey, transient network) because
        the worker treats "no status" identically to "stale status" — it just
        moves on to the next job. Logging is the caller's responsibility.
        """
        try:
            obj = self._s3.get_object(Bucket=self._bucket, Key=key)
        except Exception:  # noqa: BLE001 — see docstring
            return None
        try:
            body = obj["Body"].read()
            return json.loads(body)
        except (json.JSONDecodeError, ValueError, KeyError):
            return None
