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
    surveillance-jobs/{job_id}/output/report.html

There is no retry logic in this layer — issue #5 deferred upload retry to a
follow-up (see ``CLAUDE.md`` TODO). Network errors propagate to the worker,
which records them as ``status: failed`` and keeps polling.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import boto3

JOBS_PREFIX = "surveillance-jobs"


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
        return f"{JOBS_PREFIX}/{job_id}/output/report.html"

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
                status = self._read_status_key(key)
                if status is None:
                    continue
                if status.get("status") != "pending":
                    continue
                # surveillance-jobs/{job_id}/status.json
                parts = key.split("/")
                if len(parts) >= 3:
                    pending.append(parts[1])
        return pending

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
                self._s3.download_file(Bucket=self._bucket, Key=key, Filename=str(local))
                downloaded.append(local)
        return sorted(downloaded)

    def upload_report(self, job_id: str, html: bytes) -> str:
        key = self._report_key(job_id)
        self._s3.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=html,
            ContentType="text/html; charset=utf-8",
        )
        return key

    # ----- internals -----

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
