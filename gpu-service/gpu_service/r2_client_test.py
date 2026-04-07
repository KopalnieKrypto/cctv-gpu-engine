"""Boundary tests for the boto3-backed R2 client.

These tests mock the boto3 S3 client at the **system boundary** (the network
edge to Cloudflare R2). The R2Client itself is exercised through its real
public methods — we never mock R2Client's internal collaborators, only the
boto3 surface it talks to.

Test strategy: each test wires a fresh ``MagicMock`` as the ``boto3.client``
return value and asserts both the calls made to it (key naming conventions
from SPEC §6.2) and the values returned by R2Client (decoded JSON, downloaded
file paths, uploaded report key).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from gpu_service.r2_client import R2Client


def _make_client(s3: MagicMock) -> R2Client:
    """Build an R2Client with the provided mock S3 client injected."""
    with patch("gpu_service.r2_client.boto3.client", return_value=s3):
        return R2Client(
            endpoint="https://example.r2.cloudflarestorage.com",
            access_key="AK",
            secret_key="SK",
            bucket="surveillance-data",
        )


def test_constructor_creates_boto3_s3_client_with_r2_endpoint() -> None:
    """R2Client must hand boto3 the S3-compatible R2 endpoint + credentials.

    SPEC §10.1 — R2_ENDPOINT, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY are
    passed through to a boto3 S3 client. Region must be a valid AWS region
    string (R2 ignores it but boto3 demands one); ``auto`` is the convention.
    """
    with patch("gpu_service.r2_client.boto3.client") as boto3_client:
        R2Client(
            endpoint="https://acct.r2.cloudflarestorage.com",
            access_key="AK",
            secret_key="SK",
            bucket="surveillance-data",
        )

    boto3_client.assert_called_once()
    args, kwargs = boto3_client.call_args
    assert args == ("s3",) or kwargs.get("service_name") == "s3"
    assert kwargs["endpoint_url"] == "https://acct.r2.cloudflarestorage.com"
    assert kwargs["aws_access_key_id"] == "AK"
    assert kwargs["aws_secret_access_key"] == "SK"


def test_list_pending_job_ids_filters_status_json_with_status_pending() -> None:
    """ListObjectsV2 returns every key under surveillance-jobs/, then we
    fetch each status.json and keep only those with ``status: pending``."""
    s3 = MagicMock()
    s3.get_paginator.return_value.paginate.return_value = [
        {
            "Contents": [
                {"Key": "surveillance-jobs/job-a/status.json"},
                {"Key": "surveillance-jobs/job-a/input/chunk_001.mp4"},
                {"Key": "surveillance-jobs/job-b/status.json"},
                {"Key": "surveillance-jobs/job-c/status.json"},
            ]
        }
    ]

    def get_object(Bucket: str, Key: str) -> dict:
        statuses = {
            "surveillance-jobs/job-a/status.json": {"status": "pending"},
            "surveillance-jobs/job-b/status.json": {"status": "completed"},
            "surveillance-jobs/job-c/status.json": {"status": "pending"},
        }
        body = json.dumps(statuses[Key]).encode()
        return {"Body": MagicMock(read=MagicMock(return_value=body))}

    s3.get_object.side_effect = get_object

    client = _make_client(s3)
    pending = client.list_pending_job_ids()

    assert pending == ["job-a", "job-c"]
    s3.get_paginator.assert_called_once_with("list_objects_v2")
    s3.get_paginator.return_value.paginate.assert_called_once_with(
        Bucket="surveillance-data", Prefix="surveillance-jobs/"
    )


def test_get_status_returns_decoded_json_or_none_when_missing() -> None:
    """get_status round-trips JSON; missing key (NoSuchKey) returns None."""
    s3 = MagicMock()
    body_bytes = json.dumps({"job_id": "j1", "status": "pending"}).encode()
    s3.get_object.return_value = {"Body": MagicMock(read=MagicMock(return_value=body_bytes))}

    client = _make_client(s3)
    result = client.get_status("j1")
    assert result == {"job_id": "j1", "status": "pending"}
    s3.get_object.assert_called_with(
        Bucket="surveillance-data", Key="surveillance-jobs/j1/status.json"
    )

    # Now simulate a NoSuchKey-style failure: any ClientError-ish exception
    # returns None instead of raising.
    s3.get_object.side_effect = Exception("NoSuchKey")
    assert client.get_status("missing") is None


def test_put_status_writes_json_bytes_with_content_type() -> None:
    s3 = MagicMock()
    client = _make_client(s3)

    client.put_status("j1", {"job_id": "j1", "status": "processing"})

    s3.put_object.assert_called_once()
    kwargs = s3.put_object.call_args.kwargs
    assert kwargs["Bucket"] == "surveillance-data"
    assert kwargs["Key"] == "surveillance-jobs/j1/status.json"
    assert kwargs["ContentType"] == "application/json"
    assert json.loads(kwargs["Body"]) == {"job_id": "j1", "status": "processing"}


def test_download_chunks_writes_input_files_to_dest_and_returns_paths(
    tmp_path: Path,
) -> None:
    s3 = MagicMock()
    s3.get_paginator.return_value.paginate.return_value = [
        {
            "Contents": [
                {"Key": "surveillance-jobs/j1/input/chunk_001.mp4"},
                {"Key": "surveillance-jobs/j1/input/chunk_002.mp4"},
            ]
        }
    ]

    written: dict[str, bytes] = {}

    def fake_download(Bucket: str, Key: str, Filename: str) -> None:
        written[Key] = b"bytes-of-" + Key.encode()
        Path(Filename).write_bytes(written[Key])

    s3.download_file.side_effect = fake_download

    client = _make_client(s3)
    paths = client.download_chunks("j1", tmp_path)

    assert sorted(p.name for p in paths) == ["chunk_001.mp4", "chunk_002.mp4"]
    for p in paths:
        assert p.exists()
        assert p.read_bytes().startswith(b"bytes-of-")

    s3.get_paginator.return_value.paginate.assert_called_once_with(
        Bucket="surveillance-data", Prefix="surveillance-jobs/j1/input/"
    )


def test_upload_report_uses_output_report_html_key_and_returns_it() -> None:
    s3 = MagicMock()
    client = _make_client(s3)

    key = client.upload_report("j1", b"<html>r</html>")

    assert key == "surveillance-jobs/j1/output/report.html"
    s3.put_object.assert_called_once()
    kwargs = s3.put_object.call_args.kwargs
    assert kwargs["Bucket"] == "surveillance-data"
    assert kwargs["Key"] == "surveillance-jobs/j1/output/report.html"
    assert kwargs["ContentType"] == "text/html; charset=utf-8"
    assert kwargs["Body"] == b"<html>r</html>"
