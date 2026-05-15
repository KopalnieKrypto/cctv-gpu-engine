"""Tests for the tenant-prefix defense-in-depth check (issue #25 AC #9).

The container does not trust the platform to enforce tenant isolation in
the presigned URL it hands us — it independently asserts the URL path
starts with ``tenants/{tenant_id}/results/{task_id}/`` and rejects anything
else as a cross-tenant write attempt.
"""

from __future__ import annotations

import pytest

from gpu_service.tenant_url import TenantPrefixError, extract_tenant_id


class TestExtractTenantId:
    def test_returns_tenant_id_for_well_formed_url(self) -> None:
        url = (
            "https://r2.example.com/tenants/acme-corp/results/"
            "11111111-2222-3333-4444-555555555555/report.html?sig=abc"
        )
        tenant = extract_tenant_id(url, task_id="11111111-2222-3333-4444-555555555555")
        assert tenant == "acme-corp"

    def test_accepts_nested_bucket_prefix(self) -> None:
        # S3-style path-rooted bucket: r2 endpoints often start with
        # /<bucket>/... before the tenants/ segment. The check only cares
        # that tenants/{tid}/results/{taskId}/ appears in the path, not
        # that it starts at position 0.
        url = (
            "https://r2.example.com/surveillance-data/tenants/acme/results/"
            "task-1/output.html?sig=abc"
        )
        assert extract_tenant_id(url, task_id="task-1") == "acme"

    def test_rejects_missing_tenants_segment(self) -> None:
        url = "https://r2.example.com/results/task-1/output.html"
        with pytest.raises(TenantPrefixError):
            extract_tenant_id(url, task_id="task-1")

    def test_rejects_task_id_mismatch(self) -> None:
        # cross-tenant attack: URL has tenants/{tid}/results/{otherTaskId}/
        # but the body claims a different task_id. Block — we'd be writing
        # to someone else's task slot.
        url = "https://r2.example.com/tenants/acme/results/other-task/output.html"
        with pytest.raises(TenantPrefixError):
            extract_tenant_id(url, task_id="my-task")

    def test_rejects_results_segment_missing(self) -> None:
        url = "https://r2.example.com/tenants/acme/uploads/task-1/output.html"
        with pytest.raises(TenantPrefixError):
            extract_tenant_id(url, task_id="task-1")

    def test_rejects_path_traversal_in_tenant(self) -> None:
        # ``..`` in the tenant segment would let an attacker climb out of
        # their prefix even on a permissive R2 policy. Treat as malformed.
        url = "https://r2.example.com/tenants/..%2Fother/results/t/output.html"
        with pytest.raises(TenantPrefixError):
            extract_tenant_id(url, task_id="t")
