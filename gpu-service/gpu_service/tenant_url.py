"""Tenant-prefix defense-in-depth for presigned URLs (issue #25 AC #9).

The gpu-exchange platform issues a presigned PUT URL scoped to
``tenants/{tenantId}/results/{taskId}/`` before dispatching a job. The
platform *should* enforce that scope when it signs the URL, but a single
misconfigured policy could let the agent slip in a cross-tenant URL — and
we'd silently overwrite another tenant's result. So the container does its
own check before calling PUT: parse the URL, walk the path segments, and
demand the exact ``tenants/<id>/results/<task_id>/...`` shape with the
expected task_id. Any deviation raises :class:`TenantPrefixError` and the
caller turns it into a 400.
"""

from __future__ import annotations

from urllib.parse import unquote, urlsplit


class TenantPrefixError(ValueError):
    """Raised when a presigned URL does not match the expected tenant prefix."""


def extract_tenant_id(url: str, task_id: str) -> str:
    """Return the tenant_id parsed from ``url`` or raise :class:`TenantPrefixError`.

    Accepts URLs of the form::

        https://<host>/[<bucket>/]tenants/<tenant_id>/results/<task_id>/<key...>

    Both segments after ``tenants/`` and ``results/`` must be non-empty,
    URL-decoded, and free of path-traversal (``..``). The ``task_id`` in
    the URL must match the ``task_id`` argument exactly — otherwise the
    URL points at someone else's task slot.
    """
    path = urlsplit(url).path
    # Drop leading "/" before splitting so a leading "/" doesn't produce an
    # empty leading segment that breaks the "tenants/" lookup.
    segments = [unquote(s) for s in path.lstrip("/").split("/") if s]

    try:
        idx = segments.index("tenants")
    except ValueError as e:
        raise TenantPrefixError(f"URL path lacks 'tenants/' segment: {path!r}") from e

    # Need: tenants / <tid> / results / <task_id> / <at-least-one-key-segment>
    if len(segments) < idx + 5:
        raise TenantPrefixError(f"URL path too short after 'tenants/': {path!r}")

    tenant_id = segments[idx + 1]
    results_segment = segments[idx + 2]
    url_task_id = segments[idx + 3]

    if not tenant_id or ".." in tenant_id or "/" in tenant_id:
        raise TenantPrefixError(f"Invalid tenant_id segment: {tenant_id!r}")
    if results_segment != "results":
        raise TenantPrefixError(f"Expected 'results' after tenant segment, got {results_segment!r}")
    if url_task_id != task_id:
        raise TenantPrefixError(
            f"URL task_id {url_task_id!r} does not match payload task_id {task_id!r}"
        )

    return tenant_id
