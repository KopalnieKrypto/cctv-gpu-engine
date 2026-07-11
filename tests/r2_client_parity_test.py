"""Parity guard for the two intentionally-duplicated R2Client copies.

``gpu_service.r2_client`` and ``client_agent.r2_client`` are kept as separate
copies on purpose (issue #40 — client-agent must ship without gpu_service).
The retry policy added in issue #61 (SPEC §8.2) must stay identical between
them, so a fix applied to one copy but not the other is caught here rather
than in production.
"""

from __future__ import annotations

from client_agent import r2_client as ca
from gpu_service import r2_client as gs


def test_retry_policy_matches_spec_8_2() -> None:
    # "retry 3× with exponential backoff, then fail" = 4 attempts total.
    assert gs.RETRY_ATTEMPTS == 4
    assert gs.RETRY_BACKOFFS == (1.0, 2.0, 4.0)


def test_retry_policy_is_identical_between_copies() -> None:
    assert gs.RETRY_ATTEMPTS == ca.RETRY_ATTEMPTS
    assert gs.RETRY_BACKOFFS == ca.RETRY_BACKOFFS


def test_both_copies_expose_the_retry_and_cache_seams() -> None:
    # The wrapping helper and the ETag-cache read path must exist in both so
    # a future edit can't silently regress one copy's retry/caching behaviour.
    for mod in (gs, ca):
        assert callable(mod._with_retry)
        assert hasattr(mod.R2Client, "_read_status_cached")
