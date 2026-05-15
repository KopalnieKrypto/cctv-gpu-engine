"""Tests for the presigned-URL retry policy (issue #25 AC #2, #3, #10).

The spec is rigid: 3 attempts, exponential backoff sleeps of 1s / 2s / 4s
between attempts, raise after the 3rd failure. We test by injecting both
the operation and the sleeper so the test runs in <10 ms.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from gpu_service.http_retry import RetryExhausted, with_retry


class TestWithRetry:
    def test_returns_value_on_first_success(self) -> None:
        op = MagicMock(return_value="ok")
        sleeper = MagicMock()

        result = with_retry(op, sleep=sleeper)

        assert result == "ok"
        assert op.call_count == 1
        sleeper.assert_not_called()

    def test_retries_then_succeeds_on_third_call(self) -> None:
        op = MagicMock(side_effect=[ConnectionError("boom"), ConnectionError("boom"), "ok"])
        sleeper = MagicMock()

        result = with_retry(op, sleep=sleeper)

        assert result == "ok"
        assert op.call_count == 3
        # AC: exp backoff 1s / 2s — sleeps happen *between* attempts, so
        # 2 sleeps total when the 3rd attempt succeeds.
        assert sleeper.call_args_list == [((1,),), ((2,),)]

    def test_raises_retry_exhausted_after_three_failures(self) -> None:
        original = ConnectionError("network down")
        op = MagicMock(side_effect=[original, original, original])
        sleeper = MagicMock()

        with pytest.raises(RetryExhausted) as exc_info:
            with_retry(op, sleep=sleeper)

        assert op.call_count == 3
        # All 3 attempts failed → 1s and 2s sleeps between them; the spec
        # mentions 4s as the "after-final-failure" wait, but that does not
        # apply when we are giving up — no point sleeping before raising.
        assert sleeper.call_args_list == [((1,),), ((2,),)]
        assert exc_info.value.__cause__ is original

    def test_only_retries_on_expected_exceptions(self) -> None:
        # A bug (e.g. ValueError from bad URL parsing) should not be
        # retried — bubble up immediately so the caller sees the real
        # cause instead of a generic RetryExhausted three sleeps later.
        op = MagicMock(side_effect=ValueError("bad url"))
        sleeper = MagicMock()

        with pytest.raises(ValueError):
            with_retry(op, sleep=sleeper, retry_on=(ConnectionError,))

        assert op.call_count == 1
        sleeper.assert_not_called()
