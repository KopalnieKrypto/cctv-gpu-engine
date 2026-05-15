"""Retry policy for presigned-URL R2 operations (issue #25 AC #2/#3/#10).

The spec is 3 attempts with exponential backoff between them. The "fail 2x
then success" AC fixes the attempt count: three attempts, two sleeps of
1s and 2s between them. Wait, this conflicts with the natural reading of
"1s/2s/4s" — but the explicit AC trumps the prose, so we treat the third
number (4s) as the wait *that would precede a hypothetical fourth attempt*
and never actually sleep it. After the 3rd failure we raise.

The decorator is intentionally tiny and synchronous — it is called from
the per-task worker thread, where a 3-second worst-case wall-clock pause
on retries is fine. Real backoff jitter is not added (R2 is internal and
not adversarial — no thundering herd to worry about with a single agent).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

# AC #10 specifies "fail 2x then success" — three attempts total.
DEFAULT_ATTEMPTS = 3
# AC #2/#3: backoff 1s, 2s (4s would precede a 4th attempt we never make).
DEFAULT_BACKOFFS = (1, 2)


class RetryExhausted(Exception):
    """Raised when all retry attempts failed. ``__cause__`` is the last error."""


def with_retry(
    op: Callable[[], T],
    *,
    attempts: int = DEFAULT_ATTEMPTS,
    backoffs: tuple[int, ...] = DEFAULT_BACKOFFS,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Run ``op`` with up to ``attempts`` tries, sleeping between retries.

    ``retry_on`` whitelists the exception types worth retrying — anything
    else propagates immediately so we don't spend 3 seconds masking a
    programming bug behind a generic ``RetryExhausted``.
    """
    last_exc: BaseException | None = None
    for i in range(attempts):
        try:
            return op()
        except retry_on as exc:
            last_exc = exc
            if i + 1 >= attempts:
                break
            sleep(backoffs[i] if i < len(backoffs) else backoffs[-1])

    raise RetryExhausted("operation failed after retries") from last_exc
