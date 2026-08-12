"""Retry scheduling.

Kept separate from the queue so the retry policy is testable in isolation and
so the archive downloader can reuse the exact same curve.
"""

from __future__ import annotations

import random
import time
from typing import Callable, Optional, Sequence, TypeVar

from ..constants import RETRY_BASE_DELAY_S, RETRY_JITTER_RATIO, RETRY_MAX_DELAY_S
from ..errors import RetryableError

T = TypeVar("T")


def backoff_delay(
    attempt: int,
    *,
    base_s: float = RETRY_BASE_DELAY_S,
    max_s: float = RETRY_MAX_DELAY_S,
    jitter_ratio: float = RETRY_JITTER_RATIO,
    rng: Optional[random.Random] = None,
) -> float:
    """Exponential backoff with full-width jitter.

    Jitter matters here: when a NAS goes offline, every worker fails at the
    same instant, and without jitter they would all retry in lockstep forever.

    Args:
        attempt: 1-based attempt number that just failed.
    """
    attempt = max(1, attempt)
    raw = float(min(max_s, base_s * (2 ** (attempt - 1))))
    if jitter_ratio <= 0:
        return raw
    generator = rng or random
    spread = raw * jitter_ratio
    return max(0.0, raw + float(generator.uniform(-spread, spread)))


def retry_call(
    func: Callable[[], T],
    *,
    attempts: int = 3,
    exceptions: Sequence[type[BaseException]] = (RetryableError,),
    base_s: float = RETRY_BASE_DELAY_S,
    max_s: float = RETRY_MAX_DELAY_S,
    sleep: Callable[[float], None] = time.sleep,
    on_retry: Optional[Callable[[int, BaseException, float], None]] = None,
) -> T:
    """Call ``func``, retrying transient failures with backoff.

    Used for network and filesystem work inside a single job; job-level retries
    are handled by the queue, which survives process restarts.
    """
    last: BaseException
    for attempt in range(1, attempts + 1):
        try:
            return func()
        except tuple(exceptions) as exc:
            last = exc
            if attempt >= attempts:
                break
            delay = backoff_delay(attempt, base_s=base_s, max_s=max_s)
            explicit = getattr(exc, "retry_after_s", None)
            if explicit:
                delay = float(explicit)
            if on_retry is not None:
                on_retry(attempt, exc, delay)
            sleep(delay)
    raise last
