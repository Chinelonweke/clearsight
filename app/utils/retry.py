"""
app/utils/retry.py
─────────────────────────────────────────────────────────
Generic async retry decorator with exponential backoff.

Usage:
    @async_retry(max_attempts=3, base_delay=1.0, exceptions=(httpx.TimeoutException,))
    async def call_external_service():
        ...

    # Or imperatively:
    result = await retry_async(my_coroutine_function, arg1, arg2, max_attempts=3)
"""

import asyncio
import functools
from typing import Any, Callable, Type, TypeVar

from app.core.logger import get_logger

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def async_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple[Type[Exception], ...] = (Exception,),
    reraise: bool = True,
) -> Callable[[F], F]:
    """
    Decorator: retry an async function on specified exceptions with exponential backoff.

    Args:
        max_attempts:   Maximum number of total attempts (including the first).
        base_delay:     Initial delay in seconds before the first retry.
        backoff_factor: Multiplier applied to the delay on each retry.
        exceptions:     Exception types that trigger a retry (others propagate immediately).
        reraise:        If True, re-raise the last exception after all retries are exhausted.
                        If False, return None instead.

    Example delays for base=1.0, factor=2.0:
        Attempt 1: fails → wait 1s
        Attempt 2: fails → wait 2s
        Attempt 3: fails → wait 4s → raise
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Exception | None = None
            delay = base_delay

            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt == max_attempts:
                        logger.error(
                            f"[retry] {func.__name__} failed after "
                            f"{max_attempts} attempts: {exc}"
                        )
                        break
                    logger.warning(
                        f"[retry] {func.__name__} attempt {attempt}/{max_attempts} "
                        f"failed: {exc}. Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                    delay *= backoff_factor

            if reraise and last_exc is not None:
                raise last_exc
            return None

        return wrapper  # type: ignore
    return decorator


async def retry_async(
    func: Callable,
    *args: Any,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    **kwargs: Any,
) -> Any:
    """
    Imperative retry helper for use without the decorator.

    Usage:
        result = await retry_async(my_func, arg1, arg2, max_attempts=5, kwarg=val)
    """
    last_exc: Exception | None = None
    delay = base_delay

    for attempt in range(1, max_attempts + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt == max_attempts:
                break
            logger.warning(
                f"retry_async: {func.__name__} attempt {attempt}/{max_attempts} "
                f"failed: {exc}. Retrying in {delay:.1f}s..."
            )
            await asyncio.sleep(delay)
            delay *= 2.0

    raise last_exc or RuntimeError("retry_async: all attempts failed")