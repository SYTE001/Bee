"""Task runner abstraction: Celery in production, ThreadPool in local dev.

Both backends call the same pipeline entrypoint, so behavior is identical.
`TASK_BACKEND=thread` removes the Redis dependency for local development.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from app.core.config import settings
from app.core.logging import get_logger

log = get_logger("tasks")

_executor: ThreadPoolExecutor | None = None


def submit_task(func: Callable[..., Any], *args: Any, **kwargs: Any) -> str:
    """Run `func` asynchronously and return a task id."""
    if settings.TASK_BACKEND == "celery":
        try:
            path = f"{func.__module__}.{func.__name__}"
            if "<locals>" in path or not getattr(func, "__name__", ""):
                raise ValueError(f"{func!r} is not a module-level function")
            from app.workers.celery_app import run_task

            async_result = run_task.delay(path, *args, **kwargs)
            return str(async_result.id)
        except Exception:  # noqa: BLE001 - fall back to threads if broker is down
            log.warning("celery unavailable, falling back to thread pool")
    return _submit_thread(func, *args, **kwargs)


def _submit_thread(func: Callable[..., Any], *args: Any, **kwargs: Any) -> str:
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(
            max_workers=max(settings.WORKER_CONCURRENCY, 1), thread_name_prefix="cf-worker"
        )
    import uuid

    task_id = str(uuid.uuid4())
    _executor.submit(_thread_wrapper, task_id, func, args, kwargs)
    return task_id


def _thread_wrapper(task_id: str, func: Callable[..., Any], args: tuple, kwargs: dict) -> None:
    from app.core.logging import setup_logging

    setup_logging()
    log.info("task %s started", task_id)
    try:
        func(*args, **kwargs)
    except Exception:  # noqa: BLE001 - background task must never leak
        log.exception("task %s failed", task_id)
    finally:
        log.info("task %s finished", task_id)


def shutdown() -> None:
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=False, cancel_futures=True)
        _executor = None
