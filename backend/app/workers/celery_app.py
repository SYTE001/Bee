"""Celery application + pipeline task.

Only used when `TASK_BACKEND=celery` (docker-compose / multi-worker). The
thread backend covers local development without Redis.
"""

from __future__ import annotations

import importlib
from typing import Any

from app.core.config import settings
from app.core.logging import get_logger, setup_logging

log = get_logger("celery")

try:
    from celery import Celery

    celery_app = Celery(
        "clipforge",
        broker=settings.CELERY_BROKER_URL,
        backend=settings.CELERY_RESULT_BACKEND,
    )
    celery_app.conf.update(
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        worker_prefetch_multiplier=1,
        task_time_limit=14400,  # 4h hard limit per job
        task_soft_time_limit=13800,
        result_expires=86400,
        broker_connection_retry_on_startup=True,
    )
except ImportError:  # pragma: no cover - celery optional
    celery_app = None  # type: ignore[assignment]
    log.warning("celery not installed; TASK_BACKEND=thread will be used")


@celery_app.task(  # type: ignore[misc]
    name="clipforge.run_task",
    bind=True,
    max_retries=1,
    default_retry_delay=30,
)
def run_task(self, func_path: str, *args: Any, **kwargs: Any) -> Any:  # type: ignore[no-untyped-def]
    """Generic entrypoint: import a module-level function and run it.

    `submit_task` passes the dotted path so the same task covers both the
    pipeline and per-clip rendering.
    """
    setup_logging()
    module_name, _, name = func_path.rpartition(".")
    if not module_name or not name:
        raise ValueError(f"cannot resolve task function: {func_path!r}")
    module = importlib.import_module(module_name)
    func = getattr(module, name)
    log.info("celery task %s: %s(%s)", self.request.id, func_path, ", ".join(map(str, args)))
    return func(*args, **kwargs)


@celery_app.task(  # type: ignore[misc]
    name="clipforge.run_pipeline",
    bind=True,
    max_retries=1,
    default_retry_delay=30,
)
def run_pipeline_task(self, project_id: str) -> str:  # type: ignore[no-untyped-def]
    """Celery entrypoint: run the pipeline for one project."""
    setup_logging()
    from app.pipelines.video_pipeline import run_pipeline

    log.info("celery task %s: project %s", self.request.id, project_id)
    run_pipeline(project_id)
    return project_id
