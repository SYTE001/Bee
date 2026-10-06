"""Tests for the task runner abstraction (thread + celery dispatch paths)."""

from __future__ import annotations

import subprocess
import sys
import time

import pytest

from app.core.config import settings
from app.services import task_runner


@pytest.fixture(autouse=True)
def _reset_executor():
    yield
    if task_runner._executor is not None:
        task_runner._executor.shutdown(wait=False)
        task_runner._executor = None


# --------------------------------------------------------------------------
# module-level target (what submit_task actually receives in production)
# --------------------------------------------------------------------------
def _sample_task(value: int, *, multiplier: int = 1) -> int:
    return value * multiplier


class TestThreadPool:
    def test_runs_and_returns_id(self):
        task_id = task_runner.submit_task(_sample_task, 4, multiplier=3)
        assert isinstance(task_id, str) and task_id

    def test_result_is_delivered(self):
        from concurrent.futures import Future

        fut: Future[int] = Future()
        task_runner.submit_task(lambda v: fut.set_result(v * 2), 21)
        assert fut.result(timeout=10) == 42

    def test_failure_does_not_leak(self, capsys):
        def boom() -> None:
            raise RuntimeError("boom")

        task_runner.submit_task(boom)
        # the wrapper logs the exception instead of propagating it. setup_logging
        # reconfigures the root logger in the worker thread, so read stdout
        # rather than relying on pytest's log capture handler.
        deadline = time.time() + 10
        while time.time() < deadline:
            if "RuntimeError: boom" in capsys.readouterr().out:
                break
            time.sleep(0.05)
        else:
            pytest.fail("the exception was never logged")


class TestCeleryDispatch:
    """The celery path must route by function identity, not hardcode the pipeline."""

    def test_dispatches_by_dotted_path(self, monkeypatch):
        calls = []

        class FakeResult:
            id = "celery-task-42"

        monkeypatch.setattr(settings, "TASK_BACKEND", "celery", raising=False)

        import app.workers.celery_app as celery_app

        monkeypatch.setattr(
            celery_app,
            "run_task",
            type("FakeTask", (), {"delay": staticmethod(lambda path, *a, **kw: calls.append((path, a, kw)) or FakeResult())}),
            raising=False,
        )

        from app.pipelines import video_pipeline as vp

        task_id = task_runner.submit_task(vp.run_render, "proj-1", "clip-9", force=True)
        assert task_id == "celery-task-42"
        assert calls == [
            ("app.pipelines.video_pipeline.run_render", ("proj-1", "clip-9"), {"force": True})
        ]

    def test_render_job_is_not_routed_to_the_pipeline(self, monkeypatch):
        """Regression: submit_task used to call run_pipeline_task for every job."""
        imported = {}

        class FakeResult:
            id = "ok"

        def fake_delay(path, *args, **kwargs):
            imported["path"] = path
            return FakeResult()

        monkeypatch.setattr(settings, "TASK_BACKEND", "celery", raising=False)

        import app.workers.celery_app as celery_app

        monkeypatch.setattr(
            celery_app, "run_task",
            type("FakeTask", (), {"delay": staticmethod(fake_delay)}),
            raising=False,
        )
        # run_pipeline_task must never be called by submit_task
        monkeypatch.setattr(
            celery_app, "run_pipeline_task",
            type("BadTask", (), {"delay": staticmethod(lambda *a, **kw: pytest.fail("routed to the pipeline task"))}),
            raising=False,
        )

        from app.pipelines import video_pipeline as vp

        task_runner.submit_task(vp.run_render, "p", "c", force=True)
        assert imported["path"].endswith("run_render")


class TestCeleryApp:
    def test_generic_task_resolves_and_runs(self):
        pytest.importorskip("celery")
        from app.workers.celery_app import run_task

        # celery binds `self` already, so .run takes the task's own arguments
        result = run_task.run(f"{_sample_task.__module__}._sample_task", 6, multiplier=7)
        assert result == 42

    def test_generic_task_rejects_bad_path(self):
        pytest.importorskip("celery")
        from app.workers.celery_app import run_task

        with pytest.raises(ValueError):
            run_task.run("no_module_at_all", 1)

    def test_pipeline_task_still_registered(self):
        pytest.importorskip("celery")
        from app.workers.celery_app import celery_app, run_pipeline_task

        assert celery_app is not None
        assert run_pipeline_task.name == "clipforge.run_pipeline"
