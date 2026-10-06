"""In-process event bus for real-time pipeline progress (SSE).

Workers publish stage events here; the SSE endpoint subscribes for a project.
For multi-worker deployments, replace this with Redis pub/sub — the
`publish()`/`subscribe()` interface is the contract to keep.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, AsyncGenerator

from app.core.logging import get_logger

log = get_logger("events")

# Keep the last N events per project so a reconnecting client can catch up.
HISTORY = 200


@dataclass
class Event:
    project_id: str
    type: str  # stage | progress | log | error | clip | done
    data: dict[str, Any]
    ts: float = field(default_factory=time.time)

    def to_sse(self) -> str:
        payload = json.dumps({"type": self.type, "data": self.data, "ts": self.ts})
        return f"event: {self.type}\ndata: {payload}\n\n"


class EventBus:
    def __init__(self) -> None:
        self._history: dict[str, deque[Event]] = defaultdict(lambda: deque(maxlen=HISTORY))
        self._queues: dict[str, set[asyncio.Queue]] = defaultdict(set)
        self._lock = Lock()

    def publish(self, project_id: str, type: str, data: dict[str, Any]) -> Event:
        evt = Event(project_id=project_id, type=type, data=data)
        with self._lock:
            self._history[project_id].append(evt)
            for q in list(self._queues.get(project_id, ())):
                try:
                    q.put_nowait(evt)
                except asyncio.QueueFull:  # pragma: no cover
                    pass
        return evt

    def history(self, project_id: str) -> list[Event]:
        with self._lock:
            return list(self._history.get(project_id, ()))

    async def subscribe(self, project_id: str) -> AsyncGenerator[Event, None]:
        q: asyncio.Queue = asyncio.Queue(maxsize=512)
        with self._lock:
            self._queues[project_id].add(q)
        try:
            for evt in self.history(project_id):
                yield evt
            while True:
                evt = await q.get()
                yield evt
                if evt.type in ("done", "error"):
                    return
        finally:
            with self._lock:
                self._queues.get(project_id, set()).discard(q)

    def clear(self, project_id: str) -> None:
        with self._lock:
            self._history.pop(project_id, None)


bus = EventBus()


def publish_stage(project_id: str, status: str, message: str, progress: float = 0.0) -> None:
    """Convenience helper: report a pipeline stage transition."""
    bus.publish(
        project_id,
        "stage",
        {"status": status, "message": message, "progress": round(progress, 3)},
    )
    log.info("[%s] %s (%.0f%%) %s", project_id, status, progress * 100, message)


def publish_progress(project_id: str, progress: float, message: str = "") -> None:
    bus.publish(project_id, "progress", {"progress": round(progress, 3), "message": message})


def publish_log(project_id: str, message: str, level: str = "info") -> None:
    bus.publish(project_id, "log", {"message": message, "level": level})


def publish_error(project_id: str, message: str) -> None:
    bus.publish(project_id, "error", {"message": message})
    log.error("[%s] %s", project_id, message)


def publish_done(project_id: str, clip_count: int = 0) -> None:
    bus.publish(project_id, "done", {"clip_count": clip_count})
