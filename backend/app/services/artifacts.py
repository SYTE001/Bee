"""Artifact store: idempotent pipeline stage outputs.

A stage writes its result via `ArtifactStore.put(...)`; the next run finds it
via `get(...)` and skips work. This is the mechanism that makes the whole
pipeline resumable and re-runnable.
"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.db.models.artifact import Artifact
from app.services.storage import storage

log = get_logger("artifacts")


class ArtifactStore:
    """DB-backed record of completed pipeline stages."""

    def __init__(self, db: Session, project_id: str) -> None:
        self.db = db
        self.project_id = project_id

    def get(self, kind: str) -> Artifact | None:
        return (
            self.db.query(Artifact)
            .filter(Artifact.project_id == self.project_id, Artifact.kind == kind)
            .first()
        )

    def path_of(self, kind: str):
        """Absolute filesystem path of an artifact, or None if it has no file."""
        art = self.get(kind)
        if art is None or not art.path:
            return None
        return storage.abspath(art.path)

    def has(self, kind: str) -> bool:
        return self.get(kind) is not None

    def get_meta(self, kind: str) -> dict[str, Any]:
        art = self.get(kind)
        if not art or not art.meta_json:
            return {}
        try:
            return json.loads(art.meta_json)
        except json.JSONDecodeError:
            return {}

    def put(
        self,
        kind: str,
        abs_path: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> Artifact:
        """Create or replace the artifact for `kind`."""
        rel = storage.relpath(abs_path) if abs_path else None
        existing = self.get(kind)
        if existing:
            existing.path = rel
            existing.meta_json = json.dumps(meta or {}, ensure_ascii=False)
            self.db.commit()
            log.debug("artifact updated: %s", kind)
            return existing
        art = Artifact(
            project_id=self.project_id,
            kind=kind,
            path=rel,
            meta_json=json.dumps(meta or {}, ensure_ascii=False),
        )
        self.db.add(art)
        self.db.commit()
        log.info("artifact stored: %s -> %s", kind, rel or "(in-memory)")
        return art

    def delete(self, kind: str) -> None:
        art = self.get(kind)
        if art:
            self.db.delete(art)
            self.db.commit()

    def all_kinds(self) -> list[str]:
        rows = (
            self.db.query(Artifact.kind)
            .filter(Artifact.project_id == self.project_id)
            .all()
        )
        return [r[0] for r in rows]
