"""Artifact model (idempotent stage output) and Clip model (ranked output)."""

from __future__ import annotations

from sqlalchemy import Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, IdMixin, TimestampMixin


class Artifact(Base, IdMixin, TimestampMixin):
    """Idempotent pipeline stage output — a row existing means the stage is done.

    This is what makes the pipeline resumable: a step checks for an artifact of
    its `kind` and skips itself if one exists.
    """

    __tablename__ = "artifacts"

    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id"), index=True, nullable=False
    )
    kind: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    # Relative path inside STORAGE_DIR (server-side absolute paths never exposed).
    path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    meta_json: Mapped[str] = mapped_column(Text, default="", server_default="")

    project: Mapped["Project"] = relationship(back_populates="artifacts")  # noqa: F821
