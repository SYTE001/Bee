"""Clip model — a ranked, generated vertical clip."""

from __future__ import annotations

from sqlalchemy import Boolean, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from app.db.base import Base, IdMixin, TimestampMixin

# JSONB on Postgres, plain JSON elsewhere (sqlite).
_Json = JSONB().with_variant(JSON(), "sqlite")


class Clip(Base, IdMixin, TimestampMixin):
    """A candidate clip that was selected, ranked, and (optionally) rendered."""

    __tablename__ = "clips"

    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id"), index=True, nullable=False
    )

    # --- selection
    rank: Mapped[int] = mapped_column(Integer, index=True, nullable=False)
    start: Mapped[float] = mapped_column(Float, nullable=False)
    end: Mapped[float] = mapped_column(Float, nullable=False)
    duration: Mapped[float] = mapped_column(Float, nullable=False)

    # --- scoring
    viral_score: Mapped[float] = mapped_column(Float, default=0.0)
    category_primary: Mapped[str] = mapped_column(String(48), default="")
    category_secondary: Mapped[str] = mapped_column(String(48), default="")
    # per-dimension scores, e.g. {"hook": 0.8, "laughter": 0.3, ...}
    scores: Mapped[dict | None] = mapped_column(_Json, nullable=True)
    # explainable reasons (list of short human strings)
    reasons: Mapped[list | None] = mapped_column(_Json, nullable=True)

    # --- text generation
    title: Mapped[str] = mapped_column(String(300), default="")
    description: Mapped[str] = mapped_column(Text, default="")
    hashtags: Mapped[str] = mapped_column(String(300), default="")

    # --- transcript of the clip (editable)
    transcript: Mapped[str] = mapped_column(Text, default="")
    # word-level timings JSON: [{word, start, end, speaker, confidence}, ...]
    words: Mapped[list | None] = mapped_column(_Json, nullable=True)

    # --- render state
    subtitle_style: Mapped[str] = mapped_column(String(80), default="classic_bold_white")
    layout_mode: Mapped[str] = mapped_column(String(40), default="auto")
    target_fps: Mapped[int] = mapped_column(Integer, default=60)
    render_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    thumb_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    rendered: Mapped[bool] = mapped_column(Boolean, default=False, server_default="0")
    render_error: Mapped[str | None] = mapped_column(Text, nullable=True)

    project: Mapped["Project"] = relationship(back_populates="clips")  # noqa: F821
