"""Project model — one source video = one project."""

from __future__ import annotations

import enum

from sqlalchemy import Enum, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, IdMixin, TimestampMixin

class ProjectStatus(str, enum.Enum):
    """Lifecycle of a project. Statuses are monotonic-ish and resumable."""

    PENDING = "pending"
    INGESTING = "ingesting"
    EXTRACTING_AUDIO = "extracting_audio"
    TRANSCRIBING = "transcribing"
    DIARIZING = "diarizing"
    ANALYZING = "analyzing"
    SEGMENTING = "segmenting"
    SCORING = "scoring"
    REFRAMING = "reframing"
    SUBTITLING = "subtitling"
    RENDERING = "rendering"
    COMPLETED = "completed"
    FAILED = "failed"


class SourceType(str, enum.Enum):
    UPLOAD = "upload"
    YOUTUBE = "youtube"


class Project(Base, IdMixin, TimestampMixin):
    __tablename__ = "projects"

    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True, nullable=False)

    # --- source
    source_type: Mapped[str] = mapped_column(Enum(SourceType), nullable=False)
    source_name: Mapped[str] = mapped_column(String(255), nullable=False)
    source_url: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    # --- media metadata (filled during ingest)
    duration: Mapped[float | None] = mapped_column(Float, nullable=True)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    fps: Mapped[float | None] = mapped_column(Float, nullable=True)
    has_video_stream: Mapped[bool] = mapped_column(default=False, server_default="0")

    # --- transcription
    language: Mapped[str] = mapped_column(String(12), default="auto", server_default="auto")
    detected_language: Mapped[str | None] = mapped_column(String(12), nullable=True)
    num_speakers: Mapped[int] = mapped_column(Integer, default=0, server_default="0")

    # --- user processing options
    target_dur_min: Mapped[int] = mapped_column(Integer, default=15)
    target_dur_max: Mapped[int] = mapped_column(Integer, default=60)
    target_clip_count: Mapped[int] = mapped_column(Integer, default=10)
    desired_categories: Mapped[str] = mapped_column(String(255), default="")
    subtitle_style: Mapped[str] = mapped_column(String(80), default="classic_bold_white")
    target_fps: Mapped[int] = mapped_column(Integer, default=60)
    layout_mode: Mapped[str] = mapped_column(String(40), default="auto")

    # --- pipeline
    status: Mapped[str] = mapped_column(
        Enum(ProjectStatus), default=ProjectStatus.PENDING, index=True
    )
    stage_message: Mapped[str] = mapped_column(String(512), default="")
    stage_progress: Mapped[float] = mapped_column(Float, default=0.0, server_default="0")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    artifacts: Mapped[list["Artifact"]] = relationship(  # noqa: F821
        back_populates="project", cascade="all, delete-orphan"
    )
    clips: Mapped[list["Clip"]] = relationship(  # noqa: F821
        back_populates="project", cascade="all, delete-orphan"
    )
