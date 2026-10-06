"""Project schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

SourceType = Literal["upload", "youtube"]
LayoutMode = Literal["auto", "single_focus", "split_screen", "fit_blur", "fill_zoom"]


class ProjectCreate(BaseModel):
    """Metadata submitted together with (or right before) the media upload."""

    source_type: SourceType = "upload"
    source_name: str = Field(min_length=1, max_length=255)
    source_url: HttpUrl | None = None

    language: str = Field(default="auto", max_length=12)
    target_dur_min: int = Field(default=15, ge=5, le=90)
    target_dur_max: int = Field(default=60, ge=5, le=90)
    target_clip_count: int = Field(default=10, ge=1, le=50)
    desired_categories: list[str] = Field(default_factory=list)
    subtitle_style: str = Field(default="classic_bold_white")
    target_fps: int = Field(default=60, ge=24, le=90)
    layout_mode: LayoutMode = "auto"

    @model_validator(mode="after")
    def _check(self) -> ProjectCreate:
        if self.source_type == "youtube" and not self.source_url:
            raise ValueError("source_url is required for youtube source")
        if self.target_dur_max < self.target_dur_min:
            raise ValueError("target_dur_max must be >= target_dur_min")
        return self


class ProjectResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    status: str
    stage_message: str
    stage_progress: float
    error: str | None = None
    source_type: str
    source_name: str
    duration: float | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    detected_language: str | None = None
    num_speakers: int = 0
    created_at: datetime
    clips: list["ClipResponse"] = Field(default_factory=list)


class ProjectListResponse(BaseModel):
    items: list[ProjectResponse]
    total: int


from app.schemas.clip import ClipResponse  # noqa: E402

ProjectResponse.model_rebuild()
