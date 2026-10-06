"""Clip schemas."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ClipWordSchema(BaseModel):
    """One word with word-level timing."""

    word: str
    start: float
    end: float
    speaker: str | None = None
    confidence: float = 1.0


class ClipResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    project_id: str
    rank: int
    start: float
    end: float
    duration: float
    viral_score: float
    category_primary: str
    category_secondary: str
    scores: dict | None = None
    reasons: list[str] | None = None
    title: str
    description: str
    hashtags: str
    transcript: str
    words: list[ClipWordSchema] | None = None
    subtitle_style: str
    layout_mode: str
    target_fps: int
    rendered: bool
    render_error: str | None = None
    # ORM-only inputs used to derive the URLs below; excluded from the payload.
    render_path: str | None = Field(default=None, exclude=True)
    thumb_path: str | None = Field(default=None, exclude=True)
    # Relative URL for the rendered mp4 / thumbnail (empty if unrendered)
    video_url: str | None = None
    thumb_url: str | None = None

    @model_validator(mode="after")
    def _media_urls(self) -> "ClipResponse":
        """Derive the media URLs so every endpoint serves the same paths."""
        if self.rendered and self.render_path and not self.video_url:
            self.video_url = f"/api/projects/{self.project_id}/clips/{self.id}/video"
        if self.thumb_path and not self.thumb_url:
            self.thumb_url = f"/api/projects/{self.project_id}/clips/{self.id}/thumb"
        return self


class ClipUpdate(BaseModel):
    """Manual edits from the clip editor."""

    start: float | None = Field(default=None, ge=0)
    end: float | None = Field(default=None, ge=0)
    transcript: str | None = None
    title: str | None = None
    description: str | None = None
    hashtags: str | None = None
    subtitle_style: str | None = None
    layout_mode: str | None = None
    target_fps: int | None = Field(default=None, ge=24, le=90)
    # Optional full replacement of word timings when transcript is edited.
    words: list[ClipWordSchema] | None = None
