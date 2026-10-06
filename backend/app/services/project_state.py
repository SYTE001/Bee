"""Helpers for reading and updating project state inside pipeline stages."""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.db.models.project import Project, ProjectStatus
from app.services import events

log = get_logger("project_state")


def set_stage(
    db: Session,
    project: Project,
    status: ProjectStatus,
    message: str,
    progress: float = 0.0,
) -> None:
    """Persist + broadcast a stage transition."""
    project.status = status
    project.stage_message = message
    project.stage_progress = max(0.0, min(1.0, progress))
    db.commit()
    events.publish_stage(project.id, status.value, message, project.stage_progress)


def set_progress(db: Session, project: Project, progress: float, message: str = "") -> None:
    project.stage_progress = max(0.0, min(1.0, progress))
    if message:
        project.stage_message = message
    db.commit()
    events.publish_progress(project.id, project.stage_progress, message)


def fail(db: Session, project: Project, message: str) -> None:
    """Mark the project failed and broadcast the error."""
    project.status = ProjectStatus.FAILED
    project.error = message
    project.stage_message = "Failed: " + message[:400]
    db.commit()
    events.publish_error(project.id, message)
    log.error("project %s failed: %s", project.id, message)


def complete(db: Session, project: Project, clip_count: int = 0) -> None:
    project.status = ProjectStatus.COMPLETED
    project.stage_progress = 1.0
    project.stage_message = f"Completed — {clip_count} clip(s)"
    db.commit()
    events.publish_done(project.id, clip_count)


def get_project(db: Session, project_id: str) -> Project | None:
    return db.get(Project, project_id)


def project_meta(project: Project) -> dict[str, Any]:
    return {
        "target_dur_min": project.target_dur_min,
        "target_dur_max": project.target_dur_max,
        "target_clip_count": project.target_clip_count,
        "desired_categories": [c for c in project.desired_categories.split(",") if c],
        "language": project.language,
        "subtitle_style": project.subtitle_style,
        "target_fps": project.target_fps,
        "layout_mode": project.layout_mode,
    }
