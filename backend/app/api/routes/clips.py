"""Clip endpoints: subtitle styles, the clip editor, rendering, batch download."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_current_user_or_query_token
from app.core.logging import get_logger
from app.db.models.clip import Clip
from app.db.models.project import Project, ProjectStatus
from app.db.models.user import User
from app.db.session import get_db
from app.pipelines.video_pipeline import enqueue_render
from app.schemas.clip import ClipResponse, ClipUpdate
from app.services import events, render
from app.services.storage import storage
from app.services.subtitle import list_styles
from app.services.subtitle_styles import load_all_styles

log = get_logger("api.clips")
router = APIRouter(tags=["clips"])

_VALID_LAYOUTS = {"auto", "single_focus", "split_screen", "fit_blur", "fill_zoom"}


# --------------------------------------------------------------------------
# Subtitle styles (for the editor dropdown + live preview)
# --------------------------------------------------------------------------
_STYLE_FIELDS = (
    "id", "name", "font", "font_size", "primary_color", "secondary_color",
    "outline_color", "back_color", "bold", "italic", "uppercase", "alignment",
    "margin_v", "margin_l", "margin_r", "words_per_line", "max_line_chars",
    "animation", "animation_intensity", "position", "outline", "shadow",
    "keyword_color", "keyword_bold", "emoji", "border_style", "scale_x",
    "scale_y", "spacing", "rotation",
)


@router.get("/subtitle-styles")
def subtitle_styles() -> list[dict]:
    """Preset metadata for the frontend style picker and preview renderer."""
    styles = []
    for style in load_all_styles().values():
        data = dataclasses.asdict(style)
        styles.append({k: data[k] for k in _STYLE_FIELDS if k in data})
    return styles


# --------------------------------------------------------------------------
# Clip editor
# --------------------------------------------------------------------------
def _owned_project(db: Session, project_id: str, user: User) -> Project:
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="project not found")
    if project.user_id != user.id:
        raise HTTPException(status_code=403, detail="not your project")
    return project


def _owned_clip(db: Session, project_id: str, clip_id: str, user: User) -> Clip:
    _owned_project(db, project_id, user)
    clip = db.get(Clip, clip_id)
    if clip is None or clip.project_id != project_id:
        raise HTTPException(status_code=404, detail="clip not found")
    return clip


def _clip_response(clip: Clip) -> ClipResponse:
    # ClipResponse derives video_url/thumb_url itself (see the schema validator).
    return ClipResponse.model_validate(clip)


@router.patch("/projects/{project_id}/clips/{clip_id}", response_model=ClipResponse)
def update_clip(
    project_id: str,
    clip_id: str,
    payload: ClipUpdate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> ClipResponse:
    """Apply manual edits from the clip editor.

    Any change that affects the output (timing, style, layout, fps, words)
    invalidates the existing render so the UI offers a re-render.
    """
    clip = _owned_clip(db, project_id, clip_id, user)

    if payload.subtitle_style is not None:
        if payload.subtitle_style not in load_all_styles():
            raise HTTPException(status_code=400, detail="unknown subtitle style")
    if payload.layout_mode is not None and payload.layout_mode not in _VALID_LAYOUTS:
        raise HTTPException(status_code=400, detail="unknown layout mode")

    edits = payload.model_dump(exclude_unset=True)
    needs_rerender = False

    if "start" in edits or "end" in edits:
        start = edits.get("start", clip.start)
        end = edits.get("end", clip.end)
        if end <= start:
            raise HTTPException(status_code=400, detail="end must be after start")
        clip.start = round(float(start), 3)
        clip.end = round(float(end), 3)
        clip.duration = round(clip.end - clip.start, 3)
        clip.transcript = payload.transcript or clip.transcript
        needs_rerender = True

    for field in ("title", "description", "hashtags"):
        if field in edits and edits[field] is not None:
            clip.__setattr__(field, edits[field])
            needs_rerender = True

    if payload.words is not None:
        if not payload.words:
            raise HTTPException(status_code=400, detail="words cannot be empty")
        clip.words = [w.model_dump() for w in payload.words]
        clip.transcript = " ".join(w.word for w in payload.words)
        needs_rerender = True

    for field in ("subtitle_style", "layout_mode", "target_fps"):
        if edits.get(field) is not None and edits[field] != getattr(clip, field):
            clip.__setattr__(field, edits[field])
            needs_rerender = True

    if needs_rerender:
        clip.rendered = False
        clip.render_error = None
    db.commit()
    db.refresh(clip)
    log.info("clip %s edited (re-render needed: %s)", clip.id, needs_rerender)
    return _clip_response(clip)


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
@router.post("/projects/{project_id}/clips/{clip_id}/render", status_code=202)
def render_one_clip(
    project_id: str,
    clip_id: str,
    force: bool = False,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict:
    """Queue a single clip for (re-)rendering. Returns immediately."""
    project = _owned_project(db, project_id, user)
    clip = _owned_clip(db, project_id, clip_id, user)
    task_id = enqueue_render(project.id, clip.id, force=force)
    events.publish_log(project.id, f"Rendering clip #{clip.rank} (task {task_id})")
    return {"task_id": task_id, "clip_id": clip.id}


@router.post("/projects/{project_id}/render", status_code=202)
def render_all_clips(
    project_id: str,
    force: bool = False,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict:
    """Queue every clip of a project for rendering."""
    project = _owned_project(db, project_id, user)
    if project.status not in (
        ProjectStatus.COMPLETED, ProjectStatus.RENDERING, ProjectStatus.FAILED,
    ):
        raise HTTPException(
            status_code=409, detail="finish the pipeline before rendering"
        )
    task_id = enqueue_render(project.id, None, force=force)
    events.publish_log(project.id, f"Rendering all clips (task {task_id})")
    return {"task_id": task_id, "project_id": project.id}


# --------------------------------------------------------------------------
# Batch download
# --------------------------------------------------------------------------
@router.get("/projects/{project_id}/download.zip")
def download_project(
    project_id: str,
    db: Session = Depends(get_db),
    # direct browser navigation cannot send an Authorization header
    user: User = Depends(get_current_user_or_query_token),
) -> FileResponse:
    """Download every rendered clip as a ZIP, with a text index of metadata."""
    project = _owned_project(db, project_id, user)
    try:
        path = render.zip_rendered_clips(db, project)
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    safe = "".join(
        c for c in project.source_name if c.isalnum() or c in "-_."
    )[:40] or "clips"
    return FileResponse(
        path,
        media_type="application/zip",
        filename=f"clipforge_{safe}.zip",
    )
