"""Project endpoints: create, upload media, start pipeline, list, delete."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_current_user_or_query_token
from app.api.routes.auth import reset_daily_quota_if_needed
from app.core.config import settings
from app.core.logging import get_logger
from app.db.models.clip import Clip
from app.db.models.project import Project, ProjectStatus
from app.db.models.user import User
from app.db.session import get_db
from app.pipelines.video_pipeline import enqueue_pipeline, resume_pipeline
from app.schemas.clip import ClipResponse
from app.schemas.project import ProjectCreate, ProjectListResponse, ProjectResponse
from app.services import events
from app.services.storage import storage

log = get_logger("api.projects")
router = APIRouter(prefix="/projects", tags=["projects"])

_MAX_BYTES = int(settings.MAX_UPLOAD_GB * 1024 ** 3)


def _to_response(project: Project) -> ProjectResponse:
    return ProjectResponse.model_validate(project)


@router.post("", response_model=ProjectResponse, status_code=201)
def create_project(
    payload: ProjectCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> ProjectResponse:
    """Create a project (metadata only). Media is attached via /upload."""
    reset_daily_quota_if_needed(user, db)
    if user.jobs_today >= user.jobs_quota_per_day:
        raise HTTPException(
            status_code=429,
            detail=f"daily job quota reached ({user.jobs_quota_per_day}/day)",
        )
    project = Project(
        user_id=user.id,
        source_type=payload.source_type,
        source_name=payload.source_name,
        source_url=str(payload.source_url) if payload.source_url else None,
        language=payload.language,
        target_dur_min=payload.target_dur_min,
        target_dur_max=payload.target_dur_max,
        target_clip_count=payload.target_clip_count,
        desired_categories=",".join(payload.desired_categories),
        subtitle_style=payload.subtitle_style,
        target_fps=payload.target_fps,
        layout_mode=payload.layout_mode,
    )
    db.add(project)
    db.commit()
    db.refresh(project)
    log.info("created project %s (%s)", project.id, project.source_type)
    return _to_response(project)


@router.post("/{project_id}/upload", response_model=ProjectResponse)
async def upload_media(
    project_id: str,
    file: UploadFile = File(...),
    auto_start: bool = Form(True),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> ProjectResponse:
    """Attach an uploaded video file to a project and (optionally) start."""
    project = _owned_project(db, project_id, user)
    if project.source_type != "upload":
        raise HTTPException(
            status_code=400, detail="this project expects a YouTube URL, not an upload"
        )
    if project.status not in (ProjectStatus.PENDING, ProjectStatus.FAILED):
        raise HTTPException(status_code=409, detail="project already has media")

    data = await _read_upload(file)
    from app.services import ingest

    try:
        result = ingest.ingest_upload(project_id, file.filename or "video.mp4", data)
    except ingest.IngestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    project.source_name = result.title
    project.duration = result.duration
    project.width = result.width
    project.height = result.height
    project.fps = result.fps
    project.has_video_stream = result.has_video
    db.commit()
    db.refresh(project)
    events.publish_log(project_id, f"Uploaded {result.title} ({result.duration:.0f}s)")

    if auto_start:
        _start(project, db, user)
    return _to_response(project)


@router.post("/{project_id}/start", response_model=ProjectResponse)
def start_pipeline(
    project_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> ProjectResponse:
    """Start (or resume) the processing pipeline for a project."""
    project = _owned_project(db, project_id, user)
    _start(project, db, user)
    return _to_response(project)


def _start(project: Project, db: Session, user: User) -> None:
    reset_daily_quota_if_needed(user, db)
    if project.status == ProjectStatus.PENDING and not _has_source(project):
        raise HTTPException(status_code=400, detail="project has no source media yet")
    if project.status in (ProjectStatus.COMPLETED,):
        # resume: artifacts make completed stages cheap
        project.error = None
        db.commit()
    task_id = resume_pipeline(project.id)
    user.jobs_today += 1
    db.commit()
    events.publish_log(project.id, f"Pipeline queued (task {task_id})")


def _has_source(project: Project) -> bool:
    if project.source_type == "youtube":
        return bool(project.source_url)
    return any(
        f.name.startswith("source_") for f in storage.project_dir(project.id).iterdir()
    )


@router.post("/{project_id}/youtube", response_model=ProjectResponse)
def fetch_youtube(
    project_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> ProjectResponse:
    """Download the project's YouTube URL and start the pipeline."""
    project = _owned_project(db, project_id, user)
    if project.source_type != "youtube" or not project.source_url:
        raise HTTPException(status_code=400, detail="no YouTube URL configured")
    if project.status not in (ProjectStatus.PENDING, ProjectStatus.FAILED):
        raise HTTPException(status_code=409, detail="project already processed")
    _start(project, db, user)
    return _to_response(project)


@router.get("", response_model=ProjectListResponse)
def list_projects(
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> ProjectListResponse:
    """List the current user's projects, newest first."""
    q = db.query(Project).filter(Project.user_id == user.id)
    total = q.count()
    items = q.order_by(Project.created_at.desc()).offset(offset).limit(min(limit, 200)).all()
    return ProjectListResponse(items=[_to_response(p) for p in items], total=total)


@router.get("/{project_id}", response_model=ProjectResponse)
def get_project(
    project_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> ProjectResponse:
    project = _owned_project(db, project_id, user)
    return _to_response(project)


@router.get("/{project_id}/clips", response_model=list[ClipResponse])
def list_clips(
    project_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> list[ClipResponse]:
    _owned_project(db, project_id, user)
    clips = (
        db.query(Clip)
        .filter(Clip.project_id == project_id)
        .order_by(Clip.rank.asc())
        .all()
    )
    return [_clip_response(db, c) for c in clips]


@router.get("/{project_id}/events")
def project_events(
    project_id: str,
    token: str | None = None,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user_or_query_token),
) -> StreamingResponse:
    """SSE stream of pipeline progress for a project.

    The token may be passed as a `token` query param because EventSource
    cannot set Authorization headers.
    """
    _owned_project(db, project_id, user)

    async def generator():
        async for evt in events.bus.subscribe(project_id):
            yield evt.to_sse()

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.get("/{project_id}/clips/{clip_id}/video")
def serve_clip_video(
    project_id: str,
    clip_id: str,
    db: Session = Depends(get_db),
    # <video> tags cannot send an Authorization header, so ?token= is allowed
    user: User = Depends(get_current_user_or_query_token),
) -> FileResponse:
    _owned_project(db, project_id, user)
    clip = db.get(Clip, clip_id)
    if clip is None or clip.project_id != project_id or not clip.render_path:
        raise HTTPException(status_code=404, detail="rendered clip not found")
    path = storage.abspath(clip.render_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="clip file missing")
    return FileResponse(path, media_type="video/mp4", filename=f"{clip.title[:60]}.mp4")


@router.get("/{project_id}/clips/{clip_id}/thumb")
def serve_clip_thumb(
    project_id: str,
    clip_id: str,
    db: Session = Depends(get_db),
    # <img> tags cannot send an Authorization header, so ?token= is allowed
    user: User = Depends(get_current_user_or_query_token),
) -> FileResponse:
    _owned_project(db, project_id, user)
    clip = db.get(Clip, clip_id)
    if clip is None or clip.project_id != project_id or not clip.thumb_path:
        raise HTTPException(status_code=404, detail="thumbnail not found")
    return FileResponse(storage.abspath(clip.thumb_path), media_type="image/jpeg")


@router.delete("/{project_id}")
def delete_project(
    project_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Response:
    project = _owned_project(db, project_id, user)
    db.delete(project)
    db.commit()
    storage.delete_project(project_id)
    events.bus.clear(project_id)
    log.info("deleted project %s", project_id)
    return Response(status_code=204)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _owned_project(db: Session, project_id: str, user: User) -> Project:
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="project not found")
    if project.user_id != user.id:
        raise HTTPException(status_code=403, detail="not your project")
    return project


def _clip_response(db: Session, clip: Clip) -> ClipResponse:
    # ClipResponse derives video_url/thumb_url itself (see the schema validator).
    return ClipResponse.model_validate(clip)


async def _read_upload(file: UploadFile) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > _MAX_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"upload exceeds {settings.MAX_UPLOAD_GB}GB limit",
            )
        chunks.append(chunk)
    return b"".join(chunks)
