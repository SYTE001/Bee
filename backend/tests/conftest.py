"""Shared pytest fixtures."""

from __future__ import annotations

import os
import sys
import subprocess
import uuid
from pathlib import Path

import pytest

# Make `app` importable when tests run from repo root
BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

# Isolated dev database + storage before importing app modules.
# NOTE: a *file* sqlite is required — with :memory: the worker thread's
# SessionLocal connection would get its own separate empty database.
os.environ.setdefault("DATABASE_URL", "sqlite:///./data/test_storage/clipforge_test.db")
os.environ.setdefault("STORAGE_DIR", "./data/test_storage")
os.environ.setdefault("TASK_BACKEND", "thread")

from app.db.base import Base  # noqa: E402
from app.db.models.user import User  # noqa: E402
from app.db.session import SessionLocal, create_all, engine  # noqa: E402
from app.core.security import create_access_token, hash_password  # noqa: E402
from app.pipelines.video_pipeline import ARTIFACT_SOURCE  # noqa: E402
from app.services import artifacts  # noqa: E402
from app.services.storage import storage  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _db_schema():
    create_all()
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(autouse=True)
def _stub_llm(monkeypatch):
    """Never hit a real LLM in tests.

    The pipeline imports llm_analyzer lazily and calls analyze_chunk over the
    network; a configured key would otherwise make the suite depend on a live
    endpoint and on reasoning-model latency (tens of seconds per chunk).
    """
    from app.services import llm_analyzer

    def _fast_analyze(sentences, chunk_index: int = 0):  # noqa: ARG001 - match analyze_chunk
        return llm_analyzer.heuristic_analysis(sentences)

    monkeypatch.setattr(llm_analyzer, "analyze_chunk", _fast_analyze)


@pytest.fixture()
def db():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def user(db) -> User:
    # Unique per invocation: the email column is unique and several tests in
    # one session each request this fixture.
    u = User(
        email=f"tester-{uuid.uuid4().hex[:8]}@example.com",
        hashed_password=hash_password("password123"),
        display_name="Tester",
    )
    db.add(u)
    db.commit()
    db.refresh(u)
    return u


@pytest.fixture()
def auth_token(user) -> str:
    return create_access_token(user.id, extra={"email": user.email})


# --------------------------------------------------------------------------
# Fixtures backed by a real (tiny) generated video, for ffmpeg-level tests.
# --------------------------------------------------------------------------
def make_video(
    path: Path, width=320, height=240, duration=2.0, fps=12, audio=False,
    pattern: str = "testsrc",
) -> Path:
    """Synthesize a deterministic test video with ffmpeg."""
    # `testsrc=s=...:d=...` but `color=c=black:s=...:d=...` (already has a key).
    sep = ":" if "=" in pattern else "="
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "lavfi"]
    cmd += ["-i", f"{pattern}{sep}s={width}x{height}:d={duration}:r={fps}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=220:duration={duration}"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    if audio:
        cmd += ["-c:a", "aac", "-shortest"]
    cmd += [str(path)]
    subprocess.run(cmd, check=True, capture_output=True, cwd=path.parent)
    return path


@pytest.fixture()
def isolated_storage(tmp_path, monkeypatch):
    """Point the storage singleton at a per-test directory."""
    root = tmp_path / "storage"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(storage, "base", root.resolve())
    return root


def _create_video_project(
    db, tmp_path, *, pattern: str = "testsrc", audio: bool = True
):
    """Build a project + source artifact over a freshly generated video."""
    import shutil

    from app.db.models.project import Project, SourceType

    user = User(email=f"stage-{uuid.uuid4().hex[:8]}@example.com", hashed_password="x")
    db.add(user)
    db.commit()

    proj = Project(
        user_id=user.id,
        source_type=SourceType.UPLOAD,
        source_name="stage.mp4",
        target_dur_min=5,
        target_dur_max=10,
        target_clip_count=2,
    )
    db.add(proj)
    db.commit()

    video = make_video(tmp_path / "stage.mp4", audio=audio, pattern=pattern)
    target = storage.project_dir(proj.id) / "source_stage.mp4"
    shutil.copy(video, target)

    store = artifacts.ArtifactStore(db, proj.id)
    store.put(
        ARTIFACT_SOURCE,
        str(target),  # put() relativizes internally; pass the absolute path
        {
            "duration": 2.0,
            "width": 320,
            "height": 240,
            "fps": 12.0,
            "has_video": True,
            "has_audio": audio,
        },
    )
    proj.width, proj.height, proj.duration = 320, 240, 2.0
    db.commit()
    return proj, store


@pytest.fixture()
def project_with_video(db, isolated_storage, tmp_path):
    """A project whose source artifact points at a real generated video.

    Returns (project, artifact_store). Clip rows are left to each test.
    """
    return _create_video_project(db, tmp_path)


@pytest.fixture()
def project_with_black_video(db, isolated_storage, tmp_path):
    """Same, but on a plain black canvas — for pixel-level subtitle checks."""
    return _create_video_project(db, tmp_path, pattern="color=c=black")


@pytest.fixture()
def make_clip(db):
    """Factory inserting one clip row with word-level timings.

        clip = make_clip(project, start=0.2, end=1.0, subtitle_style="hormozi")
    """
    from app.db.models.clip import Clip

    def _make(project, *, rank=1, start=0.0, end=1.5, **kwargs) -> Clip:
        text = kwargs.pop("text", "this is a short test clip")
        words = kwargs.pop("words", None)
        if words is None:
            tokens = text.split()
            span = max(end - start - 0.05, 0.1)
            words = [
                {
                    "word": tok,
                    "start": start + span * i / len(tokens),
                    "end": start + span * (i + 1) / len(tokens),
                }
                for i, tok in enumerate(tokens)
            ]
        clip = Clip(
            project_id=project.id,
            rank=rank,
            start=start,
            end=end,
            duration=round(end - start, 3),
            viral_score=70.0,
            category_primary="insight",
            category_secondary="story",
            title="Short Test Clip",
            description="",
            hashtags="#test",
            transcript=text,
            words=words,
            subtitle_style=kwargs.pop("subtitle_style", "classic_bold_white"),
            layout_mode=kwargs.pop("layout_mode", "fit_blur"),
            target_fps=kwargs.pop("target_fps", 30),
        )
        for key, value in kwargs.items():
            setattr(clip, key, value)
        db.add(clip)
        db.commit()
        return clip

    return _make
