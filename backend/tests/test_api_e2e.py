"""End-to-end Fase 1 smoke test against the live FastAPI app (TestClient).

Generates a real media file with ffmpeg (synthetic speech-like tone + text),
uploads it, and runs the pipeline up to transcription — with a stubbed
transcriber so the test does not need the whisper model download.
"""

from __future__ import annotations

import subprocess
import uuid
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from app.db.models.project import Project, ProjectStatus
from app.services import transcribe as tr
from app.services.segments import Word

pytest.importorskip("fastapi.testclient")
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.db.session import get_db  # noqa: E402


def _synth_video(path: Path, seconds: float = 3.0) -> Path:
    """Create a tiny mp4 with an audio tone using ffmpeg lavfi sources."""
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi",
        "-i", f"color=c=blue:s=320x240:d={seconds}:r=24",
        "-f", "lavfi",
        "-i", f"sine=frequency=300:duration={seconds}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest",
        str(path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return path


def _fake_transcribe(audio_path, *, language="auto", cache_path=None, on_progress=None, on_status=None):
    """Deterministic stand-in for faster-whisper (no model download needed).

    Produces ~95s of plausible podcast speech so the scorer has enough material
    to build clips inside a 15–60s band.
    """
    sentences = [
        "Welcome back to the show everyone, today we have a really special guest.",
        "So tell me, how did you actually get started in this industry?",
        "Nobody talks about this, but the secret is consistency over talent.",
        "Wait, really? That is honestly shocking to hear.",
        "Yes, and the numbers back it up completely, it is crazy.",
        "I think the biggest mistake people make is giving up too early.",
        "That is such an important point, and honestly it changed my whole mindset.",
        "Let me give you three actionable steps you can use today.",
        "First, audit your time for one week and cut the bottom twenty percent.",
        "Second, batch your deep work into ninety minute blocks.",
        "Third, review your progress every single Friday afternoon.",
        "Honestly, if you do just those three things, your output doubles.",
        "That is insane value, and I have never heard it explained that way.",
        "There is one more thing though, and nobody ever mentions it.",
        "The emotional toll of entrepreneurship is the real challenge.",
        "I remember sitting in my car crying after that deal fell apart.",
        "But that failure taught me more than any success ever did.",
        "This is why I always say failure is just data in disguise.",
        "What advice would you give to someone listening right now?",
        "Start before you are ready, because perfect is the enemy of done.",
    ]
    words: list[Word] = []
    t = 0.0
    for sentence in sentences:
        for w in sentence.split():
            words.append(Word(word=w, start=round(t, 3), end=round(t + 0.28, 3)))
            t += 0.32
        t += 0.9  # inter-sentence pause
    result = tr.TranscriptionResult(
        text=" ".join(sentences),
        language="en",
        words=words,
        segments=[{"start": 0.0, "end": t, "text": " ".join(sentences)}],
        duration=t,
    )
    if on_progress:
        on_progress(1.0)
    return result


@pytest.fixture()
def client(db):
    """TestClient with the per-test DB session override."""
    from app.db.session import SessionLocal

    def _override():
        yield db

    app.dependency_overrides[get_db] = _override
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def _register_and_login(client, email: str | None = None) -> str:
    addr = email or f"e2e-{uuid.uuid4().hex[:8]}@example.com"
    r = client.post(
        "/api/auth/register", json={"email": addr, "password": "password123"}
    )
    assert r.status_code == 201, r.text
    return r.json()["access_token"]


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


class TestAuth:
    def test_register_login_me(self, client):
        token = _register_and_login(client)
        r = client.get("/api/auth/me", headers=_auth(token))
        assert r.status_code == 200
        assert r.json()["email"].endswith("@example.com")

    def test_login_wrong_password(self, client):
        token = _register_and_login(client)
        # decode the e-mail from the JWT to log in against the right account
        import jwt as _jwt

        from app.core.config import settings as _s

        email = _jwt.decode(token, _s.SECRET_KEY, algorithms=[_s.JWT_ALGORITHM])["email"]
        r = client.post("/api/auth/login", json={"email": email, "password": "nope"})
        assert r.status_code == 401
        ok = client.post("/api/auth/login", json={"email": email, "password": "password123"})
        assert ok.status_code == 200

    def test_protected_without_token(self, client):
        assert client.get("/api/auth/me").status_code == 401


class TestUploadAndPipeline:
    def test_full_fase1_flow(self, client, db, tmp_path):
        video = _synth_video(tmp_path / "podcast.mp4")
        token = _register_and_login(client)

        create = client.post(
            "/api/projects",
            json={
                "source_type": "upload",
                "source_name": "podcast.mp4",
                "language": "en",
                "target_dur_min": 15,
                "target_dur_max": 60,
                "target_clip_count": 5,
            },
            headers=_auth(token),
        )
        assert create.status_code == 201, create.text
        project_id = create.json()["id"]

        with patch.object(tr, "transcribe_audio", _fake_transcribe):
            with open(video, "rb") as fh:
                up = client.post(
                    f"/api/projects/{project_id}/upload",
                    files={"file": ("podcast.mp4", fh, "video/mp4")},
                    data={"auto_start": "true"},
                    headers=_auth(token),
                )
            assert up.status_code == 200, up.text
            assert up.json()["duration"] == pytest.approx(3.0, abs=0.1)

            # pipeline runs in a thread; poll until done. The patch must stay
            # active for the whole window — the transcribe stage runs after the
            # HTTP response returns.
            import time

            proj = db.get(Project, project_id)
            deadline = time.time() + 60
            while proj.status not in (ProjectStatus.COMPLETED, ProjectStatus.FAILED):
                if time.time() > deadline:
                    break
                time.sleep(0.3)
                db.expire(proj)
                proj = db.get(Project, project_id)

        assert proj.status == ProjectStatus.COMPLETED, proj.error or proj.stage_message
        assert proj.detected_language == "en"
        assert proj.width == 320 and proj.height == 240

        # idempotency: re-running skips completed stages (artifacts exist)
        from app.services.artifacts import ArtifactStore

        store = ArtifactStore(db, proj.id)
        assert store.has("transcript")

        # --- Fase 2: clips must be ranked and persisted
        db.expire_all()
        from app.db.models.clip import Clip

        clips = (
            db.query(Clip)
            .filter(Clip.project_id == project_id)
            .order_by(Clip.rank.asc())
            .all()
        )
        assert len(clips) > 0, "pipeline produced no clips"
        assert all(c.viral_score >= 0.0 for c in clips)
        # ranks are contiguous 1..N
        assert [c.rank for c in clips] == list(range(1, len(clips) + 1))
        # every clip is inside the requested duration band
        for c in clips:
            assert 15.0 <= c.duration <= 60.0, c.duration
            assert c.title and c.transcript
            assert c.end > c.start
        # no two selected clips overlap
        starts = sorted(c.start for c in clips)
        ends_before = [c.end for c in sorted(clips, key=lambda c: c.start)]
        for prev_end, next_start in zip(ends_before, starts[1:]):
            assert next_start >= prev_end - 0.05

        # clips are also reachable through the API
        r = client.get(f"/api/projects/{project_id}/clips", headers=_auth(token))
        assert r.status_code == 200
        assert len(r.json()) == len(clips)

    def test_rejects_bad_extension(self, client, tmp_path):
        token = _register_and_login(client)
        create = client.post(
            "/api/projects",
            json={"source_type": "upload", "source_name": "notes.txt"},
            headers=_auth(token),
        )
        pid = create.json()["id"]
        bad = tmp_path / "notes.txt"
        bad.write_text("not a video")
        with open(bad, "rb") as fh:
            r = client.post(
                f"/api/projects/{pid}/upload",
                files={"file": ("notes.txt", fh, "text/plain")},
                headers=_auth(token),
            )
        assert r.status_code == 400
        assert "unsupported file type" in r.json()["detail"]

    def test_project_isolation(self, client):
        token_a = _register_and_login(client)
        create = client.post(
            "/api/projects",
            json={"source_type": "upload", "source_name": "a.mp4"},
            headers=_auth(token_a),
        )
        pid = create.json()["id"]

        token_b_resp = client.post(
            "/api/auth/register",
            json={"email": "other@example.com", "password": "password123"},
        )
        token_b = token_b_resp.json()["access_token"]
        r = client.get(f"/api/projects/{pid}", headers=_auth(token_b))
        assert r.status_code == 403
