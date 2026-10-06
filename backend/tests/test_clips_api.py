"""Tests for the clip API: styles, the editor, render triggers, ZIP export."""

from __future__ import annotations

import io
import json
import zipfile

import pytest

from app.db.models.clip import Clip
from app.db.models.project import Project, ProjectStatus
from app.db.models.user import User
from app.core.security import create_access_token

pytest.importorskip("fastapi.testclient")
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.db.session import get_db  # noqa: E402


@pytest.fixture()
def client(db, isolated_storage):
    """A TestClient whose DB session is the test session.

    Overriding the `get_db` dependency keeps reads/writes in the isolated
    sqlite database so the assertions below see the same rows.
    """
    def _override():
        try:
            yield db
        finally:
            pass

    app.dependency_overrides[get_db] = _override
    try:
        with TestClient(app) as c:
            yield c
    finally:
        app.dependency_overrides.pop(get_db, None)


def _auth_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _seed_project(client, token: str, db, make_clip, *, clips=2) -> str:
    """Create a project through the API and add clip rows directly."""
    r = client.post(
        "/api/projects",
        json={
            "source_type": "upload",
            "source_name": "podcast.mp4",
            "target_clip_count": clips,
        },
        headers=_auth_headers(token),
    )
    assert r.status_code == 201, r.text
    project_id = r.json()["id"]
    for i in range(clips):
        make_clip(
            Project(id=project_id),
            rank=i + 1,
            start=0.1 + i,
            end=0.9 + i,
            title=f"Clip number {i + 1}",
        )
    return project_id


# --------------------------------------------------------------------------
# Subtitle styles
# --------------------------------------------------------------------------
def test_subtitle_styles_endpoint(client):
    r = client.get("/api/subtitle-styles")
    assert r.status_code == 200
    styles = r.json()
    assert len(styles) >= 15
    ids = {s["id"] for s in styles}
    assert "classic_bold_white" in ids
    first = styles[0]
    for key in ("id", "name", "font", "font_size", "primary_color", "position"):
        assert key in first


# --------------------------------------------------------------------------
# Clip editor (PATCH)
# --------------------------------------------------------------------------
class TestUpdateClip:
    def test_rename_marks_rerender(self, client, auth_token, db, make_clip):
        project_id = _seed_project(client, auth_token, db, make_clip)
        clip = db.query(Clip).filter(Clip.project_id == project_id).first()
        clip.rendered = True
        db.commit()

        r = client.patch(
            f"/api/projects/{project_id}/clips/{clip.id}",
            json={"title": "Better hook title"},
            headers=_auth_headers(auth_token),
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["title"] == "Better hook title"
        db.refresh(clip)
        assert clip.rendered is False

    def test_trim_adjusts_duration(self, client, auth_token, db, make_clip):
        project_id = _seed_project(client, auth_token, db, make_clip)
        clip = db.query(Clip).filter(Clip.project_id == project_id).first()

        r = client.patch(
            f"/api/projects/{project_id}/clips/{clip.id}",
            json={"start": 0.5, "end": 2.5},
            headers=_auth_headers(auth_token),
        )
        assert r.status_code == 200, r.text
        db.refresh(clip)
        assert clip.start == 0.5 and clip.end == 2.5
        assert clip.duration == pytest.approx(2.0, abs=0.01)

    def test_rejects_inverted_range(self, client, auth_token, db, make_clip):
        project_id = _seed_project(client, auth_token, db, make_clip)
        clip = db.query(Clip).filter(Clip.project_id == project_id).first()
        r = client.patch(
            f"/api/projects/{project_id}/clips/{clip.id}",
            json={"start": 2.0, "end": 1.0},
            headers=_auth_headers(auth_token),
        )
        assert r.status_code == 400

    def test_rejects_unknown_style(self, client, auth_token, db, make_clip):
        project_id = _seed_project(client, auth_token, db, make_clip)
        clip = db.query(Clip).filter(Clip.project_id == project_id).first()
        r = client.patch(
            f"/api/projects/{project_id}/clips/{clip.id}",
            json={"subtitle_style": "not_a_style"},
            headers=_auth_headers(auth_token),
        )
        assert r.status_code == 400

    def test_rejects_unknown_layout(self, client, auth_token, db, make_clip):
        project_id = _seed_project(client, auth_token, db, make_clip)
        clip = db.query(Clip).filter(Clip.project_id == project_id).first()
        r = client.patch(
            f"/api/projects/{project_id}/clips/{clip.id}",
            json={"layout_mode": "widescreen"},
            headers=_auth_headers(auth_token),
        )
        assert r.status_code == 400

    def test_layout_change_marks_rerender(self, client, auth_token, db, make_clip):
        project_id = _seed_project(client, auth_token, db, make_clip)
        clip = db.query(Clip).filter(Clip.project_id == project_id).first()
        clip.rendered = True
        db.commit()
        r = client.patch(
            f"/api/projects/{project_id}/clips/{clip.id}",
            json={"layout_mode": "fill_zoom", "target_fps": 30},
            headers=_auth_headers(auth_token),
        )
        assert r.status_code == 200, r.text
        db.refresh(clip)
        assert clip.layout_mode == "fill_zoom"
        assert clip.target_fps == 30
        assert clip.rendered is False

    def test_replacing_words_updates_transcript(
        self, client, auth_token, db, make_clip
    ):
        project_id = _seed_project(client, auth_token, db, make_clip)
        clip = db.query(Clip).filter(Clip.project_id == project_id).first()
        r = client.patch(
            f"/api/projects/{project_id}/clips/{clip.id}",
            json={
                "words": [
                    {"word": "edited", "start": 0.0, "end": 0.5},
                    {"word": "text", "start": 0.5, "end": 1.0},
                ]
            },
            headers=_auth_headers(auth_token),
        )
        assert r.status_code == 200, r.text
        db.refresh(clip)
        assert clip.transcript == "edited text"
        assert len(clip.words) == 2

    def test_empty_words_rejected(self, client, auth_token, db, make_clip):
        project_id = _seed_project(client, auth_token, db, make_clip)
        clip = db.query(Clip).filter(Clip.project_id == project_id).first()
        r = client.patch(
            f"/api/projects/{project_id}/clips/{clip.id}",
            json={"words": []},
            headers=_auth_headers(auth_token),
        )
        assert r.status_code == 400

    def test_other_users_clip_forbidden(self, client, db, make_clip, user, auth_token):
        other = User(email="other-clips@example.com", hashed_password="x")
        db.add(other)
        db.commit()

        project_id = _seed_project(client, auth_token, db, make_clip)
        clip = db.query(Clip).filter(Clip.project_id == project_id).first()

        other_token = create_access_token(other.id, extra={"email": other.email})
        r = client.patch(
            f"/api/projects/{project_id}/clips/{clip.id}",
            json={"title": "hijacked"},
            headers=_auth_headers(other_token),
        )
        assert r.status_code == 403

    def test_missing_clip_404(self, client, auth_token, db, make_clip):
        project_id = _seed_project(client, auth_token, db, make_clip)
        r = client.patch(
            f"/api/projects/{project_id}/clips/does-not-exist",
            json={"title": "x"},
            headers=_auth_headers(auth_token),
        )
        assert r.status_code == 404


# --------------------------------------------------------------------------
# Render triggers
# --------------------------------------------------------------------------
class TestRenderEndpoints:
    def test_render_one_clip_accepted(self, client, auth_token, db, make_clip):
        project_id = _seed_project(client, auth_token, db, make_clip, clips=1)
        clip = db.query(Clip).filter(Clip.project_id == project_id).first()
        r = client.post(
            f"/api/projects/{project_id}/clips/{clip.id}/render",
            headers=_auth_headers(auth_token),
        )
        assert r.status_code == 202
        assert "task_id" in r.json()

    def test_render_all_requires_completed_status(
        self, client, auth_token, db, make_clip
    ):
        project_id = _seed_project(client, auth_token, db, make_clip)
        # project is still pending -> 409
        r = client.post(
            f"/api/projects/{project_id}/render",
            headers=_auth_headers(auth_token),
        )
        assert r.status_code == 409

    def test_render_all_after_completion(self, client, auth_token, db, make_clip):
        project_id = _seed_project(client, auth_token, db, make_clip)
        project = db.get(Project, project_id)
        project.status = ProjectStatus.COMPLETED
        db.commit()

        r = client.post(
            f"/api/projects/{project_id}/render",
            headers=_auth_headers(auth_token),
        )
        assert r.status_code == 202
        assert "task_id" in r.json()


# --------------------------------------------------------------------------
# ZIP download
# --------------------------------------------------------------------------
class TestZipDownload:
    def test_zip_without_renders_is_404(self, client, auth_token, db, make_clip):
        project_id = _seed_project(client, auth_token, db, make_clip)
        r = client.get(
            f"/api/projects/{project_id}/download.zip",
            headers=_auth_headers(auth_token),
        )
        assert r.status_code == 404

    def test_download_zip(self, client, auth_token, db, make_clip):
        project_id = _seed_project(client, auth_token, db, make_clip)
        clip = db.query(Clip).filter(Clip.project_id == project_id).first()
        clip.rendered = True
        clip.render_path = f"{project_id}/clips/clip_01_x.mp4"
        # Fake a rendered file where the storage layer expects it.
        from app.services.storage import storage

        clips_dir = storage.clips_dir(project_id)
        (clips_dir / "clip_01_x.mp4").write_bytes(b"not really an mp4")
        db.commit()

        r = client.get(
            f"/api/projects/{project_id}/download.zip",
            headers=_auth_headers(auth_token),
        )
        assert r.status_code == 200
        assert r.headers["content-type"] == "application/zip"
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            names = zf.namelist()
            # Entries are named by rank + title, not by the render filename.
            assert any(n.startswith("01_Clip_number_1.mp4") for n in names)
            assert "clips.txt" in names

    def test_other_user_cannot_download(self, client, auth_token, db, make_clip):
        other = User(email="other-zip@example.com", hashed_password="x")
        db.add(other)
        db.commit()

        project_id = _seed_project(client, auth_token, db, make_clip)
        other_token = create_access_token(other.id, extra={"email": other.email})
        r = client.get(
            f"/api/projects/{project_id}/download.zip",
            headers=_auth_headers(other_token),
        )
        assert r.status_code == 403


# --------------------------------------------------------------------------
# Media access (?token= for <img>/<video>/navigation)
# --------------------------------------------------------------------------
class TestMediaAccess:
    def _clip_with_media(self, client, auth_token, db, make_clip):
        project_id = _seed_project(client, auth_token, db, make_clip, clips=1)
        clip = db.query(Clip).filter(Clip.project_id == project_id).first()
        clip.rendered = True
        clip.render_path = f"{project_id}/clips/clip_01_x.mp4"
        clip.thumb_path = f"{project_id}/clips/thumb_01_x.jpg"
        from app.services.storage import storage

        clips_dir = storage.clips_dir(project_id)
        (clips_dir / "clip_01_x.mp4").write_bytes(b"not really an mp4")
        (clips_dir / "thumb_01_x.jpg").write_bytes(b"not really a jpg")
        db.commit()
        return project_id, clip

    def test_video_served_via_query_token(self, client, auth_token, db, make_clip):
        project_id, clip = self._clip_with_media(client, auth_token, db, make_clip)
        r = client.get(
            f"/api/projects/{project_id}/clips/{clip.id}/video?token={auth_token}"
        )
        assert r.status_code == 200
        assert r.headers["content-type"] == "video/mp4"

    def test_project_response_exposes_media_urls(
        self, client, auth_token, db, make_clip
    ):
        # The project endpoint (used by the clip grid) must carry the URLs too.
        project_id, clip = self._clip_with_media(client, auth_token, db, make_clip)
        r = client.get(
            f"/api/projects/{project_id}", headers=_auth_headers(auth_token)
        )
        assert r.status_code == 200
        body = r.json()
        got = {c["id"]: c for c in body["clips"]}
        assert got[clip.id]["video_url"].endswith(f"/clips/{clip.id}/video")
        assert got[clip.id]["thumb_url"].endswith(f"/clips/{clip.id}/thumb")
        # internal storage paths must not leak
        assert "render_path" not in got[clip.id]
        assert "thumb_path" not in got[clip.id]

    def test_thumb_served_via_query_token(self, client, auth_token, db, make_clip):
        project_id, clip = self._clip_with_media(client, auth_token, db, make_clip)
        r = client.get(
            f"/api/projects/{project_id}/clips/{clip.id}/thumb?token={auth_token}"
        )
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/jpeg"

    def test_zip_served_via_query_token(self, client, auth_token, db, make_clip):
        project_id, _ = self._clip_with_media(client, auth_token, db, make_clip)
        r = client.get(f"/api/projects/{project_id}/download.zip?token={auth_token}")
        assert r.status_code == 200
        assert r.headers["content-type"] == "application/zip"

    def test_media_without_token_is_401(self, client, auth_token, db, make_clip):
        project_id, clip = self._clip_with_media(client, auth_token, db, make_clip)
        r = client.get(f"/api/projects/{project_id}/clips/{clip.id}/thumb")
        assert r.status_code == 401

    def test_media_404_when_not_rendered(self, client, auth_token, db, make_clip):
        project_id = _seed_project(client, auth_token, db, make_clip, clips=1)
        clip = db.query(Clip).filter(Clip.project_id == project_id).first()
        r = client.get(
            f"/api/projects/{project_id}/clips/{clip.id}/video?token={auth_token}"
        )
        assert r.status_code == 404
