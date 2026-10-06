"""Tests for storage path safety and artifact idempotency."""

import pytest

from app.services.artifacts import ArtifactStore
from app.services.storage import Storage
from app.db.models.project import Project, ProjectStatus, SourceType


def test_safe_name_strips_paths_and_dirs(tmp_path):
    # basename() already removes directory components; sanitizer keeps it safe
    assert Storage._safe_name("../../etc/passwd") == "passwd"
    assert Storage._safe_name("My Video (2024) [final].mp4") == "My_Video__2024___final_.mp4"
    name = Storage._safe_name("a" * 500 + ".mp4")
    assert name.endswith(".mp4") and len(name) <= 129
    # no shell metacharacters survive
    assert ";" not in Storage._safe_name("evil;rm -rf.mp4")


def test_storage_roundtrip(tmp_path):
    st = Storage(tmp_path / "store")
    p = st.project_dir("proj-1")
    rel = st.relpath(p / "source_video.mp4")
    assert rel.startswith("proj-1/")
    assert st.abspath(rel) == (p / "source_video.mp4")


def test_storage_rejects_traversal(tmp_path):
    st = Storage(tmp_path / "store")
    with pytest.raises(ValueError):
        st.abspath("../outside.mp4")


def test_artifact_store_idempotent(db, user):
    project = Project(
        user_id=user.id,
        source_type=SourceType.UPLOAD,
        source_name="t.mp4",
        status=ProjectStatus.PENDING,
    )
    db.add(project)
    db.commit()

    store = ArtifactStore(db, project.id)
    assert not store.has("transcript")

    store.put("transcript", meta={"language": "en", "word_count": 10})
    assert store.has("transcript")
    assert store.get_meta("transcript")["language"] == "en"

    # re-put replaces, does not duplicate
    store.put("transcript", meta={"language": "id"})
    assert store.get_meta("transcript")["language"] == "id"
    rows = store.all_kinds()
    assert rows.count("transcript") == 1
