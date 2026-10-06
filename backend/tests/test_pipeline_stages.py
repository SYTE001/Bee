"""Tests for the Fase 3 pipeline stages (visual analysis + reframing).

These exercise the stages directly with a real (tiny) generated video, so
they cover the ffmpeg plumbing the unit tests stub out.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from app.db.models.clip import Clip
from app.db.models.project import Project, ProjectStatus, SourceType
from app.db.models.user import User
from app.pipelines import video_pipeline as vp
from app.services import visual_analysis
from app.services.storage import storage


# --------------------------------------------------------------------------
# stage_analyze_visual
# --------------------------------------------------------------------------
class TestStageAnalyzeVisual:
    def test_produces_artifact(self, db, project_with_video):
        proj, store = project_with_video
        vp.stage_analyze_visual(db, proj, store)

        assert store.has(vp.ARTIFACT_VISUAL)
        meta = store.get_meta(vp.ARTIFACT_VISUAL)
        assert "samples" in meta and "cuts" in meta

    def test_writes_loadable_features(self, db, project_with_video):
        proj, store = project_with_video
        vp.stage_analyze_visual(db, proj, store)

        path = store.path_of(vp.ARTIFACT_VISUAL)
        feats = visual_analysis.load_features(path)  # type: ignore[arg-type]
        assert feats.duration > 0
        assert len(feats.times) > 0

    def test_is_idempotent(self, db, project_with_video):
        proj, store = project_with_video
        vp.stage_analyze_visual(db, proj, store)
        first = store.path_of(vp.ARTIFACT_VISUAL)
        # Second run must not fail and must not change the artifact.
        vp.stage_analyze_visual(db, proj, store)
        assert store.path_of(vp.ARTIFACT_VISUAL) == first

    def test_missing_source_raises(self, db, project_with_video):
        proj, store = project_with_video
        store.delete(vp.ARTIFACT_SOURCE)
        with pytest.raises(RuntimeError, match="source artifact missing"):
            vp.stage_analyze_visual(db, proj, store)


# --------------------------------------------------------------------------
# stage_reframe
# --------------------------------------------------------------------------
class TestStageReframe:
    def _clips(self, db, project_with_video):
        proj, store = project_with_video
        for i, (s, e) in enumerate(((0.0, 1.0), (1.0, 2.0))):
            db.add(
                Clip(
                    project_id=proj.id,
                    rank=i + 1,
                    start=s,
                    end=e,
                    duration=e - s,
                    viral_score=50.0,
                    layout_mode="fit_blur",
                )
            )
        db.commit()
        return proj, store

    def test_builds_plan_per_clip(self, db, project_with_video):
        proj, store = self._clips(db, project_with_video)
        vp.stage_reframe(db, proj, store)

        assert store.has(vp.ARTIFACT_REFRAME)
        plans = json.loads(
            Path(str(store.path_of(vp.ARTIFACT_REFRAME))).read_text(encoding="utf-8")  # type: ignore[arg-type]
        )
        assert len(plans) == 2
        for plan in plans.values():
            assert plan["layout"] == "fit_blur"
            assert plan["crop_w"] > 0 and plan["crop_h"] > 0
            assert len(plan["samples"]) >= 1

    def test_no_clips_is_a_noop(self, db, project_with_video):
        proj, store = project_with_video
        vp.stage_reframe(db, proj, store)
        assert not store.has(vp.ARTIFACT_REFRAME)

    def test_missing_source_raises(self, db, project_with_video):
        proj, store = self._clips(db, project_with_video)
        store.delete(vp.ARTIFACT_SOURCE)
        with pytest.raises(RuntimeError, match="source artifact missing"):
            vp.stage_reframe(db, proj, store)

    def test_reuses_cached_plan_for_same_range(self, db, project_with_video):
        """A re-score mints new clip ids but the same range hits the cache."""
        proj, store = self._clips(db, project_with_video)
        vp.stage_reframe(db, proj, store)
        first = json.loads(
            Path(str(store.path_of(vp.ARTIFACT_REFRAME))).read_text(encoding="utf-8")  # type: ignore[arg-type]
        )
        old_ids = set(first)

        # Simulate a re-score: delete and recreate the clips at the same times.
        db.query(Clip).filter(Clip.project_id == proj.id).delete()
        db.commit()
        for i, (s, e) in enumerate(((0.0, 1.0), (1.0, 2.0))):
            db.add(
                Clip(
                    project_id=proj.id, rank=i + 1, start=s, end=e,
                    duration=e - s, viral_score=50.0, layout_mode="fit_blur",
                )
            )
        db.commit()
        store.delete(vp.ARTIFACT_REFRAME)
        # Re-seed the artifact store from the on-disk JSON so the cache lookup
        # sees the previous plans, mirroring a real resume.
        plans_path = storage.project_dir(proj.id) / "reframe_plans.json"
        store.put(vp.ARTIFACT_REFRAME, str(plans_path), {"plans": len(first)})

        vp.stage_reframe(db, proj, store)
        second = json.loads(
            Path(str(store.path_of(vp.ARTIFACT_REFRAME))).read_text(encoding="utf-8")  # type: ignore[arg-type]
        )
        # New clip ids, but the plans themselves were reused verbatim.
        assert set(second) != old_ids
        reused = {v["crop_w"] for v in second.values()}
        original = {v["crop_w"] for v in first.values()}
        assert reused == original

    def test_audio_only_project_skips_reframe(self, db, project_with_video, monkeypatch):
        proj, store = self._clips(db, project_with_video)

        class _FakeInfo:
            has_video = False
            width = 0
            height = 0

        monkeypatch.setattr(vp.media, "probe", lambda src: _FakeInfo())
        # Should log and return without raising.
        vp.stage_reframe(db, proj, store)
        assert not store.has(vp.ARTIFACT_REFRAME)


# --------------------------------------------------------------------------
# Stage ordering
# --------------------------------------------------------------------------
def test_stages_include_new_fase3_stages():
    names = [s.__name__ for s in vp._stage_list()]
    assert names.index("stage_analyze_visual") < names.index("stage_score_and_select")
    assert names.index("stage_score_and_select") < names.index("stage_reframe")


def test_artifact_keys_are_unique():
    keys = {
        vp.ARTIFACT_SOURCE, vp.ARTIFACT_AUDIO, vp.ARTIFACT_AUDIO_NORM,
        vp.ARTIFACT_TRANSCRIPT, vp.ARTIFACT_AUDIO_FEATURES,
        vp.ARTIFACT_DIARIZATION, vp.ARTIFACT_LLM_ANALYSIS,
        vp.ARTIFACT_VISUAL, vp.ARTIFACT_CLIPS, vp.ARTIFACT_REFRAME,
    }
    assert len(keys) == 10
