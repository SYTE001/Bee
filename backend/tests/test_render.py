"""Tests for the final render service and the render pipeline stage."""

from __future__ import annotations

import json
import subprocess
import zipfile
from pathlib import Path

import pytest

from app.core.config import settings
from app.db.models.clip import Clip
from app.pipelines import video_pipeline as vp
from app.services import media, render
from app.services.reframe import CropPlan, CropSample
from app.services.storage import storage

pytestmark = pytest.mark.usefixtures("isolated_storage")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _probe(path: Path) -> media.MediaInfo:
    return media.probe(path)


def _seed_plans(store, project, clip: Clip, plan: CropPlan) -> None:
    """Write a reframe-plans artifact containing one plan for `clip`."""
    from app.services import reframe

    out = storage.project_dir(project.id) / "reframe_plans.json"
    data = {
        clip.id: {
            "_cache_key": reframe.plan_cache_key(
                clip.start, clip.end, clip.layout_mode or "auto"
            ),
            "_cache_version": reframe.PLAN_VERSION,
            "_source_sig": f"{plan.source_width}x{plan.source_height}",
            **plan.to_dict(),
        }
    }
    out.write_text(json.dumps(data), encoding="utf-8")
    store.put(vp.ARTIFACT_REFRAME, str(out), {"plans": 1})


def _bright_rows(video: Path, t: float) -> int:
    """Count rows containing visible (bright) pixels at time `t`."""
    import numpy as np  # type: ignore[import-untyped]

    frame = video.with_suffix(".gray")
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-ss", f"{t}", "-i", str(video),
         "-frames:v", "1", "-pix_fmt", "gray", "-f", "rawvideo", str(frame)],
        check=True,
        capture_output=True,
    )
    arr = np.frombuffer(frame.read_bytes(), dtype=np.uint8).reshape(1920, 1080)
    frame.unlink()
    return int((arr > 120).any(axis=1).sum())


# --------------------------------------------------------------------------
# Filter construction
# --------------------------------------------------------------------------
class TestBuildVideoFilters:
    def _plan(self, layout="single_focus") -> CropPlan:
        return CropPlan(
            layout=layout,
            source_width=1280,
            source_height=720,
            crop_w=405,
            crop_h=720,
            samples=[CropSample(0.0, 100.0, 50.0), CropSample(1.0, 300.0, 120.0)],
        )

    def _filters(self, plan, **kw):
        base = dict(
            ass_path=None, target_fps=30, source_fps=30.0, duration=1.0
        )
        base.update(kw)
        return render.build_video_filters(plan, **base)

    def test_crop_layout_uses_crop_and_scale(self):
        steps = self._filters(self._plan())
        assert steps[0].startswith("crop=405:720")
        assert steps[1] == "scale=1080:1920:flags=lanczos"

    def test_fit_blur_uses_the_blur_chain(self):
        steps = self._filters(self._plan("fit_blur"))
        assert "split=2[bg][fg]" in steps[0]
        assert "overlay=" in steps[-1]

    def test_subtitles_step_escaped(self, tmp_path):
        ass = tmp_path / "s.ass"
        ass.write_text("[Script Info]\n", encoding="utf-8")
        steps = self._filters(self._plan(), ass_path=ass)
        sub = next(s for s in steps if s.startswith("subtitles="))
        # Every drive-letter colon must be escaped; the unescaped remainder
        # must contain no bare colon that would split the filter arguments.
        assert sub.count("\\:") >= 1
        assert ":" not in sub.replace("\\:", "")

    def test_no_zoom_for_static_plans(self):
        assert not any(s.startswith("zoompan") for s in self._filters(self._plan()))

    def test_zoom_added_when_plan_has_one(self):
        zoomed = CropPlan(
            layout="fill_zoom", source_width=1280, source_height=720,
            crop_w=405, crop_h=720, samples=[CropSample(0.0, 100.0, 50.0)],
            zoom_start=1.0, zoom_end=1.25,
        )
        steps = self._filters(zoomed, duration=2.0)
        assert any(
            s.startswith("zoompan=z='1+(0.25)*in_time/2") for s in steps
        )

    def test_hook_title_and_watermark(self):
        steps = self._filters(
            self._plan(), hook_text="My Hook Title", watermark="ClipForge"
        )
        drawtext = [s for s in steps if "drawtext" in s]
        assert len(drawtext) == 2
        assert any("My Hook Title" in s and "between(t,0" in s for s in drawtext)
        assert any("ClipForge" in s for s in drawtext)

    def test_frame_rate_filter(self):
        assert render._frame_rate_filter(60, 30.0) == "fps=fps=60"
        settings.FRAME_INTERPOLATION = True
        try:
            assert render._frame_rate_filter(60, 30.0).startswith("minterpolate")
        finally:
            settings.FRAME_INTERPOLATION = False


# --------------------------------------------------------------------------
# Encoder selection
# --------------------------------------------------------------------------
class TestEncoder:
    def test_resolve_returns_a_known_encoder(self):
        assert render.resolve_encoder() in (
            "libx264", "libx265", "h264_nvenc", "hevc_nvenc",
        )

    def test_nvenc_usable_is_boolean_and_cached(self):
        first = render.nvenc_usable()
        assert isinstance(first, bool)
        assert render.nvenc_usable() is first

    def test_nvenc_args_include_quality_flags(self):
        args = render.encoder_args("h264_nvenc")
        assert "-cq" in args and "-rc" in args

    def test_cpu_args_include_crf(self):
        assert "-crf" in render.encoder_args("libx264")
        assert "-crf" in render.encoder_args("libx265")

    def test_codec_setting_respected(self, monkeypatch):
        monkeypatch.setattr(settings, "VIDEO_CODEC", "h265")
        monkeypatch.setattr(render, "_nvenc_usable", False)
        assert render.resolve_encoder() == "libx265"

    def test_preset_setting_is_passed_through(self, monkeypatch):
        monkeypatch.setattr(settings, "H264_PRESET", "veryfast")
        monkeypatch.setattr(settings, "H265_PRESET", "faster")
        h264 = render.encoder_args("libx264")
        h265 = render.encoder_args("libx265")
        assert h264[h264.index("-preset") + 1] == "veryfast"
        assert h265[h265.index("-preset") + 1] == "faster"
        # CRF must still pair with the configured preset.
        assert h264[h264.index("-crf") + 1] == str(settings.H264_CRF)

    def test_nvenc_ignores_cpu_preset(self, monkeypatch):
        # NVENC keeps its own p1-p7 scale, so the CPU preset must not leak in.
        monkeypatch.setattr(settings, "H264_PRESET", "veryfast")
        args = render.encoder_args("h264_nvenc")
        assert args[args.index("-preset") + 1] == "p5"


# --------------------------------------------------------------------------
# Rendering a clip
# --------------------------------------------------------------------------
class TestRenderClip:
    def test_produces_vertical_video(self, db, project_with_video, make_clip):
        project, store = project_with_video
        clip = make_clip(project, rank=1, start=0.2, end=1.4)
        result = render.render_clip(db, clip, project, store)

        assert result.path.exists()
        info = _probe(result.path)
        assert info.width == 1080 and info.height == 1920
        assert abs(info.duration - 1.2) < 0.2
        assert clip.rendered is True
        assert clip.render_path and storage.abspath(clip.render_path).exists()
        assert clip.thumb_path and storage.abspath(clip.thumb_path).exists()

    def test_renders_audio_track(self, db, project_with_video, make_clip):
        project, store = project_with_video
        clip = make_clip(project, start=0.2, end=1.0)
        result = render.render_clip(db, clip, project, store)
        assert _probe(result.path).has_audio is True

    def test_subtitles_are_burned_in(
        self, db, project_with_black_video, make_clip, monkeypatch
    ):
        # On a black canvas, any bright pixels must come from the subtitles.
        project, store = project_with_black_video
        clip = make_clip(project, start=0.1, end=1.0, subtitle_style="hormozi")
        monkeypatch.setattr(settings, "HOOK_TITLE_SECONDS", 0.0)
        result = render.render_clip(db, clip, project, store)
        assert _bright_rows(result.path, 0.5) > 5

    def test_hook_title_is_drawn(
        self, db, project_with_black_video, make_clip, monkeypatch
    ):
        project, store = project_with_black_video
        clip = make_clip(project, start=0.1, end=1.0)
        monkeypatch.setattr(settings, "WATERMARK_TEXT", "")
        result = render.render_clip(db, clip, project, store)
        # The title overlay shows from t=0; words start at 0.1s.
        assert _bright_rows(result.path, 0.02) > 5

    def test_skips_already_rendered_clip(
        self, db, project_with_video, make_clip, monkeypatch
    ):
        project, store = project_with_video
        clip = make_clip(project, start=0.2, end=1.0)
        first = render.render_clip(db, clip, project, store)

        calls: list[str] = []
        original = media.cut_segment
        monkeypatch.setattr(
            media,
            "cut_segment",
            lambda *a, **k: calls.append(str(a[1] if len(a) > 1 else k.get("out_path")))
            or original(*a, **k),
        )
        second = render.render_clip(db, clip, project, store)
        assert second.path == first.path
        assert calls == []  # no re-encode

    def test_force_rerenders(
        self, db, project_with_video, make_clip, monkeypatch
    ):
        project, store = project_with_video
        clip = make_clip(project, start=0.2, end=1.0)
        render.render_clip(db, clip, project, store)

        calls: list[int] = []
        original = media.cut_segment
        monkeypatch.setattr(
            media,
            "cut_segment",
            lambda *a, **k: calls.append(1) or original(*a, **k),
        )
        render.render_clip(db, clip, project, store, force=True)
        assert calls == [1]

    def test_missing_source_raises(self, db, project_with_video):
        project, store = project_with_video
        store.delete(vp.ARTIFACT_SOURCE)
        clip = Clip(
            project_id=project.id, rank=1, start=0.0, end=0.5,
            duration=0.5, viral_score=1.0,
        )
        db.add(clip)
        db.commit()
        with pytest.raises(RuntimeError, match="source artifact missing"):
            render.render_clip(db, clip, project, store)

    def test_render_honours_configured_preset(
        self, db, project_with_video, make_clip, monkeypatch
    ):
        project, store = project_with_video
        clip = make_clip(project, start=0.1, end=0.9)

        cmds: list[list[str]] = []
        original = media.cut_segment

        def _spy(source, out_path, start, end, **kw):
            cmds.append(kw.get("extra_args") or [])
            return original(source, out_path, start, end, **kw)

        monkeypatch.setattr(media, "cut_segment", _spy)
        monkeypatch.setattr(settings, "H264_PRESET", "ultrafast")
        try:
            render.render_clip(db, clip, project, store)
        finally:
            media.cut_segment = original
        # The extra_args carry the configured preset, and cut_segment must not
        # append its own (later flags win in ffmpeg, which would override it).
        assert ["-preset", "ultrafast"] in [cmds[0][i:i + 2] for i in range(len(cmds[0]))]


# --------------------------------------------------------------------------
# Plan resolution (cache reuse vs rebuild)
# --------------------------------------------------------------------------
class TestResolvePlan:
    def test_cached_plan_reused(self, db, project_with_video, make_clip):
        project, store = project_with_video
        clip = make_clip(project, start=0.2, end=1.2, layout_mode="fit_blur")
        plan = CropPlan(
            layout="fit_blur", source_width=320, source_height=240,
            crop_w=135, crop_h=240, samples=[CropSample(0.0, 0.0, 0.0)],
        )
        _seed_plans(store, project, clip, plan)
        info = _probe(store.path_of(vp.ARTIFACT_SOURCE))
        assert render.resolve_plan(db, clip, project, store, info) == plan

    def test_stale_dimensions_trigger_rebuild(
        self, db, project_with_video, make_clip, monkeypatch
    ):
        from app.services import reframe

        project, store = project_with_video
        clip = make_clip(project, start=0.2, end=1.0, layout_mode="fit_blur")
        plan = CropPlan(
            layout="fit_blur", source_width=999, source_height=999,
            crop_w=100, crop_h=100, samples=[CropSample(0.0, 0.0, 0.0)],
        )
        _seed_plans(store, project, clip, plan)

        rebuilt = CropPlan(
            layout="fit_blur", source_width=320, source_height=240,
            crop_w=135, crop_h=240, samples=[CropSample(0.0, 5.0, 5.0)],
        )
        monkeypatch.setattr(reframe, "build_crop_plan", lambda frames, **kw: rebuilt)
        monkeypatch.setattr(reframe, "analyze_faces", lambda src, **kw: ([], "none"))
        info = _probe(store.path_of(vp.ARTIFACT_SOURCE))
        assert render.resolve_plan(db, clip, project, store, info) is rebuilt

    def test_plan_from_older_schema_is_rebuilt(
        self, db, project_with_video, make_clip, monkeypatch
    ):
        """A cached plan written by an older build must not be replayed."""
        from app.services import reframe

        project, store = project_with_video
        clip = make_clip(project, start=0.2, end=1.0, layout_mode="fit_blur")
        plan = CropPlan(
            layout="fit_blur", source_width=320, source_height=240,
            crop_w=135, crop_h=240, samples=[CropSample(0.0, 0.0, 0.0)],
        )
        _seed_plans(store, project, clip, plan)
        # Strip the version marker, as if written by a build before it existed.
        out = storage.project_dir(project.id) / "reframe_plans.json"
        stale = json.loads(out.read_text(encoding="utf-8"))
        del stale[clip.id]["_cache_version"]
        out.write_text(json.dumps(stale), encoding="utf-8")

        rebuilt = CropPlan(
            layout="fit_blur", source_width=320, source_height=240,
            crop_w=135, crop_h=240, samples=[CropSample(0.0, 5.0, 5.0)],
        )
        monkeypatch.setattr(reframe, "build_crop_plan", lambda frames, **kw: rebuilt)
        monkeypatch.setattr(reframe, "analyze_faces", lambda src, **kw: ([], "none"))
        info = _probe(store.path_of(vp.ARTIFACT_SOURCE))
        assert render.resolve_plan(db, clip, project, store, info) is rebuilt
        # The rebuild is written back to the file the reader uses, so the next
        # render does not pay for the analysis again.
        persisted = json.loads(
            storage.project_dir(project.id).joinpath("reframe_plans.json")
            .read_text(encoding="utf-8")
        )
        assert persisted[clip.id]["_cache_version"] == reframe.PLAN_VERSION
        assert persisted[clip.id]["crop_w"] == 135.0

    def test_rebuilt_plan_is_served_on_the_next_resolve(
        self, db, project_with_video, make_clip, monkeypatch
    ):
        """The persisted rebuild is reused without re-running the analysis."""
        from app.services import reframe

        project, store = project_with_video
        clip = make_clip(project, start=0.2, end=1.0, layout_mode="fit_blur")
        rebuilt = CropPlan(
            layout="fit_blur", source_width=320, source_height=240,
            crop_w=135, crop_h=240, samples=[CropSample(0.0, 5.0, 5.0)],
        )
        monkeypatch.setattr(reframe, "build_crop_plan", lambda frames, **kw: rebuilt)
        monkeypatch.setattr(reframe, "analyze_faces", lambda src, **kw: ([], "none"))
        info = _probe(store.path_of(vp.ARTIFACT_SOURCE))
        first = render.resolve_plan(db, clip, project, store, info)

        calls = []
        monkeypatch.setattr(
            reframe, "build_crop_plan",
            lambda frames, **kw: calls.append(frames) or rebuilt,
        )
        second = render.resolve_plan(db, clip, project, store, info)
        # Round-tripped through the store, so it is an equal plan rather than
        # the same object — but it must not have been rebuilt.
        assert second == first
        assert calls == []


# --------------------------------------------------------------------------
# Batch rendering
# --------------------------------------------------------------------------
class TestRenderProjectClips:
    def test_renders_every_clip(self, db, project_with_video, make_clip):
        project, store = project_with_video
        for i, (s, e) in enumerate(((0.1, 0.9), (1.0, 1.9))):
            make_clip(project, rank=i + 1, start=s, end=e)
        rendered, failed = render.render_project_clips(db, project, store)
        assert rendered == 2 and failed == 0
        rows = db.query(Clip).filter(Clip.project_id == project.id).all()
        assert all(c.rendered for c in rows)

    def test_one_bad_clip_does_not_stop_the_batch(
        self, db, project_with_video, make_clip, monkeypatch
    ):
        project, store = project_with_video
        good = make_clip(project, rank=1, start=0.1, end=0.9)
        bad = make_clip(project, rank=2, start=0.1, end=0.9)

        original = render.render_clip

        def _fail(db, clip, project, store, **kw):
            if clip.id == bad.id:
                raise RuntimeError("boom")
            return original(db, clip, project, store, **kw)

        monkeypatch.setattr(render, "render_clip", _fail)
        rendered, failed = render.render_project_clips(db, project, store)
        assert rendered == 1 and failed == 1
        assert bad.rendered is False and bad.render_error == "boom"
        assert good.rendered is True

    def test_no_clips(self, db, project_with_video):
        project, store = project_with_video
        assert render.render_project_clips(db, project, store) == (0, 0)

    def test_parallel_batch_renders_every_clip(
        self, db, project_with_video, make_clip, monkeypatch
    ):
        project, store = project_with_video
        for i, (s, e) in enumerate(((0.1, 0.9), (1.0, 1.9), (1.5, 1.9))):
            make_clip(project, rank=i + 1, start=s, end=e)
        monkeypatch.setattr(render, "_render_concurrency", lambda n, enc=None: 2)

        rendered, failed = render.render_project_clips(db, project, store)

        assert rendered == 3 and failed == 0
        rows = db.query(Clip).filter(Clip.project_id == project.id).all()
        assert all(c.rendered and c.render_path for c in rows)
        assert all(c.thumb_path for c in rows)

    def test_parallel_batch_survives_one_failure(
        self, db, project_with_video, make_clip, monkeypatch
    ):
        project, store = project_with_video
        good = make_clip(project, rank=1, start=0.1, end=0.9)
        bad = make_clip(project, rank=2, start=0.1, end=0.9)

        original = render.render_clip

        def _fail(session, clip, project, store, **kw):
            if clip.id == bad.id:
                raise RuntimeError("boom")
            return original(session, clip, project, store, **kw)

        monkeypatch.setattr(render, "render_clip", _fail)
        monkeypatch.setattr(render, "_render_concurrency", lambda n, enc=None: 2)

        rendered, failed = render.render_project_clips(db, project, store)
        assert rendered == 1 and failed == 1
        # the caller's objects must reflect what the worker sessions committed
        assert bad.rendered is False and bad.render_error == "boom"
        assert good.rendered is True


# --------------------------------------------------------------------------
# Render concurrency
# --------------------------------------------------------------------------
class TestRenderConcurrency:
    def test_single_clip_is_never_parallel(self):
        assert render._render_concurrency(1, "libx264") == 1

    def test_caps_at_half_the_cpus(self, monkeypatch):
        monkeypatch.setattr(settings, "WORKER_CONCURRENCY", 64)
        monkeypatch.setattr(render.os, "cpu_count", lambda: 8)
        assert render._render_concurrency(6, "libx264") == 4

    def test_respects_worker_concurrency(self, monkeypatch):
        monkeypatch.setattr(settings, "WORKER_CONCURRENCY", 2)
        monkeypatch.setattr(render.os, "cpu_count", lambda: 32)
        assert render._render_concurrency(10, "libx264") == 2

    def test_never_exceeds_clip_count(self, monkeypatch):
        monkeypatch.setattr(settings, "WORKER_CONCURRENCY", 8)
        monkeypatch.setattr(render.os, "cpu_count", lambda: 32)
        assert render._render_concurrency(2, "libx264") == 2

    def test_nvenc_cap_applies(self, monkeypatch):
        monkeypatch.setattr(settings, "WORKER_CONCURRENCY", 8)
        monkeypatch.setattr(render.os, "cpu_count", lambda: 32)
        monkeypatch.setattr(settings, "NVENC_MAX_PARALLEL", 2)
        assert render._render_concurrency(10, "h264_nvenc") == 2

    def test_nvenc_cap_does_not_apply_to_cpu_encoders(self, monkeypatch):
        monkeypatch.setattr(settings, "WORKER_CONCURRENCY", 8)
        monkeypatch.setattr(render.os, "cpu_count", lambda: 32)
        monkeypatch.setattr(settings, "NVENC_MAX_PARALLEL", 1)
        assert render._render_concurrency(10, "libx264") == 8


# --------------------------------------------------------------------------
# ZIP export
# --------------------------------------------------------------------------
class TestZipExport:
    def test_zip_contains_clips_and_index(
        self, db, project_with_video, make_clip
    ):
        project, store = project_with_video
        make_clip(project, rank=1, start=0.1, end=0.9, title="First Clip")
        make_clip(project, rank=2, start=1.0, end=1.9, title="Second Clip")
        render.render_project_clips(db, project, store)

        out = render.zip_rendered_clips(db, project)
        assert out.exists() and out.stat().st_size > 0
        with zipfile.ZipFile(out) as zf:
            names = zf.namelist()
            assert any(n.startswith("01_First_Clip") for n in names)
            assert any(n.startswith("02_Second_Clip") for n in names)
            assert "clips.txt" in names
            assert "First Clip" in zf.read("clips.txt").decode("utf-8")

    def test_duplicate_titles_do_not_clobber(
        self, db, project_with_video, make_clip
    ):
        project, store = project_with_video
        for i in range(2):
            make_clip(project, rank=1, start=0.1 * (i + 1), end=0.9, title="Same")
        render.render_project_clips(db, project, store)

        with zipfile.ZipFile(render.zip_rendered_clips(db, project)) as zf:
            mp4s = [n for n in zf.namelist() if n.endswith(".mp4")]
            assert len(mp4s) == 2

    def test_unrendered_clips_listed_but_skipped(
        self, db, project_with_video, make_clip
    ):
        project, store = project_with_video
        make_clip(project, rank=1, start=0.1, end=0.9)
        with pytest.raises(RuntimeError, match="no rendered clips"):
            render.zip_rendered_clips(db, project)

    def test_no_clips_raises(self, db, project_with_video):
        project, store = project_with_video
        with pytest.raises(RuntimeError, match="no clips"):
            render.zip_rendered_clips(db, project)


# --------------------------------------------------------------------------
# Pipeline render stage
# --------------------------------------------------------------------------
class TestStageRender:
    def test_renders_pending_clips(self, db, project_with_video, make_clip):
        project, store = project_with_video
        make_clip(project, rank=1, start=0.1, end=0.9)
        vp.stage_render(db, project, store)
        clip = db.query(Clip).filter(Clip.project_id == project.id).one()
        assert clip.rendered is True
        assert project.status == vp.ProjectStatus.RENDERING

    def test_noop_when_all_rendered(
        self, db, project_with_video, make_clip, monkeypatch
    ):
        project, store = project_with_video
        clip = make_clip(project, rank=1, start=0.1, end=0.9)
        clip.rendered = True
        db.commit()

        def _should_not_run(*a, **k):
            raise AssertionError("render should have been skipped")

        monkeypatch.setattr(render, "render_project_clips", _should_not_run)
        vp.stage_render(db, project, store)

    def test_stage_list_ends_with_render(self):
        names = [s.__name__ for s in vp._stage_list()]
        assert names[-1] == "stage_render"
        assert names.index("stage_reframe") < names.index("stage_render")
