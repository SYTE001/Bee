"""Tests for visual signal analysis (scene cuts, face presence, motion)."""

from __future__ import annotations

import pytest

from app.services import visual_analysis


def synth_video(path, duration: float = 4.0, fps: int = 15, cuts: int = 2):
    """A test video with hard cuts every `duration/cuts` seconds."""
    import subprocess

    segs = []
    colors = ["red", "green", "blue", "yellow"]
    for i in range(cuts):
        seg = path.with_suffix(f".seg{i}.mp4")
        c = colors[i % len(colors)]
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
             "-i", f"color=c={c}:s=320x180:d={duration / cuts}:r={fps}",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", str(seg)],
            check=True, capture_output=True,
        )
        segs.append(str(seg))
    list_file = path.with_suffix(".list.txt")
    list_file.write_text("".join(f"file '{s}'\n" for s in segs), encoding="utf-8")
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
         "-i", str(list_file), "-c", "copy", str(path)],
        check=True, capture_output=True,
    )
    return str(path)


@pytest.fixture(scope="module")
def cut_video(tmp_path_factory):
    d = tmp_path_factory.mktemp("visual")
    return synth_video(d / "cuts.mp4", duration=4.0, fps=15, cuts=4)


# --------------------------------------------------------------------------
# Feature extraction
# --------------------------------------------------------------------------
class TestAnalyzeVideo:
    def test_runs_on_real_video(self, cut_video):
        feats = visual_analysis.analyze_video(cut_video, fps=5)
        assert feats.duration == pytest.approx(4.0, abs=0.2)
        assert len(feats.times) > 0
        assert len(feats.face_area) == len(feats.times)
        assert len(feats.motion) == len(feats.times)

    def test_detects_hard_cuts(self, cut_video):
        feats = visual_analysis.analyze_video(cut_video, fps=10)
        # 4 solid-colour segments = 3 transitions.
        assert len(feats.cuts) >= 2, f"expected >=2 cuts, got {feats.cuts}"
        # Cuts should be spread across the timeline, not all at the start.
        assert max(feats.cuts) > 1.0

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(Exception):
            visual_analysis.analyze_video(tmp_path / "nope.mp4")


class TestDetectCuts:
    def test_empty_motion(self):
        assert visual_analysis._detect_cuts([], []) == []

    def test_flat_motion_has_no_cuts(self):
        times = [i * 0.2 for i in range(10)]
        motion = [0.01] * 10
        assert visual_analysis._detect_cuts(times, motion) == []

    def test_spike_marks_a_cut(self):
        times = [0.0, 0.2, 0.4, 0.6]
        motion = [0.01, 0.02, 0.60, 0.02]
        cuts = visual_analysis._detect_cuts(times, motion)
        assert cuts == [pytest.approx(0.4, abs=1e-6)]

    def test_spikes_merge_when_close(self):
        times = [0.0, 0.2, 0.3, 0.5]
        motion = [0.01, 0.60, 0.55, 0.02]
        cuts = visual_analysis._detect_cuts(times, motion)
        # 0.2 and 0.3 are within 0.4s of each other, so they merge into one.
        assert len(cuts) == 1

    def test_ignores_ramp_up(self):
        # A slow fade must not be mistaken for a hard cut.
        times = [i * 0.2 for i in range(8)]
        motion = [0.01, 0.03, 0.06, 0.10, 0.15, 0.20, 0.25, 0.30]
        assert visual_analysis._detect_cuts(times, motion) == []


# --------------------------------------------------------------------------
# Clip rollup + scoring
# --------------------------------------------------------------------------
class TestClipStats:
    def _feats(self):
        return visual_analysis.VisualFeatures(
            duration=10.0,
            cuts=[1.0, 5.0, 9.0],
            face_area=[0.2, 0.3, 0.25],
            motion=[0.05, 0.06, 0.05],
            times=[0.0, 0.2, 0.4],
        )

    def test_counts_cuts_in_window(self):
        feats = self._feats()
        stats = visual_analysis.clip_stats(feats, 0.0, 10.0)
        assert stats.cuts_in_clip == 3
        assert stats.cut_density == pytest.approx(0.3, abs=1e-6)

    def test_window_outside_cuts(self):
        feats = self._feats()
        stats = visual_analysis.clip_stats(feats, 2.0, 4.0)
        assert stats.cuts_in_clip == 0
        assert stats.cut_density == pytest.approx(0.0)

    def test_cut_at_start_flag(self):
        feats = self._feats()
        assert visual_analysis.clip_stats(feats, 1.0, 3.0).cut_at_start
        assert not visual_analysis.clip_stats(feats, 3.0, 4.0).cut_at_start

    def test_face_and_motion_means(self):
        feats = self._feats()
        stats = visual_analysis.clip_stats(feats, 0.0, 0.4)
        assert stats.face_presence == pytest.approx(0.25, abs=1e-6)
        assert stats.face_motion == pytest.approx(0.053, abs=1e-3)

    def test_empty_features_is_neutral(self):
        feats = visual_analysis.VisualFeatures(duration=0.0)
        stats = visual_analysis.clip_stats(feats, 0.0, 5.0)
        assert stats.cuts_in_clip == 0
        assert stats.face_presence == pytest.approx(0.0)


class TestVisualScore:
    def test_none_is_neutral(self):
        assert visual_analysis.visual_score(None) == pytest.approx(0.5)

    def test_sweet_spot_scores_high(self):
        stats = visual_analysis.ClipVisualStats(
            cut_density=1.0 / 2.5,
            face_presence=0.28,
            face_motion=0.0625,  # -> mo=0.5 -> motion_score 1.0
            cut_at_start=True,
        )
        score = visual_analysis.visual_score(stats)
        assert score > 0.8
        assert score <= 1.0

    def test_no_face_no_cuts_is_middling(self):
        stats = visual_analysis.ClipVisualStats(
            cut_density=0.0, face_presence=0.02, face_motion=0.01
        )
        score = visual_analysis.visual_score(stats)
        assert 0.2 < score < 0.6

    def test_score_always_in_range(self):
        extremes = [
            visual_analysis.ClipVisualStats(cut_density=10.0, face_presence=0.99, face_motion=1.0),
            visual_analysis.ClipVisualStats(cut_density=0.0, face_presence=0.0, face_motion=0.0),
            visual_analysis.ClipVisualStats(cut_density=0.001, face_presence=0.28, face_motion=0.0625),
        ]
        for s in extremes:
            assert 0.0 <= visual_analysis.visual_score(s) <= 1.0

    def test_face_presence_sweet_spot_beats_extremes(self):
        base = dict(cut_density=0.4, face_motion=0.0625)
        good = visual_analysis.visual_score(
            visual_analysis.ClipVisualStats(face_presence=0.28, **base)
        )
        tiny = visual_analysis.visual_score(
            visual_analysis.ClipVisualStats(face_presence=0.01, **base)
        )
        huge = visual_analysis.visual_score(
            visual_analysis.ClipVisualStats(face_presence=0.9, **base)
        )
        assert good > tiny
        assert good > huge


# --------------------------------------------------------------------------
# Serialization
# --------------------------------------------------------------------------
class TestSerialization:
    def test_roundtrip(self, tmp_path):
        feats = visual_analysis.VisualFeatures(
            duration=12.0,
            cuts=[1.0, 3.5],
            face_area=[0.1, 0.2],
            motion=[0.05, 0.07],
            times=[0.0, 0.2],
        )
        p = visual_analysis.save_features(feats, tmp_path / "vis.json")
        assert p.exists()
        loaded = visual_analysis.load_features(p)
        assert loaded.duration == pytest.approx(12.0)
        assert loaded.cuts == [1.0, 3.5]
        assert loaded.face_area == [0.1, 0.2]
        assert loaded.times == [0.0, 0.2]

    def test_from_dict_defaults(self):
        loaded = visual_analysis.VisualFeatures.from_dict({"duration": 5.0})
        assert loaded.duration == pytest.approx(5.0)
        assert loaded.cuts == []
        assert loaded.sample_fps == pytest.approx(visual_analysis.VISUAL_FPS)
