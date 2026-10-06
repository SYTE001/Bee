"""Tests for the Fase 3 reframing service and ffmpeg crop expressions."""

from __future__ import annotations

import subprocess
import threading

import pytest

from app.services import reframe


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
def synth_face_video(path, duration: float = 3.0, fps: int = 15, moving: bool = True):
    """Generate a small test video with a bright 'face'-like moving patch.

    The patch is not a real face, so MediaPipe/Haar will not detect it —
    these tests exercise the *geometry* (crop plans, expressions, layouts),
    which is what actually has to be correct. Detector behaviour is covered
    by the smoke run against real content.
    """
    crop_x = "40+120*t" if moving else "160"
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", "testsrc=s=320x180:d={}:r={fps}".format(duration, fps=fps),
        "-f", "lavfi", "-i", "color=c=white:s=64x64:d={}:r={fps}".format(duration, fps=fps),
        "-filter_complex",
        f"[1:v]crop=64:64:{crop_x}:60[face];[0:v][face]overlay=shortest=1",
        "-frames:v", str(int(duration * fps)),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return str(path)


@pytest.fixture(scope="module")
def face_video(tmp_path_factory):
    d = tmp_path_factory.mktemp("reframe")
    return synth_face_video(d / "face.mp4")


# --------------------------------------------------------------------------
# Crop plan construction
# --------------------------------------------------------------------------
class TestBuildCropPlan:
    def _frames(self, times, faces_per_t):
        return [
            reframe.FrameFaces(t=t, faces=[reframe.FaceBox(*f) for f in faces])
            for t, faces in zip(times, faces_per_t)
        ]

    def test_no_faces_chooses_fit_blur(self):
        frames = self._frames([0.0, 0.2, 0.4], [[], [], []])
        plan = reframe.build_crop_plan(frames, layout="auto", width=1280, height=720)
        assert plan.layout == "fit_blur"
        assert len(plan.samples) == 1
        assert plan.is_static

    def test_single_face_chooses_single_focus(self):
        frames = self._frames(
            [0.0, 0.2, 0.4],
            [[(400, 200, 120, 120)], [(420, 210, 120, 120)], [(440, 200, 120, 120)]],
        )
        plan = reframe.build_crop_plan(frames, layout="auto", width=1280, height=720)
        assert plan.layout == "single_focus"
        # The window is sized to the face so the speaker fills the frame:
        # 120px face / 0.32 ratio = ~316px tall, 9:16 wide = ~178px.
        # It is clamped to at most the max 9:16 window (405x720 here).
        assert 170.0 <= plan.crop_w <= 405.0
        assert plan.crop_w / plan.crop_h == pytest.approx(1080 / 1920, rel=1e-3)
        # The face drifts only 40px, well inside the deadzone, so the camera
        # holds still instead of reproducing the drift as crop motion.
        assert plan.is_static

    def test_single_focus_follows_a_speaker_past_the_deadzone(self):
        """Once the face leaves the deadzone the camera glides after it."""
        frames = self._frames(
            [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            [
                [(400, 200, 120, 120)],
                [(420, 200, 120, 120)],
                [(440, 200, 120, 120)],
                [(700, 200, 120, 120)],
                [(960, 200, 120, 120)],
                [(980, 200, 120, 120)],
            ],
        )
        plan = reframe.build_crop_plan(frames, layout="auto", width=1280, height=720)
        assert plan.layout == "single_focus"
        assert not plan.is_static
        xs = [s.x for s in plan.samples]
        # The camera ends on the final pose, having followed the speaker right.
        assert xs[-1] > xs[0] + 100.0
        # And it moved monotonically: no back-and-forth on the way there.
        deltas = [b - a for a, b in zip(xs, xs[1:])]
        assert all(d >= -1e-6 for d in deltas)

    def test_two_faces_landscape_chooses_split_screen(self):
        frames = self._frames(
            [0.0, 0.2, 0.4],
            [
                [(100, 200, 120, 120), (1000, 200, 120, 120)],
                [(110, 210, 120, 120), (990, 200, 120, 120)],
                [(100, 200, 120, 120), (1000, 210, 120, 120)],
            ],
        )
        plan = reframe.build_crop_plan(frames, layout="auto", width=1280, height=720)
        assert plan.layout == "split_screen"
        # The speakers are 900 px apart, far wider than any 9:16 window, so the
        # plan must be a side-by-side composite rather than one centred crop.
        assert plan.is_composite
        assert len(plan.panels) == 2
        for panel in plan.panels:
            assert panel.out_w == pytest.approx(1080 / 2)
            assert panel.out_h == 1920
            for s in panel.samples:
                assert 0.0 <= s.x <= 1280 - panel.crop_w + 0.5

    def test_close_pair_uses_one_window(self):
        """Two speakers that fit a single window are not split into panels."""
        frames = self._frames(
            [0.0, 0.2],
            [[(490, 200, 120, 120), (650, 200, 120, 120)]] * 2,
        )
        plan = reframe.build_crop_plan(frames, layout="split_screen", width=1280, height=720)
        assert plan.layout == "split_screen"
        assert not plan.is_composite
        assert plan.samples

    def test_extra_faces_do_not_crash_split_screen(self):
        # A frame with 3+ detections (audience, false positives) must still
        # build a split-screen plan from the two largest faces.
        frames = self._frames(
            [0.0, 0.2, 0.4],
            [
                [(100, 200, 120, 120), (1000, 200, 120, 120), (500, 0, 20, 20)],
                [(110, 210, 120, 120), (990, 200, 120, 120), (505, 0, 20, 20)],
                [(100, 200, 120, 120), (1000, 210, 120, 120)],
            ],
        )
        plan = reframe.build_crop_plan(frames, layout="auto", width=1280, height=720)
        assert plan.layout == "split_screen"
        # The tiny third face is ignored: each panel tracks one of the two
        # large faces, not pulled toward the small one near the centre.
        left, right = plan.panels
        # Speaker centres: x=160 and x=1060; panel centres on its speaker.
        assert left.samples[0].x < 250
        assert right.samples[0].x > 750

    def test_explicit_layout_is_respected(self):
        frames = self._frames(
            [0.0, 0.2], [[(400, 200, 120, 120)], [(420, 210, 120, 120)]]
        )
        plan = reframe.build_crop_plan(frames, layout="fill_zoom", width=1280, height=720)
        assert plan.layout == "fill_zoom"
        assert plan.zoom_end > plan.zoom_start

    def test_unknown_layout_falls_back(self):
        frames = self._frames([0.0], [[(400, 200, 120, 120)]])
        plan = reframe.build_crop_plan(frames, layout="bogus", width=1280, height=720)
        assert plan.layout == "fit_blur"

    def test_crop_window_matches_output_aspect(self):
        frames = self._frames([0.0, 0.2], [[(400, 200, 120, 120)], [(420, 210, 120, 120)]])
        plan = reframe.build_crop_plan(frames, layout="single_focus", width=1280, height=720)
        assert plan.crop_w / plan.crop_h == pytest.approx(1080 / 1920, rel=1e-3)

    def test_crop_stays_inside_source(self):
        # Face far off the top-left corner must not push the crop negative.
        frames = self._frames([0.0, 0.2], [[(0, 0, 120, 120)], [(10, 10, 120, 120)]])
        plan = reframe.build_crop_plan(frames, layout="single_focus", width=1280, height=720)
        for s in plan.samples:
            assert 0.0 <= s.x <= 1280 - plan.crop_w + 0.5
            assert 0.0 <= s.y <= 720 - plan.crop_h + 0.5

    def test_face_in_safe_zone(self):
        """The tracked face should land in the upper-middle, not the edge."""
        frames = self._frames(
            [0.0, 0.2, 0.4],
            [[(640, 360, 120, 120)], [(640, 360, 120, 120)], [(640, 360, 120, 120)]],
        )
        plan = reframe.build_crop_plan(frames, layout="single_focus", width=1280, height=720)
        assert plan.focus
        for f in plan.focus:
            # Face centre in output coords: x around 45%, y around 40%.
            assert 0.30 * 1080 <= f["x"] <= 0.60 * 1080
            assert 0.20 * 1920 <= f["y"] <= 0.65 * 1920


# --------------------------------------------------------------------------
# Tracking + smoothing
# --------------------------------------------------------------------------
class TestTracking:
    def test_track_follows_nearest_face(self):
        frames = [
            reframe.FrameFaces(t=0.0, faces=[reframe.FaceBox(100, 100, 80, 80)]),
            reframe.FrameFaces(t=0.2, faces=[reframe.FaceBox(110, 105, 80, 80)]),
            reframe.FrameFaces(t=0.4, faces=[]),  # brief miss
            reframe.FrameFaces(t=0.6, faces=[reframe.FaceBox(130, 110, 80, 80)]),
        ]
        track = reframe._track_faces(frames)
        assert len(track) == 4
        # True face centres (x + w/2) move 140 -> 150 -> (gap) -> 170.
        # The gap frame inherits the Kalman state, extrapolating the observed
        # motion instead of going stale.
        assert track[0].box.cx == pytest.approx(140, abs=5)
        assert 145.0 <= track[1].box.cx <= 152.0
        assert track[2].box.cx >= track[1].box.cx  # keeps drifting forward
        # The real detection after the gap is recovered with only the small
        # residual lag a Kalman filter has on an accelerating target.
        assert track[3].box.cx == pytest.approx(170, abs=8)

    def test_reappearing_face_is_tracked(self):
        """A face reappearing after a long gap still extends the track."""
        frames = [
            reframe.FrameFaces(t=0.0, faces=[reframe.FaceBox(100, 100, 80, 80)]),
            reframe.FrameFaces(t=10.0, faces=[reframe.FaceBox(900, 500, 80, 80)]),
        ]
        track = reframe._track_faces(frames, max_gap=2.0)
        # Both detections are kept; max_gap only governs *empty* frames.
        assert len(track) == 2
        assert track[0].box.x == pytest.approx(100, abs=10)
        # A single-sample jump of 800px is treated as a probable glitch and
        # only partially applied — the crop eases toward it instead of
        # teleporting, which is the behaviour we want on a bad detection.
        assert 600.0 < track[1].box.x <= 900.0

    def test_short_gap_is_bridged(self):
        frames = [
            reframe.FrameFaces(t=0.0, faces=[reframe.FaceBox(100, 100, 80, 80)]),
            reframe.FrameFaces(t=1.0, faces=[]),
            reframe.FrameFaces(t=2.0, faces=[reframe.FaceBox(140, 110, 80, 80)]),
        ]
        track = reframe._track_faces(frames, max_gap=2.0)
        assert len(track) == 3
        # The missing frame inherits the tracked state.
        assert track[1].box.cx == pytest.approx(120, abs=30)

    def test_gap_beyond_max_gap_is_not_bridged(self):
        frames = [
            reframe.FrameFaces(t=0.0, faces=[reframe.FaceBox(100, 100, 80, 80)]),
            reframe.FrameFaces(t=10.0, faces=[]),
            reframe.FrameFaces(t=20.0, faces=[reframe.FaceBox(140, 110, 80, 80)]),
        ]
        track = reframe._track_faces(frames, max_gap=2.0)
        # The unbridged middle frame is dropped; only real detections remain.
        assert len(track) == 2
        assert track[0].box.x == pytest.approx(100, abs=10)
        assert track[1].box.x == pytest.approx(140, abs=10)

    def test_kalman_follows_a_steady_pan(self):
        """The filter tracks a constant-velocity pan with no residual lag."""
        kf = reframe.Kalman1D(0.0)
        # 10 px per step.
        vals = [kf.update(i * 10.0, 0.2) for i in range(1, 21)]
        # By the end the filter should sit at the true value, not behind it.
        assert vals[-1] == pytest.approx(200.0, abs=5.0)

    def test_kalman_damps_jitter(self):
        kf = reframe.Kalman1D(100.0)
        noisy = [100 + 8 * ((-1) ** i) for i in range(1, 21)]
        out = [kf.update(v, 0.2) for v in noisy]
        # The smoothed output has far less swing than the input.
        assert max(out) - min(out) < max(noisy) - min(noisy)
        # And stays centred on the true value.
        assert abs(sum(out) / len(out) - 100.0) < 3.0

    def test_kalman_zero_dt_is_safe(self):
        kf = reframe.Kalman1D(5.0)
        assert kf.update(9.0, 0.0) == pytest.approx(5.0)

    def test_ema_smooths_jitter(self):
        raw = [100.0, 140.0, 60.0, 150.0, 70.0, 145.0]
        out = reframe._ema_smooth(raw, alpha=0.3)
        # The smoothed series is monotonically less wild than the input.
        assert max(out) - min(out) < max(raw) - min(raw)
        assert out[0] == raw[0]

    def test_decimate_keeps_endpoints_and_limit(self):
        samples = [reframe.CropSample(t=i * 0.2, x=float(i), y=0.0) for i in range(100)]
        out = reframe._decimate(samples, max_keys=10)
        assert len(out) <= 10
        assert out[0].t == pytest.approx(0.0)
        assert out[-1].t == pytest.approx(samples[-1].t)


class TestDeadzoneCamera:
    """The virtual camera: hold, glide, snap."""

    def test_noise_inside_the_deadzone_is_ignored(self):
        """Small oscillation never moves the camera — this is the anti-jitter core."""
        raw = [500.0 + 12 * ((-1) ** i) for i in range(20)]
        out = reframe._deadzone_smooth(raw, deadzone_px=45.0)
        # The camera never moves off its opening position.
        assert max(out) - min(out) < 1e-6
        assert out[0] == pytest.approx(raw[0])

    def test_a_drift_past_the_deadzone_glidess_after_it(self):
        """A steady drift is followed, but always lagging by about a deadzone."""
        raw = [500.0 + 30.0 * i for i in range(20)]
        out = reframe._deadzone_smooth(raw, deadzone_px=45.0, smooth=0.5)
        assert out[-1] > out[0]
        # The camera trails the target: it moves, but never overshoots it.
        assert all(o <= r + 1e-6 for o, r in zip(out, raw))
        assert out[-1] < raw[-1]

    def test_a_jump_beyond_snap_teleports(self):
        """A shot cut is taken instantly rather than panned across."""
        raw = [500.0, 520.0, 1100.0, 1120.0]
        out = reframe._deadzone_smooth(raw, deadzone_px=45.0, snap_px=150.0)
        assert out[2] == pytest.approx(1100.0)

    def test_a_jump_below_snap_glidess(self):
        """A jump smaller than the snap threshold is still followed smoothly."""
        raw = [500.0, 520.0, 600.0, 620.0]
        out = reframe._deadzone_smooth(raw, deadzone_px=45.0, snap_px=150.0)
        assert out[2] < 600.0  # not a snap
        assert out[2] > out[0]  # but it did move

    def test_empty_series_is_safe(self):
        assert reframe._deadzone_smooth([], deadzone_px=45.0) == []


# --------------------------------------------------------------------------
# ffmpeg expression generation
# --------------------------------------------------------------------------
class TestCropFilter:
    def test_static_plan_uses_plain_crop(self):
        plan = reframe.CropPlan(
            layout="fit_blur",
            source_width=1280,
            source_height=720,
            crop_w=405,
            crop_h=720,
            samples=[reframe.CropSample(t=0.0, x=437.5, y=0.0)],
        )
        f = reframe.crop_filter(plan)
        assert f == "crop=405:720:437.5:0"
        assert "if(" not in f

    def test_animated_plan_builds_if_chain(self):
        plan = reframe.CropPlan(
            layout="single_focus",
            source_width=1280,
            source_height=720,
            crop_w=405,
            crop_h=720,
            samples=[
                reframe.CropSample(t=0.0, x=100, y=50),
                reframe.CropSample(t=1.0, x=200, y=50),
                reframe.CropSample(t=2.0, x=300, y=50),
            ],
        )
        f = reframe.crop_filter(plan)
        assert f.startswith("crop=405:720:x='")
        # Each animated axis interpolates across 2 intervals with a nested
        # branch per keyframe, so 2 axes * 4 branches = 8 occurrences.
        assert f.count("if(gt(t,") == 8
        # Linear interpolation: at the midpoints the value is halfway.
        assert expr_value(f, "x", 0.5) == pytest.approx(150, abs=1)
        assert expr_value(f, "x", 1.5) == pytest.approx(250, abs=1)
        # Held constant after the last keyframe.
        assert expr_value(f, "x", 2.5) == pytest.approx(300, abs=1)

    def test_single_sample_never_produces_if(self):
        plan = reframe.CropPlan(
            layout="fill_zoom",
            source_width=1280,
            source_height=720,
            crop_w=405,
            crop_h=720,
            samples=[reframe.CropSample(t=0.0, x=400, y=0)],
        )
        assert "if(" not in reframe.crop_filter(plan)

    def test_fit_blur_chain_is_well_formed(self):
        plan = reframe.CropPlan(
            layout="fit_blur", source_width=1280, source_height=720
        )
        chain = reframe.fit_blur_filters(plan)
        assert chain[0].startswith("split=2")
        assert chain[-1].startswith("[bgfill][fgfit]overlay")
        assert all(s for s in chain)

    def test_split_screen_chain_is_well_formed(self):
        plan = reframe.CropPlan(
            layout="split_screen",
            source_width=1280,
            source_height=720,
            panels=[
                reframe.PanelPlan(
                    crop_w=202.5, crop_h=720, canvas_x=0, canvas_y=0,
                    out_w=540, out_h=1920,
                    samples=[reframe.CropSample(t=0.0, x=59, y=0)],
                ),
                reframe.PanelPlan(
                    crop_w=202.5, crop_h=720, canvas_x=540, canvas_y=0,
                    out_w=540, out_h=1920,
                    samples=[reframe.CropSample(t=0.0, x=959, y=0)],
                ),
            ],
        )
        chain = reframe.split_screen_filters(plan)
        assert chain[0] == "split=2[src0][src1]"
        assert chain[-2] == "[pan0][pan1]hstack=inputs=2"
        assert chain[-1].startswith("drawbox")
        assert all(s for s in chain)

    def test_split_screen_chain_runs_in_ffmpeg(self, face_video):
        """The composite chain ffmpeg receives must parse and render."""
        plan = reframe.CropPlan(
            layout="split_screen",
            source_width=320,
            source_height=180,
            panels=[
                reframe.PanelPlan(
                    crop_w=50, crop_h=180, canvas_x=0, canvas_y=0,
                    out_w=540, out_h=1920,
                    samples=[reframe.CropSample(t=0.0, x=0, y=0)],
                ),
                reframe.PanelPlan(
                    crop_w=50, crop_h=180, canvas_x=540, canvas_y=0,
                    out_w=540, out_h=1920,
                    samples=[reframe.CropSample(t=0.0, x=270, y=0)],
                ),
            ],
        )
        vf = ",".join(reframe.split_screen_filters(plan))
        out = face_video.replace(".mp4", "_split.mp4")
        r = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", face_video,
             "-vf", vf, "-t", "1.0", "-an", "-c:v", "libx264", out],
            capture_output=True,
        )
        assert r.returncode == 0, r.stderr.decode()[-500:]
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0", out],
            capture_output=True,
        )
        assert probe.stdout.decode().strip() == "1080,1920"

    def test_scale_filter_targets_canvas(self):
        plan = reframe.CropPlan(layout="fit_blur", source_width=1280, source_height=720)
        assert reframe.scale_filter(plan) == "scale=1080:1920:flags=lanczos"

    def test_generated_filter_runs_in_ffmpeg(self, face_video):
        """The expression ffmpeg receives must actually parse and render."""
        plan = reframe.CropPlan(
            layout="single_focus",
            source_width=320,
            source_height=180,
            crop_w=101,
            crop_h=180,
            samples=[
                reframe.CropSample(t=0.0, x=20, y=0),
                reframe.CropSample(t=1.0, x=60, y=0),
                reframe.CropSample(t=2.0, x=100, y=0),
            ],
        )
        vf = f"{reframe.crop_filter(plan)},{reframe.scale_filter(plan)}"
        out = face_video.replace(".mp4", "_cropped.mp4")
        r = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", face_video,
             "-vf", vf, "-t", "2.5", "-an", "-c:v", "libx264", out],
            capture_output=True,
        )
        assert r.returncode == 0, r.stderr.decode()[-500:]
        import os

        assert os.path.exists(out) and os.path.getsize(out) > 0


def expr_value(filter_str: str, axis: str, t: float) -> float:
    """Evaluate one crop axis expression at time `t`.

    Parses the same nested `if(gt(t,k), A, B)` form ffmpeg's crop filter
    evaluates per frame, so the assertions check exactly what the renderer
    will compute rather than an approximation of it.
    """
    import re as _re

    m = _re.search(rf"{axis}='([^']+)'", filter_str)
    if not m:
        raise ValueError(f"no {axis} expression in {filter_str}")
    return _eval_expr(m.group(1), t)


def _split_args(expr: str) -> tuple[str, str]:
    """Split `if(COND, A, B)` into (A, B) at the top-level commas."""
    depth = 0
    splits: list[int] = []
    for i, ch in enumerate(expr):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 1:
            splits.append(i)
    if len(splits) < 2:
        raise ValueError(f"not a 2-arg branch: {expr}")
    a = expr[splits[0] + 1 : splits[1]]
    b = expr[splits[1] + 1 : -1]
    return a, b


def _eval_expr(expr: str, t: float) -> float:
    """Recursively evaluate the ffmpeg crop expression at time `t`."""
    import re as _re

    expr = expr.strip()
    if not expr.startswith("if(gt(t,"):
        # Leaf: either a plain number or a linear formula `v0+(slope)*(t-t0)`.
        lin = _re.match(r"^([0-9.\-]+)\+\(([0-9.\-]+)\)\*\(t-([0-9.\-]+)\)$", expr)
        if lin:
            v0, slope, t0 = (float(x) for x in lin.groups())
            return v0 + slope * (t - t0)
        return float(expr)

    m = _re.match(r"if\(gt\(t,([0-9.\-]+)\),", expr)
    if not m:
        raise ValueError(f"unparsed branch: {expr}")
    k = float(m.group(1))
    above, below = _split_args(expr)
    if t > k:
        return _eval_expr(above, t)
    return _eval_expr(below, t)


# --------------------------------------------------------------------------
# Serialization
# --------------------------------------------------------------------------
class TestSerialization:
    def test_roundtrip(self):
        plan = reframe.CropPlan(
            layout="single_focus",
            source_width=1280,
            source_height=720,
            crop_w=405,
            crop_h=720,
            samples=[reframe.CropSample(t=0.0, x=1.5, y=2.5)],
            focus=[{"t": 0.0, "x": 486.0, "y": 768.0}],
            zoom_start=1.0,
            zoom_end=1.12,
            detector="mediapipe",
            face_count=3,
        )
        blob = plan.to_dict()
        restored = reframe.CropPlan.from_dict(blob)
        assert restored.layout == plan.layout
        assert restored.crop_w == pytest.approx(plan.crop_w)
        assert restored.samples[0].x == pytest.approx(1.5)
        assert restored.focus == plan.focus
        assert restored.detector == "mediapipe"
        assert restored.face_count == 3

    def test_save_load(self, tmp_path):
        plan = reframe.CropPlan(
            layout="fit_blur",
            source_width=1280,
            source_height=720,
            crop_w=405,
            crop_h=720,
            samples=[reframe.CropSample(t=0.0, x=437.5, y=0.0)],
        )
        p = reframe.save_plan(plan, tmp_path / "plan.json")
        assert p.exists()
        loaded = reframe.load_plan(p)
        assert loaded.layout == "fit_blur"
        assert loaded.samples[0].x == pytest.approx(437.5)


# --------------------------------------------------------------------------
# Frame sampling
# --------------------------------------------------------------------------
class TestSampleFrames:
    def test_sample_frames_yields_expected_count(self, face_video):
        frames = list(reframe.sample_frames(face_video, fps=5, max_frames=60))
        # 3s at 5fps = 15 frames (may be a couple short after decode).
        assert 10 <= len(frames) <= 16
        assert all(t >= 0.0 for t, _ in frames)
        # Timestamps advance at the sampling cadence.
        assert frames[1][0] == pytest.approx(0.2, abs=0.05)

    def test_sample_frames_respects_range(self, face_video):
        frames = list(reframe.sample_frames(face_video, start=1.0, end=2.0, fps=5))
        assert 3 <= len(frames) <= 6
        assert all(t <= 1.05 for t, _ in frames)

    def test_downscale_shapes_match(self, face_video):
        for _, rgb in reframe.sample_frames(face_video, fps=2, max_frames=2):
            assert rgb.shape[:2] == (180, 320)
            break


# --------------------------------------------------------------------------
# Public API sanity
# --------------------------------------------------------------------------
def test_constants_and_exports():
    assert reframe.OUT_WIDTH == 1080
    assert reframe.OUT_HEIGHT == 1920
    assert "single_focus" in reframe.LAYOUTS
    assert "auto" in reframe.LAYOUTS
    assert reframe.SAFE_BOTTOM > reframe.SAFE_TOP


def test_face_box_geometry():
    box = reframe.FaceBox(x=100, y=50, w=80, h=120)
    assert box.cx == pytest.approx(140)
    assert box.cy == pytest.approx(110)
    assert box.area == pytest.approx(80 * 120)
