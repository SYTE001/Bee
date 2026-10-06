"""Visual signal analysis: scene cuts, face presence, and motion.

Fase 2's scoring engine reads a `visual` dimension that Fase 1 stubbed at a
constant 0.5. This module computes the real value from the video stream:

  - scene cut density  – fast cuts signal edited, high-energy content
  - face presence      – a visible, close speaker performs better
  - face motion        – gestures and head movement read as energy on camera
  - shot change inside a clip – a cut mid-clip hurts standalone clarity

Everything degrades to a neutral score when the vision stack or the video
stream is unavailable, so the pipeline never fails on analysis alone.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from app.core.logging import get_logger
from app.services import media

log = get_logger("visual")

# Frames per second for the visual sample grid (must match reframe cadence).
VISUAL_FPS = 5.0

# A cut every N seconds counts as "fast". TikTok-grade editing sits around
# one cut per 2-4s; below 1.5s is aggressive even for short form.
FAST_CUT_INTERVAL = 2.5


@dataclass
class VisualFeatures:
    """Visual signals across the whole source, sampled at VISUAL_FPS."""

    duration: float = 0.0
    # Scene-cut times in seconds.
    cuts: list[float] = field(default_factory=list)
    # Per-sample face presence (fraction 0..1: area of the largest face / frame).
    face_area: list[float] = field(default_factory=list)
    # Per-sample inter-frame motion (mean absolute pixel delta, 0..1).
    motion: list[float] = field(default_factory=list)
    # Per-sample timestamps so the above lists line up with the timeline.
    times: list[float] = field(default_factory=list)
    sample_fps: float = VISUAL_FPS

    def to_dict(self) -> dict:
        return {
            "duration": self.duration,
            "cuts": self.cuts,
            "face_area": self.face_area,
            "motion": self.motion,
            "times": self.times,
            "sample_fps": self.sample_fps,
        }

    @classmethod
    def from_dict(cls, d: dict) -> VisualFeatures:
        return cls(
            duration=float(d.get("duration", 0.0)),
            cuts=[float(t) for t in d.get("cuts", [])],
            face_area=[float(v) for v in d.get("face_area", [])],
            motion=[float(v) for v in d.get("motion", [])],
            times=[float(t) for t in d.get("times", [])],
            sample_fps=float(d.get("sample_fps", VISUAL_FPS)),
        )


@dataclass
class ClipVisualStats:
    """Visual signals rolled up for one clip window."""

    cut_density: float = 0.0       # cuts per second inside the clip
    cuts_in_clip: int = 0
    face_presence: float = 0.0     # mean largest-face area fraction
    face_motion: float = 0.0       # mean inter-frame motion
    cut_at_start: bool = False     # clip opens on a shot change (good hook)

    def to_dict(self) -> dict:
        return {
            "cut_density": self.cut_density,
            "cuts_in_clip": self.cuts_in_clip,
            "face_presence": self.face_presence,
            "face_motion": self.face_motion,
            "cut_at_start": self.cut_at_start,
        }


def save_features(feats: VisualFeatures, path: str | Path) -> Path:
    """Write visual features to disk as JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(feats.to_dict()), encoding="utf-8")
    return out


def load_features(path: str | Path) -> VisualFeatures:
    """Load visual features from disk."""
    return VisualFeatures.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------
def analyze_video(
    source: str | Path,
    *,
    start: float = 0.0,
    end: float | None = None,
    fps: float = VISUAL_FPS,
    max_frames: int = 2400,
) -> VisualFeatures:
    """Compute visual features over the source (or a clip range).

    Scene cuts are detected with a cheap frame-difference detector: a
    normalised mean absolute difference above a threshold marks a cut. This
    avoids the PySceneDetect dependency while matching its content-detector
    behaviour closely enough for ranking purposes.
    """
    import numpy as np  # type: ignore[import-untyped]

    from app.services import reframe

    feats = VisualFeatures()
    prev: np.ndarray | None = None
    # A probe failure means the source is missing or unreadable — that is a
    # real error the caller must see, unlike a vision-stack hiccup below.
    info_probe = media.probe(source)

    if not info_probe.has_video:
        log.info("visual analysis: no video stream, skipping")
        feats.duration = info_probe.duration
        return feats

    feats.duration = (end if end is not None else info_probe.duration) - start
    frames_iter = reframe.sample_frames(
        source, start=start, end=end, fps=fps, max_frames=max_frames
    )
    detector = reframe.FaceDetector()
    try:
        for t, rgb in frames_iter:
            feats.times.append(t)
            small = _downsample(rgb)
            # Motion: mean absolute difference against the previous sample,
            # normalised to 0..1.
            if prev is not None and prev.shape == small.shape:
                feats.motion.append(float(np.abs(small.astype(np.int16) - prev.astype(np.int16)).mean()) / 255.0)
            else:
                feats.motion.append(0.0)
            prev = small

            # Face presence: largest face area as a fraction of the frame.
            try:
                faces = detector.detect(rgb)
            except Exception as exc:  # noqa: BLE001
                log.debug("face detect failed at %.2fs: %s", t, exc)
                faces = []
            frame_area = float(rgb.shape[0] * rgb.shape[1])
            largest = max((b.area for b in faces), default=0.0)
            feats.face_area.append(largest / max(frame_area, 1.0))
    finally:
        detector.close()

    feats.cuts = _detect_cuts(feats.times, feats.motion)
    log.info(
        "visual analysis: %d sample(s), %d cut(s), mean face=%.3f, mean motion=%.3f",
        len(feats.times), len(feats.cuts),
        sum(feats.face_area) / max(len(feats.face_area), 1),
        sum(feats.motion) / max(len(feats.motion), 1),
    )
    return feats


def _downsample(rgb: "np.ndarray", target_w: int = 160) -> "np.ndarray":
    """Shrink a frame for cheap motion comparison (aspect preserved)."""
    import numpy as np  # type: ignore[import-untyped]

    h, w = rgb.shape[:2]
    if w <= target_w:
        return rgb
    step = max(w // target_w, 1)
    return rgb[::step, ::step]


def _detect_cuts(
    times: list[float], motion: list[float], threshold: float = 0.22
) -> list[float]:
    """Scene cuts from the motion trace.

    A cut is a sample-to-sample change far larger than the local trend. The
    absolute threshold catches hard cuts; ignoring the first sample avoids a
    spurious cut at t=0.
    """
    if len(motion) < 3:
        return []
    cuts: list[float] = []
    for i in range(1, len(motion)):
        if motion[i] >= threshold and motion[i] > motion[i - 1] * 2.0:
            cuts.append(times[i])
    # Merge cuts closer than 0.4s — one transition can span two samples.
    merged: list[float] = []
    for t in cuts:
        if not merged or t - merged[-1] >= 0.4:
            merged.append(t)
    return merged


# --------------------------------------------------------------------------
# Clip-level rollup + scoring
# --------------------------------------------------------------------------
def clip_stats(
    feats: VisualFeatures, start: float, end: float, *, clip_start_bias: float = 0.35
) -> ClipVisualStats:
    """Roll visual features up for the window [start, end] (absolute times).

    `clip_start_bias` rewards a clip that opens on (or very near) a shot
    change — a hard cut at the first frame is a strong hook cue.
    """
    stats = ClipVisualStats()
    duration = max(end - start, 1e-6)

    cuts = [t for t in feats.cuts if start <= t < end]
    stats.cuts_in_clip = len(cuts)
    stats.cut_density = len(cuts) / duration
    stats.cut_at_start = any(abs(t - start) <= clip_start_bias for t in feats.cuts)

    if feats.times:
        # Samples whose midpoint falls in the clip window.
        lo, hi = start, end
        areas = [
            feats.face_area[i]
            for i, t in enumerate(feats.times)
            if lo <= t <= hi and i < len(feats.face_area)
        ]
        motions = [
            feats.motion[i]
            for i, t in enumerate(feats.times)
            if lo <= t <= hi and i < len(feats.motion)
        ]
        stats.face_presence = sum(areas) / max(len(areas), 1)
        stats.face_motion = sum(motions) / max(len(motions), 1)
    return stats


def visual_score(stats: ClipVisualStats | None) -> float:
    """Map clip visual stats onto the 0..1 `visual` score dimension.

    Weighted blend of the three signals that matter for short-form retention:
    a visible close face, on-camera energy, and an edited (cut-rich) look.
    A clip opening on a hard cut gets a small bonus.
    """
    if stats is None:
        return 0.5

    # Face presence: a face filling 15-40% of the frame is the sweet spot;
    # tiny faces and extreme close-ups both underperform.
    fp = stats.face_presence
    face_score = 1.0 - min(abs(fp - 0.28) / 0.28, 1.0)

    # Motion: moderate movement is good; pure static is flat, and jittery
    # camera work is distracting.
    mo = min(stats.face_motion * 8.0, 1.0)
    motion_score = 1.0 - abs(mo - 0.5) * 1.4
    motion_score = max(min(motion_score, 1.0), 0.0)

    # Cut density: ~1 cut per 2.5s is peak; both extremes lose points.
    density = stats.cut_density
    if density <= 0:
        cut_score = 0.35  # a single static shot is fine but not dynamic
    else:
        cut_score = 1.0 - min(abs(1.0 / max(density, 1e-6) - FAST_CUT_INTERVAL) / FAST_CUT_INTERVAL, 1.0)

    score = 0.45 * face_score + 0.30 * motion_score + 0.25 * cut_score
    if stats.cut_at_start:
        score = min(score + 0.05, 1.0)
    return max(min(score, 1.0), 0.0)
