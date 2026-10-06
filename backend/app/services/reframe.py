"""9:16 reframing with face tracking.

Turns any source into a vertical 1080x1920 canvas using one of four layouts:

  single_focus  – tight crop around the active speaker's face, tracked frame
                  by frame and smoothed with an EMA filter.
  split_screen  – two speakers side by side (podcast / interview).
  fit_blur      – the full frame visible, blurred background fill.
  fill_zoom     – centre crop with a slow dynamic zoom (Ken Burns).

Detection uses MediaPipe Face Mesh when installed, an OpenCV Haar cascade as
a fallback, and a deterministic centre fallback so the pipeline still produces
a valid (static) plan on machines with no vision stack.

The analysis pass samples a few frames per second and writes a *crop plan*:
the crop window at a handful of keyframes. The renderer (Fase 5) converts that
into an ffmpeg `crop` expression with piecewise-linear interpolation between
keyframes. That keeps analysis cheap while the pan stays smooth at full fps.

ffmpeg constraint: `crop` evaluates its `x`/`y` expressions per frame but its
`w`/`h` only once at setup, so a plan's window size is constant and only the
position animates. Zoom therefore changes the *crop size*, which is why a plan
carries its own fixed `crop_w`/`crop_h` and a `zoom` curve for the renderer to
apply via `zoompan` on the already-cropped stream.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.core.logging import get_logger
from app.services import media

log = get_logger("reframe")

# Vertical output canvas (TikTok / Reels / Shorts).
OUT_WIDTH = 1080
OUT_HEIGHT = 1920

# TikTok safe zones, as fractions of the output canvas:
#  - top: camera notch, username, and the "following" banner
#  - bottom: caption text plus the like / comment / share rail
#  - right: the engagement button stack
SAFE_TOP = 0.10
SAFE_BOTTOM = 0.80
SAFE_RIGHT = 0.93

# Sampling rate for face detection. Per-frame detection is far too slow; with
# an EMA-smoothed crop path, 5 Hz is indistinguishable from full tracking.
DETECT_FPS = 5.0

# How far above the crop centre the face sits. Viewers expect the eyes in the
# upper third, not the middle of the frame.
HEADROOM = 0.40

# Target face height as a fraction of the output frame. 0.32 leaves room
# around the speaker — at 0.38 the framing read as a close-up that cropped
# shoulders out of the shot.
TARGET_FACE_RATIO = 0.32

# Hardest the single-focus layout is allowed to zoom in past the widest 9:16
# window. A tighter cap keeps the framing from closing in on a small face.
ZOOM_MAX = 1.35

# Virtual camera, adapted from the open-source-clipping project: the camera
# holds dead still while the speaker stays inside a deadzone, glides after
# them once they leave it, and snaps outright across a shot cut instead of
# panning slowly across the frame. A filter (EMA, Kalman) tracks the detector
# and so replays its per-sample noise as camera motion — measured on real
# podcast footage, the crop reversed direction at 70-80% of keyframes.
CAM_DEADZONE = 0.15  # share of the crop width the camera tolerates
CAM_SMOOTH = 0.30    # share of the remaining gap closed per sample
CAM_SNAP = 0.15      # share of the source width that means a cut

LAYOUTS = ("auto", "single_focus", "split_screen", "fit_blur", "fill_zoom")

# Bumped whenever the plan schema changes. Plans are cached by time range across
# re-runs, so an old cache must not be replayed under new semantics: the version
# rides along in the cache key and the stored dict, and a mismatch means
# "rebuild this plan".
PLAN_VERSION = 3


def plan_cache_key(start: float, end: float, layout: str) -> str:
    """Cache identity for one clip's crop plan."""
    return f"{start:.3f}:{end:.3f}:{layout}:v{PLAN_VERSION}"


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------
@dataclass
class FaceBox:
    """A detected face at one sample, in source-pixel coordinates."""

    x: float
    y: float
    w: float
    h: float
    confidence: float = 1.0

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0

    @property
    def area(self) -> float:
        return self.w * self.h


@dataclass
class FrameFaces:
    """All faces found in one sampled frame."""

    t: float
    faces: list[FaceBox] = field(default_factory=list)


@dataclass
class CropSample:
    """One keyframe of the crop path, in source-pixel coordinates."""

    t: float
    x: float
    y: float


@dataclass
class PanelPlan:
    """One speaker panel of a side-by-side composite.

    `crop_w`/`crop_h` are the (constant) source window for this panel and
    `samples` animate its position to follow the speaker; `canvas_x`/`canvas_y`
    plus `out_w`/`out_h` place the scaled panel on the output canvas.
    """

    crop_w: float = 0.0
    crop_h: float = 0.0
    canvas_x: float = 0.0
    canvas_y: float = 0.0
    out_w: float = 0.0
    out_h: float = 0.0
    samples: list[CropSample] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "samples": [asdict(s) for s in self.samples],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PanelPlan:
        return cls(
            crop_w=float(d.get("crop_w", 0.0)),
            crop_h=float(d.get("crop_h", 0.0)),
            canvas_x=float(d.get("canvas_x", 0.0)),
            canvas_y=float(d.get("canvas_y", 0.0)),
            out_w=float(d.get("out_w", 0.0)),
            out_h=float(d.get("out_h", 0.0)),
            samples=[
                CropSample(t=float(s["t"]), x=float(s["x"]), y=float(s["y"]))
                for s in d.get("samples", [])
            ],
        )


@dataclass
class CropPlan:
    """A complete reframing plan for one clip.

    `crop_w`/`crop_h` are the (constant) source-pixel window; `samples` give
    its animated position. `zoom_start`/`zoom_end` let the renderer add a
    slow push-in. `focus` is the tracked face centre in *output* coordinates,
    used to place titles and progress bars without covering the speaker.
    `panels` holds the per-speaker windows of a `split_screen` composite; when
    it is non-empty `samples` is unused because the canvas is assembled from
    the panels instead of a single crop.
    """

    layout: str
    source_width: int
    source_height: int
    out_width: int = OUT_WIDTH
    out_height: int = OUT_HEIGHT
    crop_w: float = 0.0
    crop_h: float = 0.0
    samples: list[CropSample] = field(default_factory=list)
    focus: list[dict[str, float]] = field(default_factory=list)
    panels: list[PanelPlan] = field(default_factory=list)
    zoom_start: float = 1.0
    zoom_end: float = 1.0
    detector: str = "none"
    face_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["samples"] = [asdict(s) for s in self.samples]
        d["panels"] = [p.to_dict() for p in self.panels]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CropPlan:
        return cls(
            layout=d.get("layout", "fit_blur"),
            source_width=int(d.get("source_width", 0)),
            source_height=int(d.get("source_height", 0)),
            out_width=int(d.get("out_width", OUT_WIDTH)),
            out_height=int(d.get("out_height", OUT_HEIGHT)),
            crop_w=float(d.get("crop_w", 0.0)),
            crop_h=float(d.get("crop_h", 0.0)),
            samples=[
                CropSample(t=float(s["t"]), x=float(s["x"]), y=float(s["y"]))
                for s in d.get("samples", [])
            ],
            focus=[dict(f) for f in d.get("focus", [])],
            panels=[PanelPlan.from_dict(p) for p in d.get("panels", [])],
            zoom_start=float(d.get("zoom_start", 1.0)),
            zoom_end=float(d.get("zoom_end", 1.0)),
            detector=d.get("detector", "none"),
            face_count=int(d.get("face_count", 0)),
        )

    @property
    def is_composite(self) -> bool:
        """True when the canvas is assembled from panels, not one crop."""
        return bool(self.panels)

    @property
    def is_static(self) -> bool:
        """True when the crop never moves (renderer can skip the expression)."""
        if len(self.samples) <= 1:
            return True
        first = self.samples[0]
        return all(
            abs(s.x - first.x) < 1.0 and abs(s.y - first.y) < 1.0 for s in self.samples
        )


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------
class FaceDetector:
    """Face detection with graceful degradation.

    Resolution order, best to cheapest:

    1. MediaPipe Face Mesh — accurate, gives the full head outline.
    2. OpenCV YuNet (``FaceDetectorYN``) — accurate DNN detector, ships as a
       tiny ONNX that is fetched once into ``MODELS_DIR``.
    3. OpenCV Haar cascade — always available with opencv-python, less
       accurate and prone to misses on profile/rotated faces.
    4. Nothing — the plan builder falls back to a centred crop so the
       pipeline still produces output.

    The first backend that initializes wins; detection failures at runtime
    fall through to the next tier so one bad model does not kill a run.
    """

    YUNET_URL = (
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx"
    )
    YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"

    def __init__(self) -> None:
        self.name = "none"
        self._mesh = None
        self._yunet = None
        self._haar = None
        self._init_backends()

    def _yunet_path(self) -> Path | None:
        """Locate the YuNet ONNX, downloading it once if needed."""
        from app.core.config import settings

        model_dir = Path(settings.MODELS_DIR)
        target = model_dir / self.YUNET_FILENAME
        if target.exists() and target.stat().st_size > 0:
            return target
        try:
            model_dir.mkdir(parents=True, exist_ok=True)
            import urllib.request

            log.info("downloading YuNet face detector -> %s", target)
            req = urllib.request.Request(self.YUNET_URL, headers={"User-Agent": "clipforge"})
            with urllib.request.urlopen(req, timeout=60) as resp, target.open("wb") as fh:
                fh.write(resp.read())
            return target
        except Exception as exc:  # noqa: BLE001 - offline / blocked hosts
            log.debug("yunet download failed: %s", exc)
            return None

    def _init_backends(self) -> None:
        try:
            import mediapipe as mp  # type: ignore[import-untyped]

            self._mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=2,
                refine_landmarks=False,
                min_detection_confidence=0.5,
                static_image_mode=True,
            )
            self.name = "mediapipe"
            return
        except Exception as exc:  # noqa: BLE001 - optional dependency
            log.debug("mediapipe unavailable: %s", exc)

        # YuNet needs OpenCV's DNN module and the ONNX model file.
        try:
            import cv2  # type: ignore[import-untyped]

            if hasattr(cv2, "FaceDetectorYN"):
                model = self._yunet_path()
                if model is not None:
                    # YuNet downscales to its input size, so a wide 1280x720
                    # frame at 320px input turns a 200px face into ~50px and
                    # it stops detecting. 640px keeps faces large enough.
                    self._yunet_size = (640, 640)
                    self._yunet = cv2.FaceDetectorYN.create(
                        str(model), "", self._yunet_size, score_threshold=0.5,
                        nms_threshold=0.3, top_k=500,
                    )
                    self.name = "yunet"
                    return
        except Exception as exc:  # noqa: BLE001 - optional dependency
            log.debug("yunet unavailable: %s", exc)

        try:
            import cv2  # type: ignore[import-untyped]

            cascade = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # type: ignore[attr-defined]
            self._haar = cv2.CascadeClassifier(cascade)
            if not self._haar.empty():
                self.name = "haar"
                return
        except Exception as exc:  # noqa: BLE001 - optional dependency
            log.debug("opencv haar cascade unavailable: %s", exc)

        log.warning(
            "no face detector available — reframing will use a centred crop. "
            "Install mediapipe (requirements-ml.txt) for tracked reframing."
        )

    def detect(self, frame_rgb: Any) -> list[FaceBox]:
        """Detect faces in an RGB frame; returns boxes in pixel coordinates."""
        if self.name == "mediapipe" and self._mesh is not None:
            try:
                return self._detect_mesh(frame_rgb)
            except Exception as exc:  # noqa: BLE001
                log.debug("mediapipe detect failed: %s", exc)
                return self._detect_fallback(frame_rgb)
        if self.name == "yunet" and self._yunet is not None:
            try:
                return self._detect_yunet(frame_rgb)
            except Exception as exc:  # noqa: BLE001
                log.debug("yunet detect failed: %s", exc)
                return self._detect_fallback(frame_rgb)
        if self.name == "haar" and self._haar is not None:
            try:
                return self._detect_haar(frame_rgb)
            except Exception as exc:  # noqa: BLE001
                log.debug("haar detect failed: %s", exc)
        return []

    def _detect_fallback(self, frame_rgb: Any) -> list[FaceBox]:
        """Try the cheaper tiers after the primary detector errors out."""
        if self._haar is not None:
            try:
                return self._detect_haar(frame_rgb)
            except Exception:  # noqa: BLE001
                pass
        return []

    def close(self) -> None:
        if self._mesh is not None:
            try:
                self._mesh.close()
            except Exception:  # noqa: BLE001
                pass
        self._mesh = None
        self._yunet = None
        self._haar = None

    # -- mediapipe ---------------------------------------------------------
    def _detect_mesh(self, frame_rgb: Any) -> list[FaceBox]:
        h, w = frame_rgb.shape[:2]
        results = self._mesh.process(frame_rgb)
        if not results.multi_face_landmarks:
            return []
        boxes: list[FaceBox] = []
        for landmarks in results.multi_face_landmarks:
            xs = [lm.x for lm in landmarks.landmark]
            ys = [lm.y for lm in landmarks.landmark]
            x0, x1 = min(xs) * w, max(xs) * w
            y0, y1 = min(ys) * h, max(ys) * h
            # Face Mesh covers the whole head; the *face* is roughly the
            # middle 80% vertically. Cropping to that avoids bald-head frames.
            face_h = (y1 - y0) * 0.80
            cy = (y0 + y1) / 2.0
            boxes.append(
                FaceBox(
                    x=max(x0, 0.0),
                    y=max(cy - face_h / 2.0, 0.0),
                    w=min(x1 - x0, float(w)),
                    h=min(face_h, float(h)),
                    confidence=0.9,
                )
            )
        return boxes

    # -- yunet ---------------------------------------------------------------
    def _detect_yunet(self, frame_rgb: Any) -> list[FaceBox]:
        import cv2  # type: ignore[import-untyped]
        import numpy as np  # type: ignore[import-untyped]

        h, w = frame_rgb.shape[:2]
        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        # Pad to a square canvas instead of stretching: YuNet is sensitive to
        # aspect distortion, and letterboxing keeps faces undistorted.
        target = max(self._yunet_size)
        scale = target / max(h, w)
        pad_w, pad_h = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(bgr, (pad_w, pad_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((target, target, 3), np.uint8)
        canvas[:pad_h, :pad_w] = resized
        self._yunet.setInputSize((target, target))
        _, faces = self._yunet.detect(canvas)
        if faces is None:
            return []
        sx, sy = w / pad_w, h / pad_h
        boxes: list[FaceBox] = []
        for f in faces:
            x, y, fw, fh = float(f[0]), float(f[1]), float(f[2]), float(f[3])
            score = float(f[14]) if len(f) > 14 else 0.5
            # Undo the padding offset, then scale back to source pixels.
            bx, by = (x - 0.0) * sx, (y - 0.0) * sy
            if fw <= 0 or fh <= 0 or score < 0.5:
                continue
            boxes.append(
                FaceBox(
                    x=max(bx, 0.0),
                    y=max(by, 0.0),
                    w=min(fw * sx, float(w) - bx),
                    h=min(fh * sy, float(h) - by),
                    confidence=score,
                )
            )
        return boxes

    # -- haar ---------------------------------------------------------------
    def _detect_haar(self, frame_rgb: Any) -> list[FaceBox]:
        import cv2  # type: ignore[import-untyped]

        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        rects = self._haar.detectMultiScale(
            gray, scaleFactor=1.15, minNeighbors=5, minSize=(48, 48)
        )
        return [
            FaceBox(x=float(x), y=float(y), w=float(w), h=float(h), confidence=0.6)
            for x, y, w, h in rects
        ]


def sample_frames(
    source: str | Path,
    *,
    start: float = 0.0,
    end: float | None = None,
    fps: float = DETECT_FPS,
    max_frames: int = 2400,
):
    """Yield (timestamp_seconds, rgb_frame) at a reduced framerate.

    Decoding goes through ffmpeg's rawvideo pipe so we don't depend on
    OpenCV's video backends, which are absent in `opencv-python-headless`.
    Timestamps are relative to `start`, i.e. clip-local time.
    """
    import numpy as np  # type: ignore[import-untyped]
    import subprocess

    info = media.probe(source)
    if not info.has_video or info.width == 0 or info.height == 0:
        return

    w, h = info.width, info.height
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{max(start, 0.0):.3f}", "-i", str(source)]
    if end is not None and end > start:
        cmd += ["-t", f"{end - start:.3f}"]
    cmd += [
        "-vf", f"fps={fps},scale={w}:{h}:flags=bicubic",
        "-pix_fmt", "rgb24",
        "-f", "rawvideo",
        "-",
    ]

    frame_size = w * h * 3
    buf = bytearray()
    frame_idx = 0
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        assert proc.stdout is not None
        while True:
            chunk = proc.stdout.read(65536)
            if not chunk:
                break
            buf.extend(chunk)
            while len(buf) >= frame_size:
                raw = bytes(buf[:frame_size])
                del buf[:frame_size]
                yield frame_idx / fps, np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3))
                frame_idx += 1
                if frame_idx >= max_frames:
                    return
    finally:
        proc.kill()
        proc.wait()


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------
def analyze_faces(
    source: str | Path,
    *,
    start: float = 0.0,
    end: float | None = None,
    fps: float = DETECT_FPS,
    max_frames: int = 2400,
) -> tuple[list[FrameFaces], str]:
    """Detect faces across a (clip) range at low framerate.

    Returns the per-frame detections and the detector backend used.
    """
    detector = FaceDetector()
    frames: list[FrameFaces] = []
    try:
        for t, rgb in sample_frames(source, start=start, end=end, fps=fps, max_frames=max_frames):
            try:
                faces = detector.detect(rgb)
            except Exception as exc:  # noqa: BLE001 - one bad frame must not kill the run
                log.debug("detect failed at %.2fs: %s", t, exc)
                faces = []
            frames.append(FrameFaces(t=t, faces=faces))
    finally:
        detector.close()

    found = sum(len(f.faces) for f in frames)
    log.info(
        "face analysis: %d sample(s), %d face(s) via %s",
        len(frames), found, detector.name,
    )
    return frames, detector.name


# --------------------------------------------------------------------------
# Tracking
# --------------------------------------------------------------------------
@dataclass
class _Tracked:
    t: float
    box: FaceBox


class Kalman1D:
    """Steady-state 1D Kalman filter (alpha-beta) for position + velocity.

    A plain EMA lags a continuously moving target — a speaker pacing across
    the frame would sit off-centre by the whole smoothing window. Tracking
    velocity lets the filter *predict* the next position, so a steady pan is
    followed with no residual offset while jitter is still damped.
    """

    def __init__(self, pos: float, *, proc_var: float = 2.0, meas_var: float = 1.0) -> None:
        self.pos = pos
        self.vel = 0.0
        self._p = proc_var
        self._q = proc_var
        self._r = meas_var

    def update(self, measurement: float, dt: float) -> float:
        """Fold in a measurement observed `dt` after the previous one."""
        if dt <= 0:
            return self.pos
        # Predict ahead using the velocity estimate.
        self.pos += self.vel * dt
        self._p += self._q * dt
        # Correct with the measurement.
        k = self._p / (self._p + self._r)
        innovation = measurement - self.pos
        self.pos += k * innovation
        self.vel += (k * innovation) / dt
        self._p *= 1.0 - k
        return self.pos


def _track_faces(frames: list[FrameFaces], max_gap: float = 2.0) -> list[_Tracked]:
    """Follow the most prominent face across samples.

    Faces link to the previous nearest detection (greedy nearest neighbour);
    gaps shorter than `max_gap` are bridged by holding the last known
    position so the crop does not snap back to centre on a momentary miss.
    Positions are smoothed with a Kalman filter so a moving speaker is
    followed without the lag a plain moving average would introduce.
    """
    track: list[_Tracked] = []
    last: FaceBox | None = None
    last_t = -1e9
    kx: Kalman1D | None = None
    ky: Kalman1D | None = None
    for ff in frames:
        box = None
        if ff.faces:
            if last is None:
                box = max(ff.faces, key=lambda b: b.area)
            else:
                box = min(
                    ff.faces,
                    key=lambda b: abs(b.cx - last.cx) + abs(b.cy - last.cy),
                )
        if box is not None:
            if kx is None:
                kx, ky = Kalman1D(box.cx), Kalman1D(box.cy)
            else:
                kx.update(box.cx, max(ff.t - last_t, 1e-3))
                ky.update(box.cy, max(ff.t - last_t, 1e-3))
            # The smoothed centre, but keep the measured size — framing needs
            # the real face extent, and it barely frame-to-frame.
            smoothed = FaceBox(
                x=box.x + (kx.pos - box.cx),
                y=box.y + (ky.pos - box.cy),
                w=box.w,
                h=box.h,
                confidence=box.confidence,
            )
            track.append(_Tracked(t=ff.t, box=smoothed))
            last = smoothed
            last_t = ff.t
        elif last is not None and ff.t - last_t <= max_gap:
            track.append(_Tracked(t=ff.t, box=last))
    return track


def _ema_smooth(values: list[float], alpha: float) -> list[float]:
    """Exponential moving average — removes per-frame jitter before keyframing."""
    if not values:
        return []
    out = [values[0]]
    for v in values[1:]:
        out.append(alpha * v + (1.0 - alpha) * out[-1])
    return out


def _deadzone_smooth(
    values: list[float],
    *,
    deadzone_px: float,
    snap_px: float = 0.0,
    smooth: float = CAM_SMOOTH,
) -> list[float]:
    """Camera positions that hold still inside a deadzone.

    A filter follows the detector and so reproduces its noise as motion; this
    instead keeps the camera put until the speaker leaves a zone around the
    current position, then closes a fraction of the remaining gap per sample.
    The result is long steady holds with smooth glides, and a snap on a jump
    big enough to be a shot cut rather than a slow pan across it.
    """
    if not values:
        return []
    # Start on the opening pose. A bad first detection is a large jump, which
    # the snap corrects on the very next sample.
    cam = values[0]
    out = [cam]
    for v in values[1:]:
        if snap_px > 0.0 and abs(v - cam) > snap_px:
            cam = v
        elif v > cam + deadzone_px:
            cam += (v - (cam + deadzone_px)) * smooth
        elif v < cam - deadzone_px:
            cam += (v - (cam - deadzone_px)) * smooth
        out.append(cam)
    return out


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------
def _target_window(width: int, height: int, out_w: int, out_h: int) -> tuple[float, float]:
    """Largest source window matching the output aspect ratio."""
    aspect = out_w / out_h
    if width / height > aspect:
        crop_h = float(height)
        crop_w = height * aspect
    else:
        crop_w = float(width)
        crop_h = width / aspect
    return crop_w, crop_h


def _zoomed_window(
    face_h: float,
    width: int,
    height: int,
    out_w: int,
    out_h: int,
) -> tuple[float, float]:
    """The crop window needed for the face to occupy TARGET_FACE_RATIO.

    When the source is landscape the largest 9:16 window is already as tall
    as the frame, so there is no freedom to position the face vertically —
    it sits wherever it sits. Zooming in (shrinking the window around the
    face) buys that freedom back: a smaller window can be placed anywhere,
    which is how the face lands in the upper third instead of mid-frame.
    """
    aspect = out_w / out_h
    # Crop height that makes the face occupy TARGET_FACE_RATIO of it.
    crop_h = face_h / TARGET_FACE_RATIO
    crop_w = crop_h * aspect
    max_w, max_h = _target_window(width, height, out_w, out_h)
    if crop_w > max_w:
        crop_w = max_w
        crop_h = crop_w / aspect
    if crop_h > max_h:
        crop_h = max_h
        crop_w = crop_h * aspect
    # Never zoom in past the point of upscaling: a tiny face in a huge frame
    # would produce a blurry crop, so cap the zoom at ZOOM_MAX times the max
    # window.
    min_w = max_w / ZOOM_MAX
    if crop_w < min_w:
        crop_w = min_w
        crop_h = crop_w / aspect
    return crop_w, crop_h


def _clamped(x: float, y: float, w: float, h: float, width: int, height: int) -> tuple[float, float]:
    """Keep the crop window fully inside the source frame."""
    x = min(max(x, 0.0), max(width - w, 0.0))
    y = min(max(y, 0.0), max(height - h, 0.0))
    return x, y


def _panel_window(
    width: int, height: int, out_w: int, out_h: int, n_panels: int
) -> tuple[float, float]:
    """Per-panel source window for a side-by-side composite.

    Each panel is a narrow portrait slice, so this is the tallest window whose
    width still matches the panel's aspect ratio — the full source height on a
    landscape input. Two speakers 970 px apart in a 1280x720 frame can never
    share a 405 px 9:16 window; giving each their own panel is the only way
    both stay on screen.
    """
    panel_ar = (out_w / n_panels) / out_h
    crop_h = float(height)
    crop_w = crop_h * panel_ar
    if crop_w > width:
        crop_w = float(width)
        crop_h = crop_w / panel_ar
    return crop_w, crop_h


def _panel_position(
    face_cx: float,
    face_cy: float,
    face_h: float,
    crop_w: float,
    crop_h: float,
    width: int,
    height: int,
) -> tuple[float, float]:
    """Centre a panel on its speaker.

    Horizontally the speaker sits mid-panel: unlike the single-crop layout
    there is no button rail to dodge, each panel is its own frame. Vertically
    the same headroom rule applies so the eyes land in the upper third.
    """
    x = face_cx - crop_w / 2.0
    y = (face_cy - face_h * 0.10) - crop_h * HEADROOM
    return _clamped(x, y, crop_w, crop_h, width, height)


def _crop_position(
    face_cx: float,
    face_cy: float,
    face_h: float,
    crop_w: float,
    crop_h: float,
    width: int,
    height: int,
) -> tuple[float, float]:
    """Position a crop window so the face sits in the safe zone.

    Horizontally the face is placed slightly left of centre so it clears the
    TikTok right-edge button rail (SAFE_RIGHT). Vertically the eyes land at
    `HEADROOM` from the top, clear of the camera notch and caption rail.
    """
    # Face at 45% of the crop width: centred, nudged off the button rail.
    x = face_cx - crop_w * 0.45
    # Eyes (roughly 10% of face height above the face centre) at HEADROOM.
    y = (face_cy - face_h * 0.10) - crop_h * HEADROOM
    return _clamped(x, y, crop_w, crop_h, width, height)


# --------------------------------------------------------------------------
# Layout builders
# --------------------------------------------------------------------------
def build_crop_plan(
    frames: list[FrameFaces],
    *,
    layout: str,
    width: int,
    height: int,
    out_w: int = OUT_WIDTH,
    out_height: int = OUT_HEIGHT,
    detector: str = "none",
) -> CropPlan:
    """Build a crop plan for one clip from its face samples.

    `layout="auto"` chooses from the detections: two consistent faces in a
    landscape source -> split_screen; a trackable face -> single_focus;
    none -> fit_blur.
    """
    layout = _resolve_layout(layout, frames, width, height)
    plan = CropPlan(
        layout=layout,
        source_width=width,
        source_height=height,
        out_width=out_w,
        out_height=out_height,
        detector=detector,
        face_count=max((len(f.faces) for f in frames), default=0),
    )

    if layout == "single_focus":
        _build_single_focus(frames, plan)
    elif layout == "split_screen":
        _build_split_screen(frames, plan)
    elif layout == "fill_zoom":
        _build_fill_zoom(frames, plan)
    else:
        _build_fit_blur(plan)

    if not plan.samples and not plan.is_composite:
        _build_fit_blur(plan)
        plan.layout = "fit_blur"
    return plan


def _resolve_layout(layout: str, frames: list[FrameFaces], width: int, height: int) -> str:
    if layout not in LAYOUTS:
        return "fit_blur"
    if layout != "auto":
        return layout

    if not frames:
        log.info("auto layout: no frames -> fit_blur")
        return "fit_blur"

    total_faces = sum(len(f.faces) for f in frames)
    if total_faces == 0:
        log.info("auto layout: no faces -> fit_blur")
        return "fit_blur"

    two_face = sum(1 for f in frames if len(f.faces) >= 2)
    if two_face / max(len(frames), 1) > 0.6 and width > height:
        log.info(
            "auto layout: %d/%d frames have 2 faces -> split_screen",
            two_face, len(frames),
        )
        return "split_screen"

    log.info("auto layout: %d face sample(s) -> single_focus", total_faces)
    return "single_focus"


def _decimate(samples: list[CropSample], max_keys: int = 24) -> list[CropSample]:
    """Reduce keyframe count for the ffmpeg expression.

    ffmpeg evaluates the crop expression per frame, so ~24 keyframes over a
    60-second clip cost nothing while 300 would bloat the command line.
    """
    if len(samples) <= max_keys:
        return samples
    step = (len(samples) - 1) / max(max_keys - 1, 1)
    idxs = sorted({int(round(i * step)) for i in range(max_keys)} | {len(samples) - 1})
    return [samples[i] for i in idxs]


# Panel windows are narrow (~202 px for a 150 px face, ~26 px of slack per
# side), so evenly spaced keyframes let the speaker drift to the panel edge
# between keys. Adaptive decimation instead keeps keyframes where the position
# actually bends, buying near-full-precision tracking for the same key budget.
PANEL_MAX_KEYS = 64
PANEL_KEY_TOL = 8.0  # source px


def _decimate_adaptive(
    samples: list[CropSample],
    max_keys: int = PANEL_MAX_KEYS,
    tol: float = PANEL_KEY_TOL,
) -> list[CropSample]:
    """Keep the fewest keyframes whose piecewise-linear path stays within `tol`.

    Greedy: repeatedly split whichever interval has the largest deviation from
    the straight line joining its kept endpoints, stopping once every sample is
    within `tol` or the budget is spent. Uniform spacing wastes keys on flat
    stretches and starves the fast ones.
    """
    n = len(samples)
    if n <= 2:
        return list(samples)
    kept = [0, n - 1]

    def worst_error(i: int) -> float:
        lo = max(k for k in kept if k < i)
        hi = min(k for k in kept if k > i)
        if hi <= lo:
            return 0.0
        f = (samples[i].t - samples[lo].t) / max(samples[hi].t - samples[lo].t, 1e-9)
        est_x = samples[lo].x + (samples[hi].x - samples[lo].x) * f
        est_y = samples[lo].y + (samples[hi].y - samples[lo].y) * f
        return max(abs(est_x - samples[i].x), abs(est_y - samples[i].y))

    while len(kept) < max_keys:
        errors = ((worst_error(i), i) for i in range(n) if i not in kept)
        err, idx = max(errors, key=lambda e: e[0])
        if err <= tol:
            break
        kept.append(idx)
        kept.sort()
    return [samples[i] for i in kept]


def _build_single_focus(frames: list[FrameFaces], plan: CropPlan) -> None:
    """Tight tracked crop around the dominant speaker."""
    width, height = plan.source_width, plan.source_height
    track = _track_faces(frames)
    if not track:
        return

    # Size the window to the face so the speaker fills the frame and there is
    # room to place them in the upper third even from a landscape source.
    mean_face_h = sum(p.box.h for p in track) / len(track)
    crop_w, crop_h = _zoomed_window(mean_face_h, width, height, plan.out_width, plan.out_height)

    times = [p.t for p in track]
    xs: list[float] = []
    ys: list[float] = []
    for p in track:
        x, y = _crop_position(p.box.cx, p.box.cy, p.box.h, crop_w, crop_h, width, height)
        xs.append(x)
        ys.append(y)

    # The tracker inherits the detector's noise; replaying it makes the crop
    # shake. Hold still inside a deadzone and glide when the speaker leaves it.
    xs = _deadzone_smooth(xs, deadzone_px=crop_w * CAM_DEADZONE, snap_px=width * CAM_SNAP)
    ys = _deadzone_smooth(ys, deadzone_px=crop_w * CAM_DEADZONE, snap_px=width * CAM_SNAP)

    plan.crop_w, plan.crop_h = crop_w, crop_h
    plan.samples = _decimate_adaptive(
        [CropSample(t=t, x=x, y=y) for t, x, y in zip(times, xs, ys)]
    )
    # Record the tracked face in output coordinates for the overlay stage.
    for p, x, y in zip(track, xs, ys):
        fx = (p.box.cx - x) / max(crop_w, 1e-6) * plan.out_width
        fy = (p.box.cy - y) / max(crop_h, 1e-6) * plan.out_height
        plan.focus.append({"t": p.t, "x": round(fx, 1), "y": round(fy, 1)})


def _build_split_screen(frames: list[FrameFaces], plan: CropPlan) -> None:
    """Two speakers, framed together.

    When the pair fits in one 9:16 window the single centred crop is used —
    both speakers in one shot at the highest resolution the source allows.
    When they are too far apart for that (a wide two-shot, which is the common
    podcast case) the plan becomes a side-by-side composite with one tracked
    panel per speaker, so neither is cut out of the frame.
    """
    width, height = plan.source_width, plan.source_height
    pairs = _pair_series(frames)
    if not pairs:
        _build_single_focus(frames, plan)
        plan.layout = "single_focus"
        return

    crop_w, crop_h = _target_window(width, height, plan.out_width, plan.out_height)

    if _pair_fits_one_window(pairs, crop_w):
        _build_split_single_window(pairs, plan, crop_w, crop_h)
    else:
        _build_split_composite(pairs, plan, crop_w, crop_h)


# A pair is framed in one window only when it fits with room to spare, so the
# composite is used for tight pairs too rather than only unreachable ones.
SPLIT_FIT_MARGIN = 1.25


def _pair_fits_one_window(pairs: list[_Pair], window_w: float) -> bool:
    """True when both speakers fit a single window in most frames.

    The span is measured face-edge to face-edge; the margin leaves breathing
    room either side instead of cropping the speakers at the window boundary.
    """
    if window_w <= 0:
        return False
    needed = window_w / SPLIT_FIT_MARGIN
    fits = sum(
        1 for p in pairs if (p.right.x + p.right.w - p.left.x) <= needed
    )
    return fits >= 0.7 * len(pairs)


@dataclass
class _Pair:
    """The two dominant speakers at one sample, ordered left to right."""

    t: float
    left: FaceBox
    right: FaceBox


def _pair_series(frames: list[FrameFaces], max_gap: float = 2.0) -> list[_Pair]:
    """The two dominant speakers per frame, ordered left to right.

    Speakers keep their side of the frame, so taking the two largest detections
    and sorting by centre-x preserves identity from sample to sample. A
    momentary miss holds the last known pair (up to `max_gap`) so a panel does
    not snap back to centre while its speaker is briefly occluded.
    """
    out: list[_Pair] = []
    last: _Pair | None = None
    last_t = -1e9
    for ff in frames:
        if len(ff.faces) >= 2:
            two = sorted(ff.faces, key=lambda b: b.w * b.h, reverse=True)[:2]
            pair = _Pair(ff.t, *sorted(two, key=lambda b: b.cx))
            out.append(pair)
            last, last_t = pair, ff.t
        elif last is not None and ff.t - last_t <= max_gap:
            out.append(_Pair(ff.t, last.left, last.right))
    return out


def _build_split_single_window(
    pairs: list[_Pair], plan: CropPlan, crop_w: float, crop_h: float
) -> None:
    """Centre one window on the gap between the two speakers."""
    width, height = plan.source_width, plan.source_height

    times: list[float] = []
    xs: list[float] = []
    ys: list[float] = []
    for pair in pairs:
        cx = (pair.left.cx + pair.right.cx) / 2.0
        cy = (pair.left.cy + pair.right.cy) / 2.0
        face_h = max(pair.left.h, pair.right.h)
        x, y = _crop_position(cx, cy, face_h, crop_w, crop_h, width, height)
        times.append(pair.t)
        xs.append(x)
        ys.append(y)

    xs = _deadzone_smooth(xs, deadzone_px=crop_w * CAM_DEADZONE, snap_px=width * CAM_SNAP)
    ys = _deadzone_smooth(ys, deadzone_px=crop_w * CAM_DEADZONE, snap_px=width * CAM_SNAP)
    plan.crop_w, plan.crop_h = crop_w, crop_h
    plan.samples = _decimate_adaptive(
        [CropSample(t=t, x=x, y=y) for t, x, y in zip(times, xs, ys)]
    )


def _build_split_composite(
    pairs: list[_Pair], plan: CropPlan, window_w: float, window_h: float
) -> None:
    """One tracked portrait panel per speaker, side by side on the canvas."""
    width, height = plan.source_width, plan.source_height
    n_panels = 2
    crop_w, crop_h = _panel_window(
        width, height, plan.out_width, plan.out_height, n_panels
    )
    panel_out_w = plan.out_width / n_panels
    panel_out_h = float(plan.out_height)

    # Report the single-window geometry for diagnostics; the panels own the
    # actual framing.
    plan.crop_w, plan.crop_h = window_w, window_h

    for index, speaker in ((0, "left"), (1, "right")):
        times: list[float] = []
        cxs: list[float] = []
        cys: list[float] = []
        hs: list[float] = []
        for pair in pairs:
            box = pair.left if speaker == "left" else pair.right
            times.append(pair.t)
            cxs.append(box.cx)
            cys.append(box.cy)
            hs.append(box.h)
        # Each panel has only ~26 px of slack around its speaker, so detector
        # jitter reads as a shaking panel. Hold the panel centred inside a
        # deadzone and glide after the speaker when they leave it.
        deadzone = crop_w * CAM_DEADZONE
        snap = width * CAM_SNAP
        cxs = _deadzone_smooth(cxs, deadzone_px=deadzone, snap_px=snap)
        cys = _deadzone_smooth(cys, deadzone_px=deadzone, snap_px=snap)
        samples = [
            CropSample(t=t, x=x, y=y)
            for t, cx, cy, face_h in zip(times, cxs, cys, hs)
            for x, y in (_panel_position(cx, cy, face_h, crop_w, crop_h, width, height),)
        ]
        plan.panels.append(
            PanelPlan(
                crop_w=crop_w,
                crop_h=crop_h,
                canvas_x=index * panel_out_w,
                canvas_y=0.0,
                out_w=panel_out_w,
                out_h=panel_out_h,
                samples=_decimate_adaptive(samples),
            )
        )

    # Panel centres in output coordinates, for the overlay stage.
    for pair in pairs:
        for index in range(n_panels):
            plan.focus.append(
                {
                    "t": pair.t,
                    "x": round(index * panel_out_w + panel_out_w / 2.0, 1),
                    "y": round(panel_out_h / 2.0, 1),
                }
            )


def _build_fill_zoom(frames: list[FrameFaces], plan: CropPlan) -> None:
    """Centre crop with a slow push-in — premium when no face is tracked.

    The window stays wide (it does not chase the speaker the way
    `single_focus` does), but it drifts gently so the shot is never frozen.
    `zoom_start`/`zoom_end` record the push-in for the renderer to apply,
    since ffmpeg's crop filter cannot resize its window mid-stream.
    """
    width, height = plan.source_width, plan.source_height
    crop_w, crop_h = _target_window(width, height, plan.out_width, plan.out_height)

    track = _track_faces(frames)
    # Per-frame focus: the tracked speaker when available, else frame centre.
    focus: list[tuple[float, float, float]] = []
    for ff, tr in _zip_track(frames, track):
        if tr is not None:
            focus.append((ff.t, tr.box.cx, tr.box.cy))
        else:
            focus.append((ff.t, width / 2.0, height / 2.0))
    if not focus:
        focus.append((0.0, width / 2.0, height / 2.0))

    times = [f[0] for f in focus]
    xs: list[float] = []
    ys: list[float] = []
    for _, cx, cy in focus:
        x, y = _crop_position(cx, cy, height * 0.15, crop_w, crop_h, width, height)
        xs.append(x)
        ys.append(y)

    xs = _ema_smooth(xs, alpha=0.30)
    ys = _ema_smooth(ys, alpha=0.30)
    plan.crop_w, plan.crop_h = crop_w, crop_h
    plan.samples = _decimate([CropSample(t=t, x=x, y=y) for t, x, y in zip(times, xs, ys)])
    plan.zoom_start, plan.zoom_end = 1.0, 1.12


def _zip_track(frames: list[FrameFaces], track: list[_Tracked]):
    """Pair each frame with its tracked box, or None where there is none.

    The track omits frames before the first detection and any gap longer
    than `max_gap`; those map to None so the caller can fall back.
    """
    by_t = {round(p.t, 3): p for p in track}
    for ff in frames:
        yield ff, by_t.get(round(ff.t, 3))


def _build_fit_blur(plan: CropPlan) -> None:
    """Full frame visible with a blurred vertical background fill.

    The renderer scales the source to fit the canvas width and blurs a copy
    behind it, so the plan only needs one static centre-crop keyframe — the
    `crop` here just defines the visible region.
    """
    width, height = plan.source_width, plan.source_height
    crop_w, crop_h = _target_window(width, height, plan.out_width, plan.out_height)
    x, y = (width - crop_w) / 2.0, (height - crop_h) / 2.0
    plan.crop_w, plan.crop_h = crop_w, crop_h
    plan.samples = [CropSample(t=0.0, x=x, y=y)]


# --------------------------------------------------------------------------
# Serialization
# --------------------------------------------------------------------------
def save_plan(plan: CropPlan, path: str | Path) -> Path:
    """Write a crop plan to disk as JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def load_plan(path: str | Path) -> CropPlan:
    """Load a crop plan from disk."""
    return CropPlan.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# --------------------------------------------------------------------------
# ffmpeg expression generation
# --------------------------------------------------------------------------
def _fmt(v: float) -> str:
    """Format a float for an ffmpeg expression (no trailing zeros)."""
    s = f"{v:.3f}".rstrip("0").rstrip(".")
    return s if s else "0"


def _interp(samples: list[CropSample], key: str, var: str = "t") -> str:
    """Linear interpolation of one crop coordinate between keyframes.

    Emits nested `if(gt(t,k), higher, lower)` chains where each branch computes
    `v0 + (v1-v0)*(t-t0)/(t1-t0)` inside its interval. Step-hold interpolation
    is simpler but visibly stutters on a fast-moving speaker: with 5 Hz
    sampling a pan of 40 px between samples would jump twice a second.
    Linear interpolation keeps the crop centred on the face at every frame
    while the expression stays a short, cheap nested chain for ffmpeg.
    """
    if not samples:
        return "0"
    if len(samples) == 1:
        return _fmt(getattr(samples[0], key))

    # Build from the last keyframe outward: for t beyond the final keyframe
    # the value is held, which ffmpeg reaches through the nested branches.
    expr = _fmt(getattr(samples[-1], key))
    for i in range(len(samples) - 2, -1, -1):
        cur, nxt = samples[i], samples[i + 1]
        va = getattr(cur, key)
        vb = getattr(nxt, key)
        dt = nxt.t - cur.t
        if dt <= 0:
            expr = f"if(gt({var},{_fmt(nxt.t)}),{expr},{_fmt(va)})"
            continue
        slope = (vb - va) / dt
        # Value inside [cur.t, nxt.t]; after nxt.t the outer branch takes over.
        linear = f"{_fmt(va)}+({_fmt(slope)})*({var}-{_fmt(cur.t)})"
        # Clamp below the interval start in case t < cur.t.
        piece = f"if(gt({var},{_fmt(cur.t)}),{linear},{_fmt(va)})"
        expr = f"if(gt({var},{_fmt(nxt.t)}),{expr},{piece})"
    return expr


def crop_filter(plan: CropPlan, *, var: str = "t") -> str:
    """The ffmpeg `crop` filter string for a plan.

    Only x and y animate (ffmpeg re-evaluates them per frame); the window size
    is fixed because ffmpeg configures `crop`'s w/h once.
    """
    w = _fmt(plan.crop_w) if plan.crop_w else f"ih*{plan.out_width / plan.out_height:.6f}"
    h = _fmt(plan.crop_h) if plan.crop_h else f"iw*{plan.out_height / plan.out_width:.6f}"
    if plan.is_static:
        s = plan.samples[0] if plan.samples else CropSample(0, 0, 0)
        return f"crop={w}:{h}:{_fmt(s.x)}:{_fmt(s.y)}"
    return f"crop={w}:{h}:x='{_interp(plan.samples, 'x', var)}':y='{_interp(plan.samples, 'y', var)}'"


def scale_filter(plan: CropPlan) -> str:
    """Scale the cropped window to the exact output canvas."""
    return f"scale={plan.out_width}:{plan.out_height}:flags=lanczos"


def fit_blur_filters(plan: CropPlan) -> list[str]:
    """The full filter chain for the `fit_blur` layout.

    Splits the stream, blurs and scales one copy to fill the canvas, then
    overlays the sharp fit-to-width original centred vertically.
    """
    w, h = plan.out_width, plan.out_height
    src_ar = plan.source_width / max(plan.source_height, 1)
    out_ar = w / h
    # The foreground keeps its aspect; the blurred background fills the rest.
    fg_h = h if src_ar >= out_ar else w / max(src_ar, 1e-6)
    fg_w = w if src_ar <= out_ar else h * src_ar
    return [
        "split=2[bg][fg]",
        f"[bg]gblur=sigma=30,scale={w}:{h}:force_original_aspect_ratio=increase,"
        f"crop={w}:{h}[bgfill]",
        f"[fg]scale={_fmt(fg_w)}:{_fmt(fg_h)}:flags=lanczos[fgfit]",
        f"[bgfill][fgfit]overlay=(W-w)/2:(H-h)/2",
    ]


def split_screen_filters(plan: CropPlan) -> list[str]:
    """The full filter chain for a `split_screen` composite.

    One stream per speaker panel: each is cropped around its tracked speaker
    and scaled to its slice of the canvas, then the panels are stacked side by
    side. ffmpeg's `crop` re-evaluates x/y per frame, so the per-panel tracking
    from `PanelPlan.samples` stays smooth at full fps.
    """
    panels = plan.panels
    if not panels:
        raise ValueError("split_screen plan has no panels")
    n = len(panels)
    labels = [f"src{i}" for i in range(n)]
    scaled = [f"pan{i}" for i in range(n)]
    steps = ["split=%d%s" % (n, "".join(f"[{l}]" for l in labels))]
    for label, out_label, panel in zip(labels, scaled, panels):
        x_expr = _interp(panel.samples, "x")
        y_expr = _interp(panel.samples, "y")
        steps.append(
            f"[{label}]crop={_fmt(panel.crop_w)}:{_fmt(panel.crop_h)}:"
            f"x='{x_expr}':y='{y_expr}',"
            f"scale={int(round(panel.out_w))}:{int(round(panel.out_h))}:flags=lanczos"
            f"[{out_label}]"
        )
    steps.append("".join(f"[{l}]" for l in scaled) + f"hstack=inputs={n}")
    # Thin dividers between panels so the two speakers read as separate shots.
    dividers = ",".join(
        f"drawbox=x={int(round(i * plan.out_width / n)) - 1}:y=0:w=2"
        f":h={plan.out_height}:color=black@0.35:t=fill"
        for i in range(1, n)
    )
    if dividers:
        steps.append(dividers)
    return steps
