"""FFmpeg wrappers: probe metadata, audio extraction, normalization.

Every call goes through `_run` which logs the command, enforces a timeout,
and raises a `MediaError` with ffmpeg's stderr on failure.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from app.core.config import settings
from app.core.logging import get_logger

log = get_logger("media")


class MediaError(RuntimeError):
    """Raised when an ffmpeg/ffprobe operation fails."""


@dataclass
class MediaInfo:
    duration: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    has_video: bool = False
    has_audio: bool = False
    video_codec: str = ""
    audio_codec: str = ""
    sample_rate: int = 0
    channels: int = 0
    metadata: dict = field(default_factory=dict)

    @property
    def is_vertical(self) -> bool:
        return self.height > self.width


def _run(cmd: list[str], timeout: int = 1800) -> tuple[int, str, str]:
    log.debug("run: %s", " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:  # pragma: no cover - env dependent
        raise MediaError(f"binary not found: {cmd[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise MediaError(f"timeout after {timeout}s: {' '.join(cmd[:3])}") from exc
    if proc.returncode != 0:
        tail = (proc.stderr or b"").decode("utf-8", "replace")[-1200:]
        raise MediaError(f"{cmd[0]} failed rc={proc.returncode}: {tail}")
    return (
        proc.returncode,
        proc.stdout.decode("utf-8", "replace"),
        proc.stderr.decode("utf-8", "replace"),
    )


def ffprobe_available() -> bool:
    return shutil.which("ffprobe") is not None and shutil.which("ffmpeg") is not None


def probe(path: str | Path) -> MediaInfo:
    """Probe a media file with ffprobe."""
    p = str(path)
    cmd = [
        "ffprobe",
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        p,
    ]
    rc, out, _ = _run(cmd)
    if rc != 0:
        raise MediaError(f"ffprobe failed for {p}")
    try:
        data = json.loads(out)
    except json.JSONDecodeError as exc:
        raise MediaError(f"bad ffprobe json for {p}") from exc

    info = MediaInfo()
    fmt = data.get("format", {})
    try:
        info.duration = float(fmt.get("duration", 0.0) or 0.0)
    except (TypeError, ValueError):
        info.duration = 0.0

    for stream in data.get("streams", []):
        ctype = stream.get("codec_type")
        if ctype == "video" and not info.has_video:
            info.has_video = True
            info.video_codec = stream.get("codec_name", "")
            info.width = int(stream.get("width", 0) or 0)
            info.height = int(stream.get("height", 0) or 0)
            # avg_frame_rate is like "30000/1001"
            fr = stream.get("avg_frame_rate", "0/1")
            try:
                num, _, den = fr.partition("/")
                info.fps = float(num) / max(float(den or 1), 1e-9)
            except (TypeError, ValueError, ZeroDivisionError):
                info.fps = 0.0
        elif ctype == "audio" and not info.has_audio:
            info.has_audio = True
            info.audio_codec = stream.get("codec_name", "")
            info.sample_rate = int(stream.get("sample_rate", 0) or 0)
            info.channels = int(stream.get("channels", 0) or 0)

    info.metadata = {k: str(v) for k, v in (fmt.get("tags", {}) or {}).items()}
    log.info(
        "probe %s: %.1fs %dx%d@%.2ffps video=%s audio=%s",
        Path(p).name,
        info.duration,
        info.width,
        info.height,
        info.fps,
        info.has_video,
        info.has_audio,
    )
    return info


def extract_audio(
    source: str | Path,
    out_path: str | Path,
    *,
    sample_rate: int = 16000,
    mono: bool = True,
) -> Path:
    """Extract audio to 16-bit PCM WAV (whisper's preferred format)."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", str(source),
        "-vn",
        "-ac", "1" if mono else "2",
        "-ar", str(sample_rate),
        "-c:a", "pcm_s16le",
        str(out),
    ]
    _run(cmd, timeout=3600)
    if not out.exists():
        raise MediaError(f"audio extraction produced no file: {out}")
    log.info("extracted audio -> %s (%d KB)", out.name, out.stat().st_size // 1024)
    return out


def normalize_loudness(
    source: str | Path,
    out_path: str | Path,
    *,
    target_lufs: float | None = None,
) -> Path:
    """Two-pass loudnorm to broadcast-standard loudness (-14 LUFS by default)."""
    target = settings.TARGET_LUFS if target_lufs is None else target_lufs
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Pass 1: measure
    cmd = [
        "ffmpeg", "-y", "-v", "error", "-i", str(source),
        "-af", f"loudnorm=I={target}:TP=-1.5:LRA=11:print_format=json",
        "-f", "null", "-",
    ]
    _, _, err = _run(cmd, timeout=1800)
    stats = _parse_loudnorm_stats(err)
    if stats:
        af = (
            f"loudnorm=I={target}:TP=-1.5:LRA=11:"
            f"measured_I={stats['input_i']}:measured_TP={stats['input_tp']}:"
            f"measured_LRA={stats['input_lra']}:measured_thresh={stats['input_thresh']}:"
            f"offset={stats['target_offset']}:linear=true:print_format=summary"
        )
    else:  # pragma: no cover - measurement fallback
        log.warning("loudnorm measurement unavailable, using single pass")
        af = f"loudnorm=I={target}:TP=-1.5:LRA=11"

    # Pass 2: apply
    cmd = [
        "ffmpeg", "-y", "-v", "error", "-i", str(source),
        "-af", af,
        "-c:a", "pcm_s16le", "-ar", "48000", "-ac", "2",
        str(out),
    ]
    _run(cmd, timeout=3600)
    log.info("normalized audio -> %s (target %.1f LUFS)", out.name, target)
    return out


def _parse_loudnorm_stats(stderr: str) -> dict[str, str] | None:
    """Extract the JSON block ffmpeg prints to stderr during loudnorm."""
    marker = "Parsed_loudnorm"
    idx = stderr.find(marker)
    if idx == -1:
        return None
    start = stderr.find("{", idx)
    end = stderr.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(stderr[start : end + 1])
    except json.JSONDecodeError:
        return None


def cut_segment(
    source: str | Path,
    out_path: str | Path,
    start: float,
    end: float,
    *,
    video_filters: list[str] | None = None,
    audio_filters: list[str] | None = None,
    encoder: str = "libx264",
    extra_args: list[str] | None = None,
) -> Path:
    """Cut [start,end] from source and encode. Stage 3/5 refine the filters.

    Encodes into a sibling temp file and renames into place, so the destination
    only ever exists as a complete, valid file. Two renders racing for the same
    path (a per-clip render fired while a batch render is running) would
    otherwise interleave writes and produce a container whose chunk offsets
    point outside its own media data.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    duration = max(end - start, 0.05)

    # Temp file lives in the destination directory: os.replace is atomic only
    # within one filesystem, and mkstemp would put us on the system temp mount.
    # The marker goes before the extension so ffmpeg still sees a ".mp4" and
    # can pick a muxer from it.
    tmp = out.with_name(f"{out.stem}.tmp{os.getpid()}{out.suffix}")
    try:
        cmd = ["ffmpeg", "-y", "-v", "error"]
        cmd += ["-ss", f"{start:.3f}", "-i", str(source), "-t", f"{duration:.3f}"]

        vf = video_filters or []
        af = audio_filters or []
        if vf:
            cmd += ["-vf", ",".join(vf)]
        if af:
            cmd += ["-af", ",".join(af)]
        cmd += ["-c:v", encoder, "-pix_fmt", "yuv420p"]
        cmd += extra_args or []
        if encoder.startswith("libx264") and "-preset" not in (extra_args or []):
            cmd += ["-crf", str(settings.H264_CRF), "-preset", settings.H264_PRESET]
        cmd += ["-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(tmp)]

        _run(cmd, timeout=7200)
        if not tmp.exists():
            raise MediaError(f"cut produced no file: {tmp}")
        os.replace(tmp, out)
    except BaseException:
        # A half-written temp file must not linger and be served as a clip.
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:  # pragma: no cover
            pass
        raise
    return out


def extract_thumbnail(source: str | Path, out_path: str | Path, at: float = 0.5) -> Path:
    """Grab a single JPEG frame at time `at` (default: 0.5s)."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-ss", f"{max(at, 0.0):.3f}", "-i", str(source),
        "-frames:v", "1", "-q:v", "3",
        str(out),
    ]
    _run(cmd, timeout=600)
    return out


@contextmanager
def with_tempfile(suffix: str, prefix: str = "clipforge_"):
    """Context manager yielding a temp path; cleaned up on exit."""
    fd, name = tempfile.mkstemp(suffix=suffix, prefix=prefix)
    os.close(fd)
    path = Path(name)
    try:
        yield path
    finally:
        if path.exists():
            try:
                path.unlink()
            except OSError:  # pragma: no cover
                pass
