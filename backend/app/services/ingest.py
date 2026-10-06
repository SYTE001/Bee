"""Ingest service: accept uploads and download YouTube URLs via yt-dlp."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from app.core.config import settings
from app.core.logging import get_logger
from app.services import media
from app.services.storage import storage

log = get_logger("ingest")

ALLOWED_UPLOAD_EXTS = {
    ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv", ".flv",
}


@dataclass
class IngestResult:
    path: Path
    title: str
    duration: float
    width: int
    height: int
    fps: float
    has_video: bool
    has_audio: bool


class IngestError(RuntimeError):
    pass


def ingest_upload(project_id: str, filename: str, data: bytes) -> IngestResult:
    """Validate and store an uploaded file, then probe it."""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_UPLOAD_EXTS:
        raise IngestError(
            f"unsupported file type '{ext}'. allowed: {', '.join(sorted(ALLOWED_UPLOAD_EXTS))}"
        )
    size_gb = len(data) / (1024 ** 3)
    if size_gb > settings.MAX_UPLOAD_GB:
        raise IngestError(
            f"file too large ({size_gb:.2f}GB > {settings.MAX_UPLOAD_GB}GB limit)"
        )
    path = storage.save_upload(project_id, filename, data)
    return _probe_and_report(path, title=Path(filename).stem)


def ingest_youtube(project_id: str, url: str) -> IngestResult:
    """Download a video with yt-dlp. Requires `yt-dlp` on PATH."""
    if shutil.which("yt-dlp") is None:
        raise IngestError(
            "yt-dlp is not installed. install it with `pip install yt-dlp` "
            "or upload the video file directly."
        )
    outdir = storage.project_dir(project_id)
    tmpl = str(outdir / "source_%(title).120s.%(ext)s")
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--no-warnings",
        "--newline",
        "-f", settings.YT_DLP_FORMAT,
        "-o", tmpl,
        "--merge-output-format", "mp4",
        url,
    ]
    log.info("yt-dlp downloading %s", url)
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=3600, check=False)
    except subprocess.TimeoutExpired as exc:
        raise IngestError("download timed out (>1h)") from exc
    if proc.returncode != 0:
        err = (proc.stderr or b"").decode("utf-8", "replace")[-800:]
        raise IngestError(f"yt-dlp failed: {err}")

    # Find the produced file: newest media file in the project dir
    candidates = sorted(
        (f for f in outdir.iterdir() if f.suffix.lower() in ALLOWED_UPLOAD_EXTS),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise IngestError("yt-dlp produced no media file")
    return _probe_and_report(candidates[0], title=candidates[0].stem)


def _probe_and_report(path: Path, *, title: str) -> IngestResult:
    if not path.exists() or path.stat().st_size == 0:
        raise IngestError(f"empty or missing file: {path.name}")
    info = media.probe(path)
    if not info.has_audio:
        raise IngestError(
            "no audio stream found — ClipForge needs speech audio to work"
        )
    return IngestResult(
        path=path,
        title=title[:255],
        duration=info.duration,
        width=info.width,
        height=info.height,
        fps=info.fps,
        has_video=info.has_video,
        has_audio=info.has_audio,
    )
