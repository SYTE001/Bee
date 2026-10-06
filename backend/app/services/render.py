"""Final clip rendering.

Takes a ranked clip and produces a 1080x1920 vertical video: the reframing
plan from Fase 3 (crop/fit-blur + face tracking), burned-in animated
subtitles from Fase 4, punch-in zoom, loudness normalization to broadcast
standard, and the optional watermark / opening title.

Everything is built as plain ffmpeg filter chains so no frames touch Python.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from app.core.config import settings
from app.core.logging import get_logger
from app.db.models.clip import Clip
from app.db.models.project import Project
from app.db.session import SessionLocal
from app.services import events, media, subtitle
from app.services.artifacts import ArtifactStore
from app.services.storage import storage

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

log = get_logger("render")

# Emojis picked per clip category when a style enables them (auto emoji).
CATEGORY_EMOJIS: dict[str, list[str]] = {
    "funny": ["😂", "🤣", "🔥", "💀"],
    "serious": ["🧠", "📖", "🎭"],
    "insight": ["💡", "🧠", "📈", "🎯"],
    "emotional": ["❤️", "🥹", "✨"],
    "controversial": ["⚡", "🔥", "🗣️"],
    "story": ["🎬", "🎧", "📖"],
    "actionable": ["✅", "🚀", "💡", "📝"],
}
_DEFAULT_EMOJIS = ["✨", "🔥", "💡"]

# Fonts tried in order for drawtext overlays (watermark / hook title).
_FONT_CANDIDATES = (
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
)


@dataclass(frozen=True)
class RenderResult:
    """Outcome of one clip render."""

    clip_id: str
    path: Path
    duration: float


# --------------------------------------------------------------------------
# Render serialization
# --------------------------------------------------------------------------
# A clip can be rendered from the batch pass and from the single-clip endpoint
# at the same time. Both would ffmpeg into the same path, and interleaved
# writes leave a container whose chunk offsets point outside its media data
# (observed as a duplicated moov and a file that freezes a few seconds in).
# One lock per clip id makes those renders queue instead of race.
_clip_locks: dict[str, threading.RLock] = defaultdict(threading.RLock)
_clip_locks_guard = threading.Lock()


def _clip_lock(clip_id: str) -> threading.RLock:
    with _clip_locks_guard:
        return _clip_locks[clip_id]


# --------------------------------------------------------------------------
# Encoder selection
# --------------------------------------------------------------------------
_nvenc_usable: bool | None = None


def _nvenc_encoder_name(codec: str) -> str | None:
    """The NVENC encoder for a codec, or None if that codec has none."""
    return {"h264": "h264_nvenc", "h265": "hevc_nvenc"}.get(codec)


def nvenc_usable() -> bool:
    """True if an NVIDIA encoder can actually encode a frame here.

    Encoders listed by `ffmpeg -encoders` can still fail at runtime when there
    is no GPU or the driver is too old, so this does a real one-frame encode.
    The result is cached for the process.
    """
    global _nvenc_usable
    if _nvenc_usable is not None:
        return _nvenc_usable
    enc = _nvenc_encoder_name(settings.VIDEO_CODEC if settings.VIDEO_CODEC != "auto" else "h264")
    if enc is None:
        _nvenc_usable = False
        return False
    probe = [
        "ffmpeg", "-y", "-v", "error", "-hide_banner",
        "-f", "lavfi", "-i", "color=c=black:s=64x64:d=0.2:r=5",
        "-c:v", enc, "-frames:v", "1", "-f", "null", "-",
    ]
    try:
        proc = subprocess.run(probe, capture_output=True, timeout=60)
        _nvenc_usable = proc.returncode == 0
    except Exception:  # noqa: BLE001 - probe failures mean "no NVENC"
        _nvenc_usable = False
    if _nvenc_usable:
        log.info("NVENC encoder %s is usable; rendering on GPU", enc)
    else:
        log.info("NVENC unavailable; rendering on CPU (libx264/libx265)")
    return _nvenc_usable


def resolve_encoder() -> str:
    """Pick the video encoder: NVENC when available and requested, else x264."""
    codec = "h264" if settings.VIDEO_CODEC == "auto" else settings.VIDEO_CODEC
    nvenc = _nvenc_encoder_name(codec)
    if nvenc and nvenc_usable():
        return nvenc
    return {"h264": "libx264", "h265": "libx265"}[codec]


def encoder_args(encoder: str) -> list[str]:
    """Encoder-specific CLI flags for `media.cut_segment`'s `extra_args`."""
    if encoder in ("h264_nvenc", "hevc_nvenc"):
        tag = "avc1" if encoder == "h264_nvenc" else "hvc1"
        return [
            "-preset", "p5",
            "-rc", "vbr",
            "-cq", str(settings.NVENC_CQ),
            "-b:v", "0",
            "-maxrate", settings.VIDEO_BITRATE_MAX,
            "-bufsize", settings.VIDEO_BITRATE_BUF,
            "-spatial-aq", "1",
            "-tag:v", tag,
        ]
    crf = settings.H264_CRF if encoder == "libx264" else settings.H265_CRF
    preset = settings.H264_PRESET if encoder == "libx264" else settings.H265_PRESET
    args = ["-crf", str(crf), "-preset", preset]
    if encoder == "libx265":
        args += ["-tag:v", "hvc1"]
    return args


# --------------------------------------------------------------------------
# Filter escaping
# --------------------------------------------------------------------------
def _filter_path(path: str | Path) -> str:
    """Escape a filesystem path for use inside an ffmpeg filter argument."""
    return str(path).replace("\\", "/").replace(":", "\\:")


def _drawtext_escape(text: str) -> str:
    """Escape text for drawtext (backslash, colon, quote, percent)."""
    out = []
    for ch in str(text):
        if ch in "\\:'%":
            out.append("\\")
        out.append(ch)
    return "".join(out)


def _find_font() -> str | None:
    for candidate in _FONT_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    return None


# --------------------------------------------------------------------------
# Filter graph
# --------------------------------------------------------------------------
def _zoom_filter(plan, source_fps: float, duration: float) -> str | None:
    """Punch-in zoom for plans that carry one (fill_zoom layout).

    `zoompan` with `d=1` emits one frame per input frame; `z` eases from
    `zoom_start` to `zoom_end` across the clip. Runs at the source framerate
    so the later `fps` filter owns the frame-rate conversion.
    """
    z0, z1 = plan.zoom_start, plan.zoom_end
    if max(z0, z1) <= 1.02 or duration <= 0.05:
        return None
    fps = round(source_fps, 2) or 30.0
    w, h = plan.out_width, plan.out_height
    return (
        f"zoompan=z='{_fmt(z0)}+({_fmt(z1 - z0)})*in_time/{_fmt(duration)}':"
        f"d=1:s={w}x{h}:fps={fps}:"
        f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
    )


def _fmt(v: float) -> str:
    return f"{v:.4f}".rstrip("0").rstrip(".")


def build_video_filters(
    plan,
    *,
    ass_path: str | Path | None,
    target_fps: int,
    source_fps: float,
    duration: float,
    hook_text: str = "",
    watermark: str = "",
) -> list[str]:
    """Compose the video filter chain for a clip.

    Each returned item is one comma-joined filter chain step; the caller joins
    them in order. Reframing first, then zoom, subtitles, overlays, fps.
    """
    from app.services import reframe

    steps: list[str] = []
    if plan.layout == "fit_blur":
        steps.append(",".join(reframe.fit_blur_filters(plan)))
    elif plan.is_composite:
        steps.append(",".join(reframe.split_screen_filters(plan)))
    else:
        steps.append(reframe.crop_filter(plan))
        steps.append(reframe.scale_filter(plan))

    zoom = _zoom_filter(plan, source_fps, duration)
    if zoom:
        steps.append(zoom)

    if ass_path is not None:
        steps.append(f"subtitles='{_filter_path(ass_path)}'")

    font = _find_font()
    if font and hook_text:
        sec = settings.HOOK_TITLE_SECONDS
        steps.append(
            "drawtext="
            f"fontfile='{_filter_path(font)}':"
            f"text='{_drawtext_escape(hook_text)}':"
            "x=(w-tw)/2:y=h*0.045:"
            "fontsize=64:fontcolor=white:shadowcolor=black@0.65:shadowx=3:shadowy=3:"
            f"line_spacing=12:enable='between(t,0,{_fmt(sec)})'"
        )
    if font and watermark:
        steps.append(
            "drawtext="
            f"fontfile='{_filter_path(font)}':"
            f"text='{_drawtext_escape(watermark)}':"
            "x=w-tw-34:y=h*0.86:"
            "fontsize=36:fontcolor=white@0.75:shadowcolor=black@0.6:shadowx=2:shadowy=2"
        )
    return steps


def _frame_rate_filter(target_fps: int, source_fps: float) -> str:
    if settings.FRAME_INTERPOLATION and target_fps > source_fps + 0.5:
        # mi_mode=mci is real motion interpolation: better motion, far slower.
        return (
            f"minterpolate=mi_mode=mci:mc_mode=aobmc:vsbmc=1:fps={target_fps}"
        )
    return f"fps=fps={target_fps}"


# --------------------------------------------------------------------------
# Plan resolution
# --------------------------------------------------------------------------
def _source_sig(info: media.MediaInfo) -> str:
    return f"{info.width}x{info.height}"


def resolve_plan(
    db: Session,
    clip: Clip,
    project: Project,
    store: ArtifactStore,
    info: media.MediaInfo,
):
    """The crop plan for a clip, rebuilding it when the cached one is stale.

    Plans are looked up by clip id, then by (range, layout) so a clip edited
    in the editor still finds the analysis for its time range. Anything whose
    source dimensions changed, or that has no plan at all (e.g. a clip created
    after reframing), is re-analyzed from the source video.
    """
    from app.pipelines.video_pipeline import ARTIFACT_REFRAME
    from app.services import reframe

    plans: dict[str, dict] = {}
    if store.has(ARTIFACT_REFRAME):
        try:
            raw = store.path_of(ARTIFACT_REFRAME)
            if raw is not None:
                plans = json.loads(raw.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            plans = {}

    wanted = _source_sig(info)
    cached = plans.get(clip.id) if clip.id in plans else None
    if cached is None:
        key = reframe.plan_cache_key(clip.start, clip.end, clip.layout_mode or "auto")
        for value in plans.values():
            if isinstance(value, dict) and value.get("_cache_key") == key:
                cached = value
                break
    # A plan from an older build is not reusable: the schema changed (e.g.
    # split_screen gained per-speaker panels) and replaying it would render
    # the old framing. Fall through to a rebuild instead.
    if isinstance(cached, dict) and cached.get("_cache_version") != reframe.PLAN_VERSION:
        log.info(
            "cached plan for clip %s is from an older schema; rebuilding", clip.id
        )
        cached = None
    if isinstance(cached, dict) and cached.get("_source_sig") == wanted:
        try:
            return reframe.CropPlan.from_dict(cached)
        except (KeyError, TypeError, ValueError):
            log.warning("cached plan for clip %s was malformed; rebuilding", clip.id)

    src = ARTIFACT_SOURCE_PATH(store)
    if src is None:
        raise RuntimeError(f"source artifact missing for project {project.id}")
    frames, detector = reframe.analyze_faces(src, start=clip.start, end=clip.end)
    plan = reframe.build_crop_plan(
        frames,
        layout=clip.layout_mode or "auto",
        width=info.width,
        height=info.height,
        detector=detector,
    )
    log.info(
        "rebuilt crop plan for clip %s (%s, %d samples)",
        clip.id, plan.layout, len(plan.samples),
    )
    _persist_plan(db, clip, store, plans, plan)
    return plan


def _persist_plan(db, clip, store, plans: dict[str, dict], plan) -> None:  # type: ignore[no-untyped-def]
    """Write a rebuilt plan back so the next render does not re-analyse.

    The face pass is the expensive part of reframing; rebuilding a plan and
    then throwing it away would make every forced re-render pay for it twice.
    Best-effort: a failing write only costs us the cache.
    """
    from app.pipelines.video_pipeline import ARTIFACT_REFRAME, ARTIFACT_REFRAME_FILE
    from app.services import reframe

    try:
        import app.services.storage as storage_module

        # Always the canonical file: the pipeline stage registers this path
        # with the store, and resolve_plan reads whatever the store has
        # registered. Writing to a path derived from the artifact *kind*
        # instead left a second file next to the real cache, so rebuilds were
        # written and then never read back.
        out = storage_module.storage.project_dir(clip.project_id) / ARTIFACT_REFRAME_FILE
        updated = dict(plans)
        updated[clip.id] = {
            "_cache_key": reframe.plan_cache_key(
                clip.start, clip.end, clip.layout_mode or "auto"
            ),
            "_cache_version": reframe.PLAN_VERSION,
            "_source_sig": f"{plan.source_width}x{plan.source_height}",
            **plan.to_dict(),
        }
        out.write_text(json.dumps(updated, ensure_ascii=False), encoding="utf-8")
        store.put(ARTIFACT_REFRAME, str(out), {"plans": len(updated)})
    except (OSError, TypeError, ValueError) as exc:  # noqa: BLE001
        log.debug("could not persist rebuilt plan for %s: %s", clip.id, exc)


def ARTIFACT_SOURCE_PATH(store: ArtifactStore):  # type: ignore[no-untyped-def]
    """Source path for the current project (None-safe)."""
    from app.pipelines.video_pipeline import ARTIFACT_SOURCE

    return store.path_of(ARTIFACT_SOURCE)


# --------------------------------------------------------------------------
# Subtitles
# --------------------------------------------------------------------------
def _keywords_for(clip: Clip) -> list[str]:
    """Highlight the salient words of the title + hashtags."""
    pool: list[str] = []
    for source in (clip.title, clip.hashtags):
        for word in str(source or "").split():
            core = "".join(c for c in word if c.isalnum()).lower()
            if len(core) >= 4:
                pool.append(core)
    return pool[:8]


def _emojis_for(clip: Clip) -> list[str]:
    if not settings.AUTO_EMOJI:
        return []
    return CATEGORY_EMOJIS.get(
        (clip.category_primary or "").lower(), _DEFAULT_EMOJIS
    )


def build_subtitles(clip: Clip, out_path: str | Path) -> Path | None:
    """Write the clip's .ass file, or None when there is nothing to subtitle."""
    words = clip.words or []
    if not words:
        log.warning("clip %s has no word timings; rendering without subtitles", clip.id)
        return None
    try:
        return subtitle.build_ass(
            words,
            clip.subtitle_style or "classic_bold_white",
            out_path,
            canvas_width=settings.RENDER_WIDTH,
            canvas_height=settings.RENDER_HEIGHT,
            censor=settings.CENSOR_PROFANITY,
            keywords=_keywords_for(clip),
            emojis=_emojis_for(clip),
        )
    except (ValueError, OSError) as exc:
        # A bad style or empty text must not kill the whole render.
        log.warning("subtitle build failed for clip %s: %s", clip.id, exc)
        return None


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
def render_clip(
    db: Session,
    clip: Clip,
    project: Project,
    store: ArtifactStore,
    *,
    force: bool = False,
) -> RenderResult:
    """Render one clip to 1080x1920 with burned-in subtitles.

    Sets `render_path`/`rendered`/`render_error` on the clip. Raises on
    unrecoverable ffmpeg failures so the batch renderer can record them.
    """
    if clip.rendered and clip.render_path and not force:
        path = storage.abspath(clip.render_path)
        if path.exists():
            return RenderResult(clip.id, path, clip.duration)

    src = ARTIFACT_SOURCE_PATH(store)
    if src is None:
        raise RuntimeError(f"source artifact missing for project {project.id}")
    info = media.probe(src)
    if not info.has_video:
        raise RuntimeError("source has no video stream; cannot render")

    plan = resolve_plan(db, clip, project, store, info)
    target_fps = clip.target_fps or settings.PREFERRED_FPS
    if target_fps > info.fps + 0.5:
        events.publish_log(
            project.id,
            f"clip #{clip.rank}: target {target_fps}fps exceeds source "
            f"{info.fps:.0f}fps; frames are duplicated, not interpolated",
            "warn",
        )

    ass = build_subtitles(clip, storage.project_dir(project.id) / f"subtitles_{clip.id}.ass")

    video_filters = build_video_filters(
        plan,
        ass_path=ass,
        target_fps=target_fps,
        source_fps=info.fps,
        duration=clip.duration,
        hook_text=str(clip.title or "")[:80],
        watermark=settings.WATERMARK_TEXT,
    )
    video_filters.append(_frame_rate_filter(target_fps, info.fps))

    audio_filters: list[str] = []
    if info.has_audio:
        audio_filters = [f"loudnorm=I={settings.TARGET_LUFS}:TP=-1.5:LRA=11"]

    out = storage.clips_dir(project.id) / f"clip_{clip.rank:02d}_{clip.id[:8]}.mp4"
    log.info(
        "rendering clip #%d (%.1fs, %s, %dfps) -> %s",
        clip.rank, clip.duration, plan.layout, target_fps, out.name,
    )
    encoder = resolve_encoder()
    with _clip_lock(clip.id):
        media.cut_segment(
            src,
            out,
            clip.start,
            clip.end,
            video_filters=video_filters,
            audio_filters=audio_filters,
            encoder=encoder,
            extra_args=encoder_args(encoder),
        )

        # A frame grab from the finished vertical file is already 9:16.
        thumb: Path | None = None
        try:
            thumb = media.extract_thumbnail(
                out, storage.clips_dir(project.id) / f"thumb_{clip.id[:8]}.jpg",
                at=min(0.4, max(clip.duration * 0.25, 0.05)),
            )
        except Exception as exc:  # noqa: BLE001 - a missing thumb is not fatal
            log.warning("thumbnail failed for clip %s: %s", clip.id, exc)
            thumb = None

        clip.render_path = storage.relpath(out) if out.exists() else None
        clip.thumb_path = storage.relpath(thumb) if thumb and thumb.exists() else None
        clip.rendered = bool(clip.render_path)
        clip.render_error = None
        db.commit()
    if ass is not None and ass.exists():
        ass.unlink()
    events.publish_log(project.id, f"Rendered clip #{clip.rank} ({clip.duration:.0f}s)")
    return RenderResult(clip.id, out, clip.duration)


def render_project_clips(
    db: Session,
    project: Project,
    store: ArtifactStore,
    *,
    clip_ids: list[str] | None = None,
    force: bool = False,
) -> tuple[int, int]:
    """Render every clip of a project. One failure does not stop the batch.

    Returns (rendered, failed). Clip-level errors are recorded on the clip so
    the UI can show them instead of a generic project failure.
    """
    q = db.query(Clip).filter(Clip.project_id == project.id)
    if clip_ids is not None:
        q = q.filter(Clip.id.in_(clip_ids))
    clips = q.order_by(Clip.rank.asc()).all()
    if not clips:
        events.publish_log(project.id, "no clips to render", "warn")
        return 0, 0

    encoder = resolve_encoder()
    events.publish_log(
        project.id, f"Rendering {len(clips)} clip(s) with {encoder}"
    )
    total = len(clips)
    if _render_concurrency(total, encoder) > 1:
        log.info("rendering %d clips in parallel", total)
        return _render_parallel(db, project, [c.id for c in clips], force=force)

    rendered = failed = 0
    for i, clip in enumerate(clips):
        try:
            render_clip(db, clip, project, store, force=force)
            rendered += 1
        except Exception as exc:  # noqa: BLE001 - keep rendering the rest
            failed += 1
            clip.rendered = False
            clip.render_error = str(exc)[:500]
            db.commit()
            log.error("clip %s failed to render: %s", clip.id, exc)
            events.publish_log(
                project.id, f"Clip #{clip.rank} failed: {exc}", "error"
            )
        events.publish_progress(
            project.id, 0.5 + 0.5 * (i + 1) / total, f"Rendered {i + 1}/{total}"
        )
    return rendered, failed


def _render_concurrency(n_clips: int, encoder: str | None = None) -> int:
    """How many clips to render at once.

    Bounded by the configured worker concurrency and half the CPU count (each
    ffmpeg encode already uses several threads). NVENC is capped separately:
    consumer GPUs allow only a few concurrent encoder sessions before they
    fail with "out of encoder memory", and VRAM is shared with whisper.
    """
    if n_clips <= 1:
        return 1
    cpu_cap = max(1, (os.cpu_count() or 4) // 2)
    workers = max(1, min(settings.WORKER_CONCURRENCY, cpu_cap, n_clips))
    if encoder and "nvenc" in encoder:
        workers = min(workers, settings.NVENC_MAX_PARALLEL)
    return workers


def _render_parallel(
    db: Session,
    project: Project,
    clip_ids: list[str],
    *,
    force: bool,
) -> tuple[int, int]:
    """Render clips on a thread pool; each job owns its DB session.

    SQLAlchemy sessions are not thread-safe, so every worker opens a fresh
    session against the same database and commits its own result. The caller's
    objects are expired at the end so it sees the committed state.
    """
    total = len(clip_ids)
    rendered = failed = 0

    def job(clip_id: str) -> tuple[str, bool]:
        session = SessionLocal()
        try:
            clip = session.get(Clip, clip_id)
            if clip is None:
                return clip_id, False
            proj = session.get(Project, project.id)
            store = ArtifactStore(session, project.id)
            try:
                render_clip(session, clip, proj, store, force=force)
                return clip_id, True
            except Exception as exc:  # noqa: BLE001 - keep rendering the rest
                clip.rendered = False
                clip.render_error = str(exc)[:500]
                session.commit()
                log.error("clip %s failed to render: %s", clip_id, exc)
                events.publish_log(project.id, f"Clip failed: {exc}", "error")
                return clip_id, False
        finally:
            session.close()

    try:
        with ThreadPoolExecutor(
            max_workers=_render_concurrency(total, resolve_encoder()),
            thread_name_prefix="cf-render",
        ) as pool:
            futures = [pool.submit(job, cid) for cid in clip_ids]
            done = 0
            for future in as_completed(futures):
                _, ok = future.result()
                done += 1
                if ok:
                    rendered += 1
                else:
                    failed += 1
                events.publish_progress(
                    project.id, 0.5 + 0.5 * done / total, f"Rendered {done}/{total}"
                )
    finally:
        # workers committed on their own connections; drop stale copies
        db.expire_all()
    return rendered, failed


# --------------------------------------------------------------------------
# Batch download
# --------------------------------------------------------------------------
def _safe_filename(text: str, maxlen: int = 60) -> str:
    keep = "-_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    # Collapse spaces to underscores so the name survives every shell/zip tool.
    collapsed = " ".join(str(text or "").split()).replace(" ", "_")
    cleaned = "".join(c if c in keep else "_" for c in collapsed)
    return cleaned[:maxlen].strip("._") or "clip"


def zip_rendered_clips(
    db: Session,
    project: Project,
    out_path: str | Path | None = None,
) -> Path:
    """Zip every rendered clip plus a text index with titles and hashtags.

    Unrendered clips are skipped (and listed as such in the index).
    """
    clips = (
        db.query(Clip)
        .filter(Clip.project_id == project.id)
        .order_by(Clip.rank.asc())
        .all()
    )
    if not clips:
        raise RuntimeError("project has no clips")

    out = Path(out_path) if out_path else storage.project_dir(project.id) / "clips.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    index_lines = [
        f"ClipForge export - {project.source_name}",
        f"Project: {project.id}  Clips: {len(clips)}",
        "=" * 60,
        "",
    ]
    included = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        used_names: set[str] = set()
        for clip in clips:
            if not clip.rendered or not clip.render_path:
                index_lines.append(f"[not rendered] #{clip.rank} {clip.title}")
                continue
            path = storage.abspath(clip.render_path)
            if not path.exists():
                index_lines.append(f"[missing file] #{clip.rank} {clip.title}")
                continue
            name = f"{clip.rank:02d}_{_safe_filename(clip.title or 'clip')}.mp4"
            # Guard against duplicate titles clobbering each other.
            base, ext = name.rsplit(".", 1)
            n = 2
            while name in used_names:
                name = f"{base}_{n}.{ext}"
                n += 1
            used_names.add(name)
            included += 1
            zf.write(path, name)
            index_lines += [
                f"#{clip.rank} {clip.title}",
                f"  file: {name}",
                f"  time: {clip.start:.1f}-{clip.end:.1f}s ({clip.duration:.1f}s) "
                f"score {clip.viral_score:.0f} [{clip.category_primary}]",
                f"  tags: {clip.hashtags}",
                "",
            ]
        zf.writestr("clips.txt", "\n".join(index_lines))
    if included == 0:
        out.unlink(missing_ok=True)
        raise RuntimeError("no rendered clips to export")
    log.info("wrote zip: %s (%d clip(s), %d bytes)", out.name, included, out.stat().st_size)
    return out
