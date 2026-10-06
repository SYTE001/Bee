"""Video processing pipeline.

Each stage is idempotent: it checks the `ArtifactStore` for a prior result of
its kind and skips itself if present. That makes jobs cheap to resume after a
crash and safe to re-run (e.g. with different options) without recomputing
expensive transcription.

Fase 1 implements stages 1-3 (ingest, audio extraction, transcription).
Later phases plug additional stages into `STAGES` below.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.db.models.project import Project, ProjectStatus
from app.db.session import SessionLocal
from app.services import events, ingest, media, project_state, task_runner, transcribe
from app.services.artifacts import ArtifactStore
from app.services.storage import storage

log = get_logger("pipeline")

ARTIFACT_SOURCE = "source_video"
ARTIFACT_AUDIO = "audio_wav"
ARTIFACT_AUDIO_NORM = "audio_normalized"
ARTIFACT_TRANSCRIPT = "transcript"
ARTIFACT_AUDIO_FEATURES = "audio_features"
ARTIFACT_DIARIZATION = "diarization"
ARTIFACT_LLM_ANALYSIS = "llm_analysis"
ARTIFACT_VISUAL = "visual_features"
ARTIFACT_CLIPS = "clips"
ARTIFACT_REFRAME = "reframe_plans"
# The single file the reframe plans are stored in. Both the pipeline stage and
# the renderer's write-back must agree on this name, or a rebuilt plan lands
# next to the cache the reader uses and is never seen again.
ARTIFACT_REFRAME_FILE = "reframe_plans.json"


# --------------------------------------------------------------------------
# Stage implementations
# --------------------------------------------------------------------------
def stage_ingest(db: Session, project: Project, store: ArtifactStore) -> None:
    """Stage 1: ingest source media (already uploaded or via yt-dlp)."""
    project_state.set_stage(db, project, ProjectStatus.INGESTING, "Preparing source video")

    existing = store.get(ARTIFACT_SOURCE)
    if existing and storage.exists(existing.path):
        log.info("ingest: artifact present, skipping")
        _apply_media_meta(project, store.get_meta(ARTIFACT_SOURCE))
        return

    if project.source_type == "youtube":
        result = ingest.ingest_youtube(project.id, project.source_url or "")
        name = result.title
    else:
        rel = _find_uploaded_source(project.id)
        if rel is None:
            raise RuntimeError("no uploaded source file found for project")
        result = ingest._probe_and_report(storage.abspath(rel), title=project.source_name)
        name = project.source_name

    meta = {
        "title": name,
        "duration": result.duration,
        "width": result.width,
        "height": result.height,
        "fps": result.fps,
        "has_video": result.has_video,
        "has_audio": result.has_audio,
    }
    store.put(ARTIFACT_SOURCE, str(result.path), meta)
    _apply_media_meta(project, meta)
    project_state.set_progress(db, project, 0.10, "Source video ready")


def _find_uploaded_source(project_id: str) -> str | None:
    d = storage.project_dir(project_id)
    candidates = sorted(
        (f for f in d.iterdir() if f.name.startswith("source_")),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    return storage.relpath(candidates[0]) if candidates else None


def _apply_media_meta(project: Project, meta: dict) -> None:
    project.duration = meta.get("duration")
    project.width = meta.get("width")
    project.height = meta.get("height")
    project.fps = meta.get("fps")
    project.has_video_stream = bool(meta.get("has_video", False))


def stage_extract_audio(db: Session, project: Project, store: ArtifactStore) -> None:
    """Stage 2: extract audio at 16 kHz mono for the ASR model."""
    project_state.set_stage(
        db, project, ProjectStatus.EXTRACTING_AUDIO, "Extracting audio track"
    )
    if store.has(ARTIFACT_AUDIO) and storage.exists(store.get(ARTIFACT_AUDIO).path):
        log.info("audio extraction: artifact present, skipping")
        return

    src = storage.abspath(store.get(ARTIFACT_SOURCE).path)
    out = storage.project_dir(project.id) / "audio_16k.wav"
    media.extract_audio(src, out, sample_rate=16000, mono=True)
    store.put(ARTIFACT_AUDIO, str(out), {"sample_rate": 16000})
    project_state.set_progress(db, project, 0.20, "Audio extracted")


def stage_transcribe(db: Session, project: Project, store: ArtifactStore) -> None:
    """Stage 3: transcribe with word-level timestamps (faster-whisper)."""
    project_state.set_stage(
        db, project, ProjectStatus.TRANSCRIBING, "Transcribing audio (this can take a while)"
    )
    if store.has(ARTIFACT_TRANSCRIPT):
        log.info("transcription: artifact present, skipping")
        meta = store.get_meta(ARTIFACT_TRANSCRIPT)
        project.detected_language = meta.get("language")
        project.num_speakers = meta.get("num_speakers", 0)
        return

    audio = storage.abspath(store.get(ARTIFACT_AUDIO).path)
    cache = storage.project_dir(project.id) / "transcript_cache.json"

    def on_progress(p: float) -> None:
        project_state.set_progress(
            db, project, 0.20 + 0.45 * p, f"Transcribing… {int(p * 100)}%"
        )

    def on_status(message: str) -> None:
        events.publish_log(project.id, message)

    result = transcribe.transcribe_audio(
        audio,
        language=project.language,
        cache_path=cache,
        on_progress=on_progress,
        on_status=on_status,
    )

    out = storage.project_dir(project.id) / "transcript.json"
    out.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    store.put(
        ARTIFACT_TRANSCRIPT,
        str(out),
        {
            "language": result.language,
            "word_count": len(result.words),
            "duration": result.duration,
            "num_speakers": 0,
        },
    )
    project.detected_language = result.language
    project_state.set_progress(db, project, 0.65, f"Transcribed {len(result.words)} words")


def stage_diarize(db: Session, project: Project, store: ArtifactStore) -> None:
    """Stage 4: speaker diarization (optional; skipped without HF_TOKEN)."""
    from app.services import diarize as diarize_service

    project_state.set_stage(
        db, project, ProjectStatus.DIARIZING, "Detecting speakers"
    )
    if store.has(ARTIFACT_DIARIZATION):
        meta = store.get_meta(ARTIFACT_DIARIZATION)
        project.num_speakers = meta.get("num_speakers", 0)
        return

    audio = store.path_of(ARTIFACT_AUDIO)
    if audio is None:
        raise RuntimeError("audio artifact missing")
    result = diarize_service.diarize_audio(audio)
    out = storage.project_dir(project.id) / "diarization.json"
    out.write_text(json.dumps(result.to_dict()), encoding="utf-8")
    store.put(ARTIFACT_DIARIZATION, str(out), {"num_speakers": result.num_speakers})
    project.num_speakers = result.num_speakers
    project_state.set_progress(db, project, 0.70, f"{result.num_speakers} speaker(s) detected")


def stage_analyze_audio(db: Session, project: Project, store: ArtifactStore) -> None:
    """Stage 5: extract audio features (energy, pitch, laughter)."""
    from app.services import audio_analysis

    project_state.set_stage(
        db, project, ProjectStatus.ANALYZING, "Analyzing audio signals"
    )
    if store.has(ARTIFACT_AUDIO_FEATURES):
        return

    audio = store.path_of(ARTIFACT_AUDIO)
    if audio is None:
        raise RuntimeError("audio artifact missing")
    try:
        feats = audio_analysis.analyze_audio(audio)
    except audio_analysis.AudioAnalysisError as exc:
        events.publish_log(project.id, f"audio analysis skipped: {exc}", "warn")
        feats = audio_analysis.AudioFeatures(duration=0.0)

    out = storage.project_dir(project.id) / "audio_features.json"
    audio_analysis.save_features(feats, out)
    store.put(ARTIFACT_AUDIO_FEATURES, str(out), {"frames": feats.frames})
    project_state.set_progress(db, project, 0.78, "Audio features extracted")


def stage_analyze_llm(db: Session, project: Project, store: ArtifactStore) -> None:
    """Stage 6: LLM semantic analysis of the transcript, chunk by chunk."""
    from app.services import llm_analyzer
    from app.services.scoring import build_sentence_groups

    project_state.set_stage(
        db, project, ProjectStatus.ANALYZING, "Analyzing transcript with AI"
    )
    if store.has(ARTIFACT_LLM_ANALYSIS):
        return

    result = _load_transcript(store)
    if not result:
        raise RuntimeError("transcript artifact missing")

    sentences = build_sentence_groups(result.words, max_seconds=30.0)
    if not sentences:
        raise RuntimeError("no sentences found in transcript")

    chunks = llm_analyzer.chunk_sentences(sentences)
    events.publish_log(
        project.id, f"Analyzing {len(sentences)} sentences in {len(chunks)} chunk(s)"
    )

    analyses: list[dict] = []
    for ci, chunk in enumerate(chunks):
        cache_file = storage.project_dir(project.id) / f"llm_cache_{ci}.json"
        key = llm_analyzer.cache_key(chunk)
        cached = llm_analyzer.load_cached(cache_file, key)
        if cached is not None:
            analyses.append(cached.to_dict())
            events.publish_log(project.id, f"chunk {ci + 1}/{len(chunks)} (cached)")
        else:
            analysis = llm_analyzer.analyze_chunk(chunk, ci)
            llm_analyzer.save_cached(cache_file, key, analysis)
            analyses.append(analysis.to_dict())
            events.publish_log(
                project.id, f"chunk {ci + 1}/{len(chunks)} analyzed ({len(analysis.sentences)} sentences)"
            )
        project_state.set_progress(
            db, project, 0.78 + 0.15 * (ci + 1) / max(len(chunks), 1), "Analyzing transcript…"
        )

    out = storage.project_dir(project.id) / "llm_analysis.json"
    out.write_text(json.dumps(analyses, ensure_ascii=False), encoding="utf-8")
    store.put(
        ARTIFACT_LLM_ANALYSIS,
        str(out),
        {"chunks": len(analyses), "sentences": len(sentences)},
    )


def stage_analyze_visual(db: Session, project: Project, store: ArtifactStore) -> None:
    """Stage 7: visual signals — scene cuts, face presence, on-camera motion.

    These feed the `visual` dimension of the viral score. Runs on the video
    stream at a reduced framerate; degrades to a neutral score when no video
    stream or vision stack is available.
    """
    from app.services import visual_analysis

    project_state.set_stage(
        db, project, ProjectStatus.ANALYZING, "Analyzing visual signals"
    )
    if store.has(ARTIFACT_VISUAL):
        return

    src = store.path_of(ARTIFACT_SOURCE)
    if src is None:
        raise RuntimeError("source artifact missing")

    info = media.probe(src)
    if not info.has_video:
        events.publish_log(project.id, "no video stream — visual scoring disabled", "warn")
        empty = visual_analysis.VisualFeatures(duration=info.duration)
        out = storage.project_dir(project.id) / "visual_features.json"
        visual_analysis.save_features(empty, out)
        store.put(ARTIFACT_VISUAL, str(out), {"samples": 0, "cuts": 0})
        return

    try:
        feats = visual_analysis.analyze_video(src)
    except Exception as exc:  # noqa: BLE001 - analysis must never kill the run
        events.publish_log(project.id, f"visual analysis failed: {exc}", "warn")
        feats = visual_analysis.VisualFeatures(duration=info.duration)

    out = storage.project_dir(project.id) / "visual_features.json"
    visual_analysis.save_features(feats, out)
    store.put(
        ARTIFACT_VISUAL,
        str(out),
        {"samples": len(feats.times), "cuts": len(feats.cuts)},
    )
    project_state.set_progress(
        db, project, 0.80, f"{len(feats.cuts)} scene cut(s) detected"
    )


def stage_score_and_select(db: Session, project: Project, store: ArtifactStore) -> None:
    """Stage 7: generate candidates, score them, and persist the Top-N clips."""
    from app.db.models.clip import Clip
    from app.services import audio_analysis, diarize as diarize_service
    from app.services.scoring import (
        build_description,
        build_hashtags,
        build_sentence_groups,
        build_title,
        generate_candidates,
        score_candidates,
        select_top_n,
    )

    project_state.set_stage(db, project, ProjectStatus.SCORING, "Scoring and ranking clips")

    result = _load_transcript(store)
    if not result:
        raise RuntimeError("transcript artifact missing")

    sentences = build_sentence_groups(result.words, max_seconds=30.0)
    chunks_raw = json.loads(
        store.path_of(ARTIFACT_LLM_ANALYSIS).read_text(encoding="utf-8")
    )

    from app.services.llm_analyzer import ChunkAnalysis, SentenceScore

    llm_index: dict[int, SentenceScore] = {}
    chunk_overalls: list[float] = []
    for raw in chunks_raw:
        ca = ChunkAnalysis.from_dict(raw)
        chunk_overalls.append(ca.overall_interest)
        for s in ca.sentences:
            llm_index[s.sentence_index] = s

    feats = None
    if store.has(ARTIFACT_AUDIO_FEATURES):
        try:
            features_path = store.path_of(ARTIFACT_AUDIO_FEATURES)
            if features_path is not None:
                feats = audio_analysis.load_features(features_path)
        except Exception:  # noqa: BLE001
            feats = None

    turns: list[dict] = []
    if store.has(ARTIFACT_DIARIZATION):
        try:
            dia = diarize_service.Diarization.from_dict(
                json.loads(
                    store.path_of(ARTIFACT_DIARIZATION).read_text(
                        encoding="utf-8"
                    )
                )
            )
            turns = dia.turns
        except Exception:  # noqa: BLE001
            turns = []

    visual_feats = None
    if store.has(ARTIFACT_VISUAL):
        try:
            visual_path = store.path_of(ARTIFACT_VISUAL)
            if visual_path is not None:
                from app.services import visual_analysis

                visual_feats = visual_analysis.load_features(visual_path)
        except Exception:  # noqa: BLE001
            visual_feats = None

    events.publish_log(project.id, "Generating clip candidates…")
    raw_candidates = generate_candidates(
        sentences,
        result.words,
        dur_min=float(project.target_dur_min),
        dur_max=float(project.target_dur_max),
    )
    events.publish_log(project.id, f"Scoring {len(raw_candidates)} candidate windows…")

    scored = score_candidates(
        raw_candidates,
        sentences,
        result.words,
        llm_index,
        chunk_overalls,
        feats,
        turns,
        visual_feats=visual_feats,
    )
    categories = [c for c in (project.desired_categories or "").split(",") if c]
    selected = select_top_n(
        scored,
        top_n=project.target_clip_count,
        categories=categories,
    )
    events.publish_log(project.id, f"Selected {len(selected)} clips")

    # Replace any previous clip rows (re-run support)
    db.query(Clip).filter(Clip.project_id == project.id).delete()
    db.flush()

    for rank, cand in enumerate(selected, start=1):
        clip = Clip(
            project_id=project.id,
            rank=rank,
            start=round(cand.start, 3),
            end=round(cand.end, 3),
            duration=round(cand.duration, 3),
            viral_score=round(cand.viral_score * 100.0, 1),
            category_primary=cand.category_primary,
            category_secondary=cand.category_secondary,
            scores={k: round(v, 4) for k, v in cand.scores.items()},
            reasons=cand.reasons,
            title=build_title(cand, language=result.language),
            description=build_description(cand),
            hashtags=build_hashtags(cand, language=result.language),
            transcript=cand.text,
            words=[w.__dict__ for w in cand.words],
            subtitle_style=project.subtitle_style,
            layout_mode=project.layout_mode,
            target_fps=project.target_fps,
        )
        db.add(clip)
    db.commit()

    store.put(
        ARTIFACT_CLIPS,
        None,
        {
            "candidates": len(raw_candidates),
            "selected": len(selected),
            "top_score": max((c.viral_score for c in selected), default=0.0),
        },
    )
    project_state.set_progress(db, project, 0.95, f"{len(selected)} clips ranked")


def stage_reframe(db: Session, project: Project, store: ArtifactStore) -> None:
    """Stage 8: build a 9:16 crop plan per clip from face tracking.

    Analyses each clip's frame range at low framerate and writes a crop plan
    (layout, keyframed crop window, zoom) that the renderer consumes. Plans
    are cached per (time range, layout) so a re-score that mints new clip ids
    still reuses the expensive face pass for an unchanged range.
    """
    from app.db.models.clip import Clip
    from app.services import reframe

    project_state.set_stage(
        db, project, ProjectStatus.REFRAMING, "Framing clips for 9:16"
    )

    clips = db.query(Clip).filter(Clip.project_id == project.id).order_by(Clip.rank).all()
    if not clips:
        return

    src = store.path_of(ARTIFACT_SOURCE)
    if src is None:
        raise RuntimeError("source artifact missing")
    info = media.probe(src)
    if not info.has_video or info.width == 0 or info.height == 0:
        events.publish_log(project.id, "no video stream — clips keep fit_blur layout", "warn")
        return

    # A single artifact holds one plan per clip id, so a re-run replaces them
    # atomically instead of leaving a mix of old and new plans. Plans are also
    # indexed by (range, layout) so a re-score — which mints new clip ids —
    # can reuse the expensive face analysis for an unchanged time range.
    plans: dict[str, dict] = {}
    plans_by_range: dict[str, str] = {}
    if store.has(ARTIFACT_REFRAME):
        try:
            plans = json.loads(
                store.path_of(ARTIFACT_REFRAME).read_text(encoding="utf-8")  # type: ignore[arg-type]
            )
            plans_by_range = {
                v["_cache_key"]: k
                for k, v in plans.items()
                if isinstance(v, dict) and v.get("_cache_key")
            }
        except (OSError, json.JSONDecodeError):
            plans = {}

    def _cached_plan(cache_key: str, source_sig: str) -> dict | None:
        """Return a usable cached plan for this range, or None."""
        cid = plans_by_range.get(cache_key)
        if cid is None:
            return None
        cached = plans.get(cid)
        if not cached or cached.get("_source_sig") != source_sig:
            return None
        return cached

    total = len(clips)
    for i, clip in enumerate(clips):
        layout = clip.layout_mode or "auto"
        # Cache on the time range + layout, not the clip id: re-scoring gives
        # clips new ids while their frame range is unchanged, so keying on
        # the id would re-run the expensive face pass for nothing. The version
        # in the key discards plans from an older build whose schema changed.
        cache_key = reframe.plan_cache_key(clip.start, clip.end, layout)
        cached = _cached_plan(cache_key, f"{info.width}x{info.height}")
        if cached is not None:
            # Reuse the plan under the new clip id.
            plans[clip.id] = cached
            continue

        events.publish_log(
            project.id,
            f"Framing clip {i + 1}/{total} ({reframe.DETECT_FPS:.0f} fps sampling)",
        )
        frames, detector = reframe.analyze_faces(
            src, start=clip.start, end=clip.end
        )
        plan = reframe.build_crop_plan(
            frames,
            layout=layout,
            width=info.width,
            height=info.height,
            detector=detector,
        )
        plans[clip.id] = {
            "_cache_key": cache_key,
            "_cache_version": reframe.PLAN_VERSION,
            "_source_sig": f"{info.width}x{info.height}",
            **plan.to_dict(),
        }
        project_state.set_progress(
            db, project, 0.95 + 0.04 * (i + 1) / total, f"Framed clip {i + 1}/{total}"
        )

    out = storage.project_dir(project.id) / ARTIFACT_REFRAME_FILE
    out.write_text(json.dumps(plans, ensure_ascii=False), encoding="utf-8")
    store.put(ARTIFACT_REFRAME, str(out), {"plans": len(plans)})
    events.publish_log(project.id, f"Reframing complete: {len(plans)} plan(s)")


def stage_render(db: Session, project: Project, store: ArtifactStore) -> None:
    """Stage 9: render every selected clip to a downloadable 9:16 video.

    Rendered clips are skipped, so re-running the pipeline (e.g. after the
    editor changed one clip) only re-encodes what changed. A clip that fails
    is recorded on that clip, not the project.
    """
    from app.db.models.clip import Clip
    from app.services import render

    pending = (
        db.query(Clip)
        .filter(Clip.project_id == project.id, Clip.rendered == False)  # noqa: E712
        .count()
    )
    if pending == 0:
        return

    project_state.set_stage(
        db, project, ProjectStatus.RENDERING, f"Rendering {pending} clip(s)"
    )
    rendered, failed = render.render_project_clips(db, project, store)
    if failed:
        events.publish_log(
            project.id, f"{failed} clip(s) failed to render", "warn"
        )
    events.publish_log(
        project.id, f"Rendered {rendered} clip(s), {failed} failed"
    )


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def _stage_list() -> list[Callable[[Session, Project, ArtifactStore], None]]:
    """Ordered stages. Later phases append analysis/render stages here."""
    return [
        stage_ingest,
        stage_extract_audio,
        stage_transcribe,
        stage_diarize,
        stage_analyze_audio,
        stage_analyze_llm,
        stage_analyze_visual,
        stage_score_and_select,
        stage_reframe,
        stage_render,
    ]


def _load_transcript(store: ArtifactStore):
    """Load the cached TranscriptionResult from the transcript artifact."""
    from app.services.transcribe import TranscriptionResult

    art = store.get(ARTIFACT_TRANSCRIPT)
    if not art or not art.path:
        return None
    try:
        data = json.loads(storage.abspath(art.path).read_text(encoding="utf-8"))
        return TranscriptionResult.from_dict(data)
    except (OSError, json.JSONDecodeError) as exc:
        log.error("failed to load transcript: %s", exc)
        return None


STAGE_NAMES = [
    ("ingesting", "Preparing source video"),
    ("extracting_audio", "Extracting audio track"),
    ("transcribing", "Transcribing audio"),
    ("diarizing", "Detecting speakers"),
    ("analyzing", "Detecting viral moments"),
    ("scoring", "Ranking clips"),
    ("reframing", "Framing clips for 9:16"),
    ("subtitling", "Generating subtitles"),
    ("rendering", "Rendering clips"),
]


def run_pipeline(project_id: str) -> None:
    """Entry point: run all stages for a project.

    Runs in its own DB session (worker context) and never raises — failures are
    recorded on the project and broadcast over the event bus.
    """
    db = SessionLocal()
    try:
        project = db.get(Project, project_id)
        if project is None:
            log.error("pipeline: project %s not found", project_id)
            return
        store = ArtifactStore(db, project_id)
        log.info("=== pipeline start: %s (%s) ===", project.source_name, project_id)
        events.publish_log(project_id, f"Pipeline started for '{project.source_name}'")

        for stage in _stage_list():
            stage(db, project, store)

        project_state.set_progress(db, project, 1.0, "Pipeline complete")
        from app.db.models.clip import Clip

        clip_count = db.query(Clip).filter(Clip.project_id == project_id).count()
        project_state.complete(db, project, clip_count)
        log.info("=== pipeline done: %s (%d clips) ===", project_id, clip_count)
    except Exception as exc:  # noqa: BLE001 - top-level guard
        log.exception("pipeline failed for %s", project_id)
        proj = db.get(Project, project_id)
        if proj is not None:
            project_state.fail(db, proj, str(exc)[:2000])
        else:
            events.publish_error(project_id, str(exc)[:2000])
    finally:
        db.close()


def enqueue_pipeline(project_id: str) -> str:
    """Submit the pipeline as a background task (celery or thread)."""
    return task_runner.submit_task(run_pipeline, project_id)


def resume_pipeline(project_id: str) -> str:
    """Re-run the pipeline; completed stages are skipped via artifact store."""
    return enqueue_pipeline(project_id)


def run_render(project_id: str, clip_id: str | None = None, force: bool = False) -> None:
    """Render task: one clip (clip_id set) or the whole project.

    Runs in its own session like `run_pipeline`. A single-clip failure is
    recorded on the clip, not the project, and the project is returned to
    `completed` afterwards so the UI does not sit in "rendering" forever.
    """
    from app.db.models.clip import Clip
    from app.services import render

    db = SessionLocal()
    try:
        project = db.get(Project, project_id)
        if project is None:
            log.error("render: project %s not found", project_id)
            return
        store = ArtifactStore(db, project_id)
        project_state.set_stage(
            db, project, ProjectStatus.RENDERING,
            f"Rendering {'clip' if clip_id else 'clips'}",
        )

        if clip_id is not None:
            clip = db.get(Clip, clip_id)
            if clip is None or clip.project_id != project_id:
                events.publish_error(project_id, "clip not found")
                return
            try:
                render.render_clip(db, clip, project, store, force=force)
            except Exception as exc:  # noqa: BLE001
                clip.rendered = False
                clip.render_error = str(exc)[:500]
                db.commit()
                events.publish_error(project_id, f"Render failed: {exc}")
        else:
            rendered, failed = render.render_project_clips(db, project, store, force=force)
            events.publish_log(
                project_id, f"Batch render: {rendered} done, {failed} failed"
            )

        count = db.query(Clip).filter(Clip.project_id == project_id).count()
        project_state.complete(db, project, count)
    except Exception as exc:  # noqa: BLE001 - top-level guard
        log.exception("render task failed for %s", project_id)
        proj = db.get(Project, project_id)
        if proj is not None:
            project_state.fail(db, proj, str(exc)[:2000])
        else:
            events.publish_error(project_id, str(exc)[:2000])
    finally:
        db.close()


def enqueue_render(
    project_id: str, clip_id: str | None = None, *, force: bool = False
) -> str:
    """Submit a render as a background task."""
    return task_runner.submit_task(run_render, project_id, clip_id, force)
