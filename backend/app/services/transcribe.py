"""Transcription with faster-whisper (word-level timestamps).

The model is loaded lazily and cached at module level so repeated jobs reuse
the same instance. If faster-whisper is not installed, a clear error is
raised instead of a confusing import failure deeper in the stack.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from app.core.config import settings
from app.core.logging import get_logger
from app.services import media
from app.services.segments import Word

log = get_logger("transcribe")

_LOCK = threading.Lock()
_MODEL = None
_MODEL_KEY: tuple[str, str] | None = None


@dataclass
class TranscriptionResult:
    """Full transcription of a media file."""

    text: str
    language: str
    words: list[Word]
    segments: list[dict]  # [{start,end,text}]
    duration: float

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "language": self.language,
            "words": [asdict(w) for w in self.words],
            "segments": self.segments,
            "duration": self.duration,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TranscriptionResult:
        return cls(
            text=d.get("text", ""),
            language=d.get("language", "en"),
            words=[Word(**w) for w in d.get("words", [])],
            segments=d.get("segments", []),
            duration=d.get("duration", 0.0),
        )


class TranscriptionError(RuntimeError):
    pass


def _require_whisper():
    """Import faster-whisper lazily with a friendly error message."""
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError as exc:  # pragma: no cover - env dependent
        raise TranscriptionError(
            "faster-whisper is not installed. Install ML dependencies with: "
            "pip install -r requirements-ml.txt"
        ) from exc
    return WhisperModel


def _resolve_device() -> str:
    dev = settings.WHISPER_DEVICE
    if dev == "auto":
        try:
            import torch  # type: ignore

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return dev


def _resolve_compute(device: str) -> str:
    ct = settings.WHISPER_COMPUTE_TYPE
    if ct == "auto":
        return "float16" if device == "cuda" else "int8"
    return ct


def get_model(on_status: Callable[[str], None] | None = None):
    """Return a cached WhisperModel for the configured model size."""
    global _MODEL, _MODEL_KEY
    device = _resolve_device()
    compute = _resolve_compute(device)
    key = (settings.WHISPER_MODEL, f"{device}/{compute}")
    with _LOCK:
        if _MODEL is None or _MODEL_KEY != key:
            if on_status:
                on_status(
                    f"Loading whisper model '{settings.WHISPER_MODEL}' ({device}/{compute})"
                    " — first run downloads it from HuggingFace"
                )
            WhisperModel = _require_whisper()
            log.info(
                "loading whisper model=%s device=%s compute=%s",
                settings.WHISPER_MODEL,
                device,
                compute,
            )
            _MODEL = WhisperModel(
                settings.WHISPER_MODEL,
                device=device,
                compute_type=compute,
                download_root=settings.MODELS_DIR,
            )
            _MODEL_KEY = key
        return _MODEL


def _audio_key(audio_path: Path) -> str:
    h = hashlib.sha256()
    h.update(str(audio_path).encode())
    h.update(str(audio_path.stat().st_size).encode())
    return h.hexdigest()[:24]


def transcribe_audio(
    audio_path: str | Path,
    *,
    language: str = "auto",
    cache_path: str | Path | None = None,
    on_progress: Callable[[float], None] | None = None,
    on_status: Callable[[str], None] | None = None,
) -> TranscriptionResult:
    """Transcribe a WAV/any audio file, returning word-level timings.

    Results are cached to `cache_path` (JSON) when provided, keyed by the
    audio file hash, so re-running a job is free. `on_status` receives human
    readable milestones (model load) that don't map to fractional progress.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise TranscriptionError(f"audio file missing: {audio_path}")

    if cache_path is not None and settings.TRANSCRIBE_CACHE:
        result = _load_cache(Path(cache_path), _audio_key(audio_path))
        if result is not None:
            log.info("transcription cache hit: %s", audio_path.name)
            if on_progress:
                on_progress(1.0)
            return result

    if not media.ffprobe_available():
        raise TranscriptionError("ffmpeg/ffprobe not found on PATH")

    model = get_model(on_status=on_status)
    lang_arg = None if language in ("auto", "", None) else language
    log.info("transcribing %s (language=%s)", audio_path.name, lang_arg or "auto")

    try:
        segments_gen, info = model.transcribe(
            str(audio_path),
            language=lang_arg,
            beam_size=settings.WHISPER_BEAM_SIZE,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            condition_on_previous_text=True,
        )
    except Exception as exc:  # noqa: BLE001 - surface model errors clearly
        raise TranscriptionError(f"whisper transcription failed: {exc}") from exc

    words: list[Word] = []
    segs: list[dict] = []
    total_text: list[str] = []
    for seg in segments_gen:
        segs.append(
            {"start": round(seg.start, 3), "end": round(seg.end, 3), "text": seg.text.strip()}
        )
        total_text.append(seg.text)
        for w in (seg.words or []):
            words.append(
                Word(
                    word=w.word,
                    start=round(w.start, 3),
                    end=round(w.end, 3),
                    confidence=float(getattr(w, "probability", 1.0)),
                )
            )
        if on_progress and info.duration:
            on_progress(min(seg.end / max(info.duration, 1e-6), 0.99))

    text = " ".join(total_text).strip()
    text = " ".join(text.split())
    result = TranscriptionResult(
        text=text,
        language=str(getattr(info, "language", language or "en")),
        words=words,
        segments=segs,
        duration=float(getattr(info, "duration", 0.0) or 0.0),
    )
    log.info(
        "transcribed %s: %d words, %.1fs, lang=%s",
        audio_path.name,
        len(words),
        result.duration,
        result.language,
    )

    if cache_path is not None and settings.TRANSCRIBE_CACHE:
        _save_cache(Path(cache_path), _audio_key(audio_path), result)
    if on_progress:
        on_progress(1.0)
    return result


def _cache_file(cache_path: Path) -> Path:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    return cache_path


def _load_cache(cache_path: Path, key: str) -> TranscriptionResult | None:
    try:
        with _cache_file(cache_path).open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    if data.get("audio_key") != key:
        return None
    try:
        return TranscriptionResult.from_dict(data["result"])
    except (KeyError, TypeError):
        return None


def _save_cache(cache_path: Path, key: str, result: TranscriptionResult) -> None:
    try:
        with _cache_file(cache_path).open("w", encoding="utf-8") as fh:
            json.dump({"audio_key": key, "result": result.to_dict()}, fh, ensure_ascii=False)
    except OSError:  # pragma: no cover - cache is best-effort
        log.warning("failed to write transcription cache")
