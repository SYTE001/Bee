"""Speaker diarization with pyannote.audio.

Optional stage: identifies who speaks when, which powers the split-screen
layout and banter/overlap detection. Skipped cleanly when pyannote or the
HuggingFace token is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from app.core.config import settings
from app.core.logging import get_logger

log = get_logger("diarize")


@dataclass
class Diarization:
    """Speaker turns across the file."""

    turns: list[dict] = field(default_factory=list)  # [{start, end, speaker}]
    num_speakers: int = 0

    def to_dict(self) -> dict:
        return {"turns": self.turns, "num_speakers": self.num_speakers}

    @classmethod
    def from_dict(cls, d: dict) -> Diarization:
        return cls(turns=d.get("turns", []), num_speakers=d.get("num_speakers", 0))


def diarize_audio(audio_path: str | Path) -> Diarization:
    """Run pyannote speaker diarization; returns empty result if unavailable."""
    if not settings.DIARIZATION_ENABLED:
        log.info("diarization disabled in config")
        return Diarization()

    if not settings.HF_TOKEN:
        log.warning(
            "HF_TOKEN not set — speaker diarization skipped "
            "(split-screen layout will fall back to single focus)"
        )
        return Diarization()

    try:
        from pyannote.audio import Pipeline
        import torch
    except ImportError:  # pragma: no cover - env dependent
        log.warning("pyannote.audio not installed — diarization skipped")
        return Diarization()

    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=settings.HF_TOKEN
        )
        if pipeline is None:
            raise RuntimeError("pipeline download failed")
    except Exception as exc:  # noqa: BLE001
        log.warning("could not load diarization pipeline: %s", exc)
        return Diarization()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("diarizing %s on %s", Path(audio_path).name, device)
    try:
        annotation = pipeline(str(audio_path), device=device)
    except Exception as exc:  # noqa: BLE001
        log.warning("diarization failed: %s", exc)
        return Diarization()

    turns: list[dict] = []
    speakers: set[str] = set()
    for turn, _, label in annotation.itertracks(yield_label=True):
        turns.append(
            {"start": round(float(turn.start), 3), "end": round(float(turn.end), 3), "speaker": label}
        )
        speakers.add(label)
    log.info("diarization: %d turns, %d speakers", len(turns), len(speakers))
    return Diarization(turns=turns, num_speakers=len(speakers))


def speaker_at(turns: list[dict], t: float) -> str | None:
    """Return the speaker active at time `t`, or None."""
    for turn in turns:
        if turn["start"] <= t <= turn["end"]:
            return turn["speaker"]
    return None


def turn_change_count(turns: list[dict], start: float, end: float) -> int:
    """Number of speaker switches inside [start, end] — the banter signal."""
    in_window = [t for t in turns if t["start"] < end and t["end"] > start]
    if len(in_window) < 2:
        return 0
    changes = 0
    prev = in_window[0]["speaker"]
    for t in in_window[1:]:
        if t["speaker"] != prev:
            changes += 1
            prev = t["speaker"]
    return changes
