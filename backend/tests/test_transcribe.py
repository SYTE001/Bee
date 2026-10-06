"""Tests for the transcription service.

The heavy test uses the real faster-whisper `tiny` model (skipped if the
package is not installed) to validate the end-to-end audio→words path.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys

import pytest

from app.services import transcribe as tr
from app.services.segments import Word


def _importable(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _synth_speech(path, seconds: float = 2.5) -> str:
    """Generate a WAV with a tone (no speech; tiny model will transcribe loosely)."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", f"sine=frequency=220:duration={seconds}",
            "-c:a", "pcm_s16le", "-ar", "16000", "-ac", "1",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return str(path)


class TestTranscriptionResult:
    def test_roundtrip(self):
        res = tr.TranscriptionResult(
            text="Hello world.",
            language="en",
            words=[Word(word="Hello", start=0.0, end=0.5)],
            segments=[{"start": 0.0, "end": 0.5, "text": "Hello"}],
            duration=0.5,
        )
        d = res.to_dict()
        back = tr.TranscriptionResult.from_dict(d)
        assert back.text == "Hello world."
        assert back.words[0].word == "Hello"
        assert back.language == "en"

    def test_from_dict_missing_fields(self):
        back = tr.TranscriptionResult.from_dict({})
        assert back.text == ""
        assert back.words == []


class TestDeviceResolution:
    def test_auto_falls_back_to_cpu(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("no torch")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        monkeypatch.setattr(tr.settings, "WHISPER_DEVICE", "auto")
        assert tr._resolve_device() == "cpu"

    def test_compute_type_maps_to_device(self, monkeypatch):
        monkeypatch.setattr(tr.settings, "WHISPER_COMPUTE_TYPE", "auto")
        assert tr._resolve_compute("cuda") == "float16"
        assert tr._resolve_compute("cpu") == "int8"


@pytest.mark.skipif(
    not _importable("faster_whisper"),
    reason="faster-whisper not installed",
)
class TestRealTranscription:
    def test_transcribes_real_audio(self, tmp_path, monkeypatch):
        # Use the tiny model to keep the test fast (~75MB one-time download)
        monkeypatch.setattr(tr.settings, "WHISPER_MODEL", "tiny")
        monkeypatch.setattr(tr.settings, "TRANSCRIBE_CACHE", False)
        tr._MODEL = None  # drop any cached larger model

        audio = _synth_speech(tmp_path / "tone.wav")
        result = tr.transcribe_audio(audio, language="en")
        assert result.duration == pytest.approx(2.5, abs=0.3)
        assert isinstance(result.text, str)

    def test_cache_is_written_and_reused(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tr.settings, "WHISPER_MODEL", "tiny")
        tr._MODEL = None
        audio = _synth_speech(tmp_path / "tone2.wav")
        cache = tmp_path / "cache.json"

        first = tr.transcribe_audio(audio, language="en", cache_path=cache)
        assert cache.exists()
        second = tr.transcribe_audio(audio, language="en", cache_path=cache)
        assert first.text == second.text


def _importable(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None
