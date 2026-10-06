"""Audio signal analysis: energy, pitch, speech rate, laughter/applause.

All heavy imports are lazy so the rest of the pipeline works without librosa or
torch installed — analysis just degrades to fewer signals with a logged
warning.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from app.core.config import settings
from app.core.logging import get_logger

log = get_logger("audio_analysis")

# AudioSet labels we treat as audience reactions
REACTION_LABELS = {
    "Laughter",
    "Applause",
    "Cheering",
    "Crowd",
    "Chuckling, sniggering",
    "Giggle",
    "Belly laugh",
}
# Frames-per-second for the feature timeline
FEATURE_FPS = 10


@dataclass
class AudioFeatures:
    """Per-10th-of-a-second features across the whole audio track."""

    duration: float
    rms: list[float] = field(default_factory=list)
    pitch: list[float] = field(default_factory=list)
    voiced: list[bool] = field(default_factory=list)
    # probability of laughter/applause per frame
    reaction_prob: list[float] = field(default_factory=list)
    # silence mask (True = silence)
    silence: list[bool] = field(default_factory=list)
    sample_rate: int = 16000

    @property
    def frames(self) -> int:
        return len(self.rms)

    def to_dict(self) -> dict:
        return {
            "duration": self.duration,
            "fps": FEATURE_FPS,
            "sample_rate": self.sample_rate,
            "rms": self.rms,
            "pitch": self.pitch,
            "voiced": self.voiced,
            "reaction_prob": self.reaction_prob,
            "silence": self.silence,
        }

    @classmethod
    def from_dict(cls, d: dict) -> AudioFeatures:
        return cls(
            duration=d["duration"],
            rms=d["rms"],
            pitch=d.get("pitch", []),
            voiced=d.get("voiced", []),
            reaction_prob=d.get("reaction_prob", []),
            silence=d.get("silence", []),
            sample_rate=d.get("sample_rate", 16000),
        )


@dataclass
class SegmentAudioStats:
    """Aggregated audio stats for a candidate clip window."""

    energy_score: float = 0.0  # 0..1 relative to whole-file distribution
    energy_peak: float = 0.0
    laughter_score: float = 0.0  # mean reaction probability in window
    laughter_peak: float = 0.0
    speech_rate: float = 0.0  # syllables-ish per second (0..~6)
    pitch_variability: float = 0.0
    silence_ratio: float = 0.0
    # dramatic pause: a long silence immediately before the window's energy peak
    pre_pause: float = 0.0


class AudioAnalysisError(RuntimeError):
    pass


def analyze_audio(audio_path: str | Path) -> AudioFeatures:
    """Extract the full audio feature timeline.

    Requires librosa. Falls back to an empty feature set if unavailable, in
    which case scoring uses the remaining (text/LLM) signals.
    """
    try:
        import librosa
        import numpy as np
    except ImportError:  # pragma: no cover - env dependent
        log.warning(
            "librosa/numpy not installed — skipping audio analysis. "
            "Install with: pip install -r requirements-ml.txt"
        )
        return AudioFeatures(duration=0.0)

    path = str(audio_path)
    log.info("analyzing audio: %s", Path(path).name)
    try:
        y, sr = librosa.load(path, sr=16000, mono=True)
    except Exception as exc:  # noqa: BLE001
        raise AudioAnalysisError(f"failed to load audio: {exc}") from exc

    duration = float(len(y) / sr)
    n_frames = max(int(duration * FEATURE_FPS), 1)
    hop = max(len(y) // n_frames, 1)

    feats = AudioFeatures(duration=duration, sample_rate=sr)

    # --- RMS energy per frame
    rms_all = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    feats.rms = [float(v) for v in _resample(rms_all, n_frames)]

    # --- Pitch (fundamental frequency) via pyin on voiced frames
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=65, fmax=500, sr=sr, hop_length=hop
        )
        f0 = np.nan_to_num(f0, nan=0.0)
        feats.pitch = [float(v) for v in _resample(f0, n_frames)]
        feats.voiced = [bool(v) for v in _resample(voiced_flag.astype(float), n_frames)]
    except Exception:  # pragma: no cover - pyin can fail on odd audio
        log.warning("pitch extraction (pyin) failed; continuing without pitch")

    # --- Silence mask (threshold relative to max energy)
    try:
        rms_np = np.asarray(feats.rms, dtype=float)
        thresh = max(float(rms_np.max()) * 0.06, 1e-5)
        feats.silence = [bool(v < thresh) for v in feats.rms]
    except ValueError:
        feats.silence = [False] * n_frames

    # --- Laughter / applause via AST (Audio Spectrogram Transformer)
    feats.reaction_prob = _detect_reactions(y, sr, n_frames, hop)

    log.info(
        "audio analysis: %.1fs, %d frames, reaction peak=%.2f",
        duration,
        n_frames,
        max(feats.reaction_prob, default=0.0),
    )
    return feats


def _resample(arr, n: int) -> list[float]:
    """Linear-interpolate a numpy array to exactly `n` samples."""
    import numpy as np

    if len(arr) == n:
        return list(arr)
    if len(arr) == 0:
        return [0.0] * n
    idx = np.linspace(0, len(arr) - 1, n)
    return list(np.interp(idx, np.arange(len(arr)), np.asarray(arr, dtype=float)))


def _detect_reactions(y, sr: int, n_frames: int, hop: int) -> list[float]:
    """Probability of laughter/applause per frame via an AudioSet model.

    Uses HuggingFace `MIT/ast-finetuned-audioset-10-10-0.4593` when available.
    Returns zeros if torch/transformers or the model are unavailable.
    """
    try:
        import numpy as np
        import torch
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
    except ImportError:  # pragma: no cover - env dependent
        log.warning("torch/transformers not installed — skipping laughter detection")
        return [0.0] * n_frames

    model_id = "MIT/ast-finetuned-audioset-10-10-0.4593"
    try:
        extractor = AutoFeatureExtractor.from_pretrained(model_id)
        model = AutoModelForAudioClassification.from_pretrained(model_id)
        model.eval()
    except Exception as exc:  # noqa: BLE001 - model download failure is not fatal
        log.warning("laughter model unavailable (%s); skipping", type(exc).__name__)
        return [0.0] * n_frames

    # Score in 3-second windows with 1.5s stride, then interpolate to frames.
    window = int(3.0 * sr)
    stride = int(1.5 * sr)
    probs: list[float] = []
    times: list[float] = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    for start in range(0, max(len(y) - window, 0) + 1, stride):
        chunk = y[start : start + window]
        if len(chunk) < window // 2:
            break
        try:
            inputs = extractor(
                chunk, sampling_rate=sr, return_tensors="pt", padding=True
            ).to(device)
            with torch.no_grad():
                logits = model(**inputs).logits[0]
            probs.append(float(_reaction_prob_from_logits(model.config, logits)))
            times.append(start / sr)
        except Exception:  # noqa: BLE001 - a bad chunk must not kill the run
            continue

    if not probs:
        return [0.0] * n_frames

    # Map to frame indices
    import numpy as np

    frame_times = np.arange(n_frames) / FEATURE_FPS
    out = np.interp(
        frame_times, np.asarray(times), np.asarray(probs), left=0.0, right=0.0
    )
    return [min(max(float(v), 0.0), 1.0) for v in out]


def _reaction_prob_from_logits(config, logits) -> float:
    """Sum softmax mass over AudioSet labels that indicate audience reaction."""
    import torch

    probs = torch.softmax(logits, dim=-1).tolist()
    id2label = getattr(config, "id2label", {})
    total = 0.0
    for idx, p in enumerate(probs):
        label = id2label.get(idx, "")
        # AST labels are like 'Laughter', 'Applause', 'Cheering', ...
        if any(tag.lower() in label.lower() for tag in REACTION_LABELS):
            total += p
    return min(total, 1.0)


def segment_stats(
    feats: AudioFeatures,
    start: float,
    end: float,
    words: list,  # list of Word-like objects with .start/.end
) -> SegmentAudioStats:
    """Aggregate features over [start, end] into 0..1 scores."""
    import numpy as np

    if not feats.frames or end <= start:
        return SegmentAudioStats()

    i0 = max(int(start * FEATURE_FPS), 0)
    i1 = min(int(end * FEATURE_FPS), feats.frames)

    rms = np.asarray(feats.rms, dtype=float)
    window = rms[i0:i1] if i1 > i0 else rms[i0 : i0 + 1]
    global_max = float(rms.max()) if len(rms) else 1e-9
    global_mean = float(rms.mean()) if len(rms) else 0.0

    stats = SegmentAudioStats()
    stats.energy_peak = float(window.max()) / max(global_max, 1e-9)
    stats.energy_score = _sigmoid(
        (float(window.mean()) - global_mean) / max(global_mean, 1e-9)
    )

    if feats.reaction_prob:
        react = np.asarray(feats.reaction_prob, dtype=float)[i0:i1]
        stats.laughter_score = float(react.mean())
        stats.laughter_peak = float(react.max())

    silence = np.asarray(feats.silence, dtype=float)[i0:i1] if feats.silence else np.zeros_like(window)
    stats.silence_ratio = float(silence.mean())

    # speech rate: words spoken per second, normalized to 0..1 (~4 words/s is fast)
    n_words = sum(1 for w in words if start - 0.05 <= getattr(w, "start", -1) <= end)
    stats.speech_rate = min(n_words / max(end - start, 0.1) / 3.5, 1.0)

    if feats.pitch and feats.voiced:
        pitch = np.asarray(feats.pitch, dtype=float)
        voiced = np.asarray(feats.voiced, dtype=bool)
        seg_voiced = voiced[i0:i1]
        if seg_voiced.any():
            seg_pitch = pitch[i0:i1][seg_voiced]
            # normalize std by a typical speech pitch spread (40 Hz)
            stats.pitch_variability = min(float(seg_pitch.std()) / 40.0, 1.0)

    # dramatic pause: longest silence run in the 3s before the energy peak
    stats.pre_pause = _pre_peak_pause(rms, feats.silence, i0, i1)

    return stats


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + pow(2.718281828459045, -max(min(x, 12), -12)))


def _pre_peak_pause(rms, silence: list[bool], i0: int, i1: int) -> float:
    """Longest silent run immediately preceding the window's loudest frame."""
    if not silence or i1 <= i0:
        return 0.0
    import numpy as np

    window = rms[i0:i1]
    if len(window) == 0:
        return 0.0
    peak_offset = int(np.argmax(window))
    peak_idx = i0 + peak_offset
    run = 0
    best = 0
    for i in range(max(i0, peak_idx - 30), peak_idx):  # up to 3s before
        if i < len(silence) and silence[i]:
            run += 1
            best = max(best, run)
        else:
            run = 0
    # 1.5s+ pause before a peak is a strong punchline signal
    return min(best / FEATURE_FPS / 1.5, 1.0)


def save_features(feats: AudioFeatures, path: str | Path) -> Path:
    Path(path).write_text(json.dumps(feats.to_dict()), encoding="utf-8")
    return Path(path)


def load_features(path: str | Path) -> AudioFeatures:
    return AudioFeatures.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
