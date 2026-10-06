"""Candidate generation + multi-signal scoring + ranking.

Pipeline:
  1. Build sentence spans from word timestamps.
  2. Slide windows of the user's target duration across sentences (snap to
     boundaries, never mid-word).
  3. Score every window with the weighted ensemble:
       llm_overall, hook, emotion, audio_energy, laughter, completeness, visual
     plus interaction (banter) and visual/scene signals when available.
  4. Deduplicate near-identical windows by text similarity.
  5. Rank, enforce the user's category filter, and emit Top-N clips.
"""

from __future__ import annotations

import math
import math
from dataclasses import dataclass, field

from app.core.config import settings
from app.core.logging import get_logger
from app.services import audio_analysis, visual_analysis
from app.services.diarize import turn_change_count
from app.services.llm_analyzer import CATEGORIES, SentenceScore
from app.services.segments import (
    Word,
    merge_overlapping,
    sentence_boundaries,
    trim_words_to_range,
)

log = get_logger("scoring")


@dataclass
class Candidate:
    """A scored candidate clip (not yet persisted)."""

    start: float
    end: float
    duration: float
    text: str
    words: list[Word]
    sentence_indices: list[int]
    scores: dict[str, float] = field(default_factory=dict)
    viral_score: float = 0.0
    category_primary: str = ""
    category_secondary: str = ""
    reasons: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    emojis: list[str] = field(default_factory=list)
    llm: SentenceScore | None = None


# --------------------------------------------------------------------------
# Candidate generation
# --------------------------------------------------------------------------
def build_sentence_groups(
    words: list[Word],
    *,
    max_seconds: float = 30.0,
) -> list[dict]:
    """Group words into sentence dicts with index/start/end/text.

    Very long sentences (rare in speech) are split at clause breaks so no single
    unit exceeds ~max_seconds.
    """
    if not words:
        return []
    spans = sentence_boundaries(words)
    out: list[dict] = []
    for s_idx, e_idx in spans:
        group_words = words[s_idx : e_idx + 1]
        start = group_words[0].start
        end = group_words[-1].end
        text = " ".join(w.word for w in group_words).strip()
        text = " ".join(text.split())
        duration = end - start
        if duration > max_seconds and len(group_words) >= 4:
            # Rare in speech (run-on captioning): split into ceil(dur/max) parts,
            # preferring clause boundaries so we never cut mid-word.
            n_parts = max(2, math.ceil(duration / max_seconds))
            for part in _split_words(group_words, n_parts):
                out.append(
                    {
                        "index": len(out),
                        "start": part[0].start,
                        "end": part[-1].end,
                        "text": " ".join(w.word for w in part),
                    }
                )
            continue
        out.append({"index": len(out), "start": start, "end": end, "text": text})
    return out


def _split_words(group_words: list[Word], n_parts: int) -> list[list[Word]]:
    """Split a word list into `n_parts` near-equal chunks at clause boundaries."""
    from app.services.segments import is_clause_end

    target = math.ceil(len(group_words) / n_parts)
    parts: list[list[Word]] = []
    i = 0
    while i < len(group_words):
        window = group_words[i : i + target + 2]
        cut = None
        for offset in range(target, min(len(window), target + 2)):
            if is_clause_end(window[offset]):
                cut = offset
                break
        if cut is None:
            cut = min(target, len(group_words) - i)
        parts.append(group_words[i : i + cut])
        i += cut
    return [p for p in parts if p]


def generate_candidates(
    sentences: list[dict],
    words: list[Word],
    *,
    dur_min: float,
    dur_max: float,
    duration_step: float = 3.0,
) -> list[tuple[float, float, list[int]]]:
    """Enumerate (start, end, sentence_indices) windows inside the target band.

    Windows start at every sentence and extend through subsequent sentences
    until the duration lands in [dur_min, dur_max] (or overshoots, whichever
    comes first). Overlap is allowed at this stage — ranking + dedup handle it.
    """
    if not sentences:
        return []

    candidates: list[tuple[float, float, list[int]]] = []
    n = len(sentences)
    for i in range(n):
        s_start = sentences[i]["start"]
        for j in range(i, n):
            end = sentences[j]["end"]
            duration = end - s_start
            if duration < dur_min:
                continue
            if duration > dur_max:
                # shrink by dropping trailing sentences once, then stop
                if j > i:
                    end = sentences[j - 1]["end"]
                    duration = end - s_start
                    if dur_min <= duration <= dur_max:
                        candidates.append((s_start, end, list(range(i, j))))
                break
            candidates.append((s_start, end, list(range(i, j + 1))))
            if duration > dur_max - duration_step:
                break  # next extension overshoots; move the start forward

    # dedupe identical ranges
    seen = set()
    unique: list[tuple[float, float, list[int]]] = []
    for c in candidates:
        key = (round(c[0], 2), round(c[1], 2))
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def _text_similarity(a: str, b: str) -> float:
    """Jaccard similarity over word sets — cheap dedup for near-identical clips."""
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def score_candidates(
    candidates: list[tuple[float, float, list[int]]],
    sentences: list[dict],
    words: list[Word],
    llm_index: dict[int, SentenceScore],
    llm_chunk_overall: list[float],
    feats: audio_analysis.AudioFeatures | None,
    diarization_turns: list[dict],
    *,
    visual_feats: visual_analysis.VisualFeatures | None = None,
    weights: dict[str, float] | None = None,
) -> list[Candidate]:
    """Score and rank all candidate windows; returns best-first list.

    `visual_feats` supplies the scene-cut / face-presence signals for the
    `visual` dimension. When absent the dimension falls back to neutral 0.5
    so scoring degrades gracefully rather than dropping the signal to zero.
    """
    w = weights or settings.score_weights
    scored: list[Candidate] = []
    visual_lookup: dict[tuple[float, float], float] = {}

    # Precompute the visual dimension per candidate window once; it is the
    # only per-window signal here that needs a video pass.
    if visual_feats is not None and visual_feats.times:
        for start, end, _ in candidates:
            stats = visual_analysis.clip_stats(visual_feats, start, end)
            visual_lookup[(round(start, 3), round(end, 3))] = visual_analysis.visual_score(stats)

    for start, end, indices in candidates:
        window_words = trim_words_to_range(words, start, end)
        text = " ".join(s["text"] for s in (sentences[i] for i in indices))

        llm_scores = [llm_index[i] for i in indices if i in llm_index]
        if not llm_scores:
            continue

        # --- per-dimension scores (each 0..1)
        llm_overall = _mean(llm_chunk_overall) if llm_chunk_overall else 0.5
        hook = max((s.hook_strength for s in llm_scores), default=0.0)
        emotion = max((s.emotion for s in llm_scores), default=0.0)
        completeness = min((s.idea_completeness for s in llm_scores), default=0.0)
        clarity = min((s.standalone_clarity for s in llm_scores), default=0.0)
        quote = max((s.quoteability for s in llm_scores), default=0.0)

        audio_stats = None
        energy = laughter = pre_pause = 0.0
        if feats is not None and feats.frames:
            audio_stats = audio_analysis.segment_stats(feats, start, end, window_words)
            energy = audio_stats.energy_score
            laughter = audio_stats.laughter_score
            pre_pause = audio_stats.pre_pause

        # visual: scene cuts + face presence + on-camera motion (Fase 3).
        # Neutral when the video pass is unavailable.
        visual = 0.5
        if visual_lookup:
            visual = visual_lookup.get((round(start, 3), round(end, 3)), 0.5)

        # banter: fast speaker switching adds energy to conversation clips
        banter = min(turn_change_count(diarization_turns, start, end) / 6.0, 1.0)

        # --- weighted ensemble
        dims = {
            "llm_overall": llm_overall,
            "hook": hook,
            "emotion": emotion,
            "audio_energy": energy,
            "laughter": laughter,
            "completeness": completeness,
            "visual": visual,
        }
        total_weight = 0.0
        viral = 0.0
        for key, value in dims.items():
            weight = w.get(key, 0.0)
            viral += weight * value
            total_weight += weight
        viral = viral / max(total_weight, 1e-9)

        # bonus signals (additive, capped)
        bonus = 0.0
        bonus += 0.05 * banter
        bonus += 0.05 * pre_pause
        bonus += 0.04 * quote
        bonus += 0.03 * clarity
        viral = min(viral + bonus, 1.0)

        # --- categories
        cats: dict[str, float] = {}
        for s in llm_scores:
            if s.category:
                cats[s.category] = cats.get(s.category, 0.0) + s.category_confidence
        ordered = sorted(cats.items(), key=lambda kv: kv[1], reverse=True)
        primary = ordered[0][0] if ordered else ""
        secondary = ordered[1][0] if len(ordered) > 1 else ""

        reasons: list[str] = []
        if hook > 0.7:
            reasons.append("Strong hook in the first sentence")
        if laughter > 0.4:
            reasons.append("Laughter/applause detected in this moment")
        if energy > 0.7:
            reasons.append("Energy spike relative to the rest of the audio")
        if completeness > 0.7:
            reasons.append("Self-contained idea with a clear beginning and end")
        if banter > 0.5:
            reasons.append("Fast back-and-forth between speakers")
        if pre_pause > 0.6:
            reasons.append("Dramatic pause right before the punchline")
        top_reason = next((s.reason for s in llm_scores if s.reason), "")
        if top_reason:
            reasons.append(top_reason)
        if not reasons:
            reasons.append("Solid all-round moment")

        keywords = sorted({k for s in llm_scores for k in s.keywords})[:10]
        emojis = sorted({e for s in llm_scores for e in s.emojis})[:4]

        scored.append(
            Candidate(
                start=start,
                end=end,
                duration=end - start,
                text=text,
                words=window_words,
                sentence_indices=indices,
                scores={
                    **dims,
                    "clarity": clarity,
                    "quote": quote,
                    "banter": banter,
                    "pre_pause": pre_pause,
                    "speech_rate": audio_stats.speech_rate if audio_stats else 0.0,
                    "silence_ratio": audio_stats.silence_ratio if audio_stats else 0.0,
                },
                viral_score=viral,
                category_primary=primary,
                category_secondary=secondary,
                reasons=reasons[:5],
                keywords=keywords,
                emojis=emojis,
            )
        )

    scored.sort(key=lambda c: c.viral_score, reverse=True)
    return scored


def deduplicate(scored: list[Candidate], *, similarity_threshold: float = 0.6) -> list[Candidate]:
    """Drop near-identical clips (Jaccard over words), keeping the higher score."""
    kept: list[Candidate] = []
    for cand in scored:
        duplicate = any(
            _text_similarity(cand.text, other.text) >= similarity_threshold
            and abs(cand.start - other.start) < (cand.duration * 0.5)
            for other in kept
        )
        if not duplicate:
            kept.append(cand)
    return kept


def enforce_non_overlap(scored: list[Candidate]) -> list[Candidate]:
    """Greedy interval scheduling: no two returned clips may overlap."""
    kept: list[Candidate] = []
    for cand in scored:
        if all(cand.start >= other.end - 0.05 or cand.end <= other.start + 0.05 for other in kept):
            kept.append(cand)
    return kept


def filter_by_categories(cands: list[Candidate], categories: list[str]) -> list[Candidate]:
    """Keep clips whose primary or secondary category is requested (case-insensitive)."""
    if not categories:
        return cands
    wanted = {c.lower() for c in categories}
    return [
        c
        for c in cands
        if c.category_primary in wanted or c.category_secondary in wanted
    ]


def select_top_n(
    scored: list[Candidate],
    *,
    top_n: int,
    categories: list[str] | None = None,
    min_score: float | None = None,
    allow_overlap: bool = False,
) -> list[Candidate]:
    """Apply category filter, dedup, non-overlap, min-score, and Top-N cut."""
    pool = filter_by_categories(scored, categories or [])
    pool = deduplicate(pool)
    if not allow_overlap:
        pool = enforce_non_overlap(pool)
    threshold = settings.MIN_CLIP_SCORE if min_score is None else min_score
    pool = [c for c in pool if c.viral_score * 100 >= threshold]
    return pool[:top_n]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# --------------------------------------------------------------------------
# Title / description / hashtag generation
# --------------------------------------------------------------------------
def build_title(cand: Candidate, *, language: str = "en") -> str:
    """Generate a click-worthy title from the strongest sentence."""
    if not cand.words:
        return ""
    # Use the first 6-10 words of the clip as the base, cleaned of punctuation
    head = cand.words[:9]
    text = " ".join(w.word for w in head).strip()
    text = " ".join(text.split())
    text = text.strip("\"'“”,.!?- ")

    suffixes = (
        [" 🤯", " 🔥", " 💡", " 😂", " ⚡"]
        if cand.category_primary == "funny"
        else [" 💡", " 🎯", " 📈", " 🧠"]
        if cand.category_primary == "insight"
        else [" ❤️", " 🥹"]
        if cand.category_primary == "emotional"
        else [" 🔥", " ⚡", " 🎯"]
    )
    suffix = cand.emojis[0] if cand.emojis else suffixes[0]
    title = f"{text}{suffix}"
    if len(title) > 100:
        title = title[:97].rstrip() + "…"
    return title


def reorder_for_hook(
    sentences: list[dict],
    llm_index: dict[int, SentenceScore],
    clip_sentences: list[int],
) -> list[int]:
    """Hook-first suggestion: promote the strongest sentence to the front.

    Returns the reordered sentence indices. Only suggests a reorder when the
    strongest sentence is not already first AND it can stand alone.
    """
    if len(clip_sentences) < 3:
        return list(clip_sentences)
    scored = sorted(
        clip_sentences,
        key=lambda i: llm_index[i].hook_strength if i in llm_index else 0.0,
        reverse=True,
    )
    best = scored[0]
    if best == clip_sentences[0]:
        return list(clip_sentences)
    if best not in llm_index or llm_index[best].standalone_clarity < 0.55:
        return list(clip_sentences)
    reordered = [best] + [i for i in clip_sentences if i != best]
    log.info("hook-first reorder suggested: %s -> %s", clip_sentences, reordered)
    return reordered


def build_description(cand: Candidate) -> str:
    """One-paragraph description anchored on the clip's transcript."""
    text = " ".join(cand.text.split())
    if len(text) > 220:
        text = text[:217].rstrip() + "…"
    parts = [text]
    if cand.reasons:
        parts.append(f"\n\nWhy this clip: {cand.reasons[0]}.")
    return "".join(parts)


def build_hashtags(cand: Candidate, *, language: str = "en") -> str:
    """Hashtags from category + keywords, TikTok-friendly."""
    tags = ["#fyp", "#foryou", "#viral", "#clipforge"]
    if cand.category_primary:
        tags.append(f"#{cand.category_primary}")
    if cand.category_secondary and cand.category_secondary != cand.category_primary:
        tags.append(f"#{cand.category_secondary}")
    for kw in cand.keywords[:5]:
        tag = "".join(ch for ch in kw if ch.isalnum())[:24]
        if len(tag) >= 3 and f"#{tag.lower()}" not in {t.lower() for t in tags}:
            tags.append(f"#{tag.lower()}")
    # de-dup preserving order, cap at 10
    seen: set[str] = set()
    out: list[str] = []
    for t in tags:
        key = t.lower()
        if key not in seen:
            seen.add(key)
            out.append(t)
    return " ".join(out[:10])
