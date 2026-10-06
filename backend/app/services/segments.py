"""Sentence-boundary snapping using word-level timestamps.

Clips must start at a sentence start and end at a sentence end — never
mid-word. This module owns that logic and is unit-tested.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Sentence terminators (latin + common CJK punctuation handled loosely)
_TERMINATORS = ".!?。！？…"
_CLAUSE_BREAKS = ",;:、，；"
# A "word" here is a token from the transcriber: may carry leading punctuation
# or trailing punctuation, e.g. '"Hello,' or 'world.'


@dataclass
class Word:
    """A word with timing."""

    word: str
    start: float
    end: float
    speaker: str | None = None
    confidence: float = 1.0

    @property
    def clean(self) -> str:
        """Alphanumeric content only, for heuristics."""
        return re.sub(r"[^\w]", "", self.word)


def is_sentence_end(w: Word) -> bool:
    """True if the word ends a sentence (trailing terminator)."""
    stripped = w.word.rstrip("\"\')]}”’›")
    return bool(stripped) and stripped[-1] in _TERMINATORS


def is_sentence_start(w: Word, prev: Word | None) -> bool:
    """True if the word begins a new sentence."""
    if prev is None:
        return True
    # after a sentence end, or after a long pause
    if is_sentence_end(prev):
        return True
    if w.start - prev.end >= 1.0:  # 1s silence counts as a boundary
        return True
    # explicit capitalization after clause break (helps for comma-heavy text)
    lead = w.word.lstrip("\"\'([{“‘‹")
    if lead[:1].isupper() and is_clause_end(prev):
        return True
    return False


def is_clause_end(w: Word) -> bool:
    stripped = w.word.rstrip("\"\')]}”’›")
    return bool(stripped) and stripped[-1] in (_CLAUSE_BREAKS + _TERMINATORS)


def sentence_boundaries(words: list[Word]) -> list[tuple[int, int]]:
    """Group word indices into sentence spans: [(start_idx, end_idx_inclusive)].

    Guarantees:
      - spans are contiguous and cover every word,
      - the first span starts at index 0,
      - empty input returns [].
    """
    if not words:
        return []
    spans: list[tuple[int, int]] = []
    start = 0
    for i in range(1, len(words)):
        if is_sentence_start(words[i], words[i - 1]):
            spans.append((start, i - 1))
            start = i
    spans.append((start, len(words) - 1))
    return spans


def snap_to_sentence_bounds(
    words: list[Word],
    t_start: float,
    t_end: float,
    *,
    prefer_expand: bool = True,
) -> tuple[float, float]:
    """Snap [t_start, t_end] outward to enclosing sentence boundaries.

    `prefer_expand=True` grows the window to include whole sentences that
    overlap the requested range; `False` shrinks to the largest contained
    sentence span.
    """
    if not words:
        return t_start, t_end

    spans = sentence_boundaries(words)
    best: tuple[float, float] | None = None
    if prefer_expand:
        # Union of every sentence span overlapping the requested window.
        hits = [
            (words[s].start, words[e].end)
            for s, e in spans
            if words[s].start <= t_end and words[e].end >= t_start
        ]
        if hits:
            best = (min(h[0] for h in hits), max(h[1] for h in hits))
    else:
        # Largest sentence span fully contained in the window.
        for s_idx, e_idx in spans:
            s, e = words[s_idx].start, words[e_idx].end
            if s >= t_start and e <= t_end:
                if best is None or (e - s) > (best[1] - best[0]):
                    best = (s, e)

    if best is None:
        # Fall back to the nearest sentence span entirely outside the window.
        if prefer_expand:
            left = [
                (words[s].start, words[e].end) for s, e in spans if words[e].end <= t_start
            ]
            right = [
                (words[s].start, words[e].end) for s, e in spans if words[s].start >= t_end
            ]
            if right:
                best = right[0]
            elif left:
                best = left[-1]
        if best is None:
            best = (words[0].start, words[-1].end)

    return max(best[0], 0.0), best[1]


def merge_overlapping(ranges: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Sort and merge overlapping (start, end) ranges."""
    if not ranges:
        return []
    ordered = sorted(ranges, key=lambda r: (r[0], r[1]))
    merged = [ordered[0]]
    for s, e in ordered[1:]:
        ls, le = merged[-1]
        if s <= le:
            merged[-1] = (ls, max(le, e))
        else:
            merged.append((s, e))
    return merged


def trim_words_to_range(
    words: list[Word], t_start: float, t_end: float
) -> list[Word]:
    """Return the words that fall inside [t_start, t_end] (inclusive)."""
    return [w for w in words if w.start >= t_start - 1e-6 and w.end <= t_end + 1e-6]
