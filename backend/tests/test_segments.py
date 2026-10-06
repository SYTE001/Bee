"""Tests for sentence-snapping — the core clip-quality guarantee."""

import pytest

from app.services.segments import (
    Word,
    merge_overlapping,
    sentence_boundaries,
    snap_to_sentence_bounds,
    trim_words_to_range,
)


def make_words(text: str, step: float = 0.4) -> list[Word]:
    """Turn a sentence into evenly-spaced Word objects."""
    words = []
    t = 0.0
    for w in text.split():
        words.append(Word(word=w, start=t, end=t + step * 0.8))
        t += step
    return words


class TestSentenceBoundaries:
    def test_single_sentence_is_one_span(self):
        words = make_words("hello world this is one sentence")
        spans = sentence_boundaries(words)
        assert spans == [(0, len(words) - 1)]

    def test_period_splits_sentence(self):
        words = make_words("First one. Second two. Third three.")
        spans = sentence_boundaries(words)
        assert len(spans) == 3
        # contiguous coverage
        assert spans[0][0] == 0
        assert spans[-1][1] == len(words) - 1
        for prev, nxt in zip(spans, spans[1:]):
            assert nxt[0] == prev[1] + 1

    def test_long_pause_splits_sentence(self):
        words = make_words("a b c d e f")
        # insert a 2s gap between index 2 and 3
        for w in words[3:]:
            w.start += 2.0
            w.end += 2.0
        spans = sentence_boundaries(words)
        assert len(spans) == 2
        assert spans[0] == (0, 2)
        assert spans[1] == (3, 5)

    def test_empty(self):
        assert sentence_boundaries([]) == []

    def test_exclamation_and_question_marks(self):
        words = make_words("Really? Yes! Okay.")
        assert len(sentence_boundaries(words)) == 3


class TestSnap:
    def test_snap_expands_to_sentence_bounds(self):
        words = make_words("Hello there friend. Another sentence here. Third one.")
        # request a window that cuts the first sentence in half
        start, end = snap_to_sentence_bounds(words, 0.6, 0.9)
        assert start == words[0].start
        assert end == words[2].end  # "friend."

    def test_snap_includes_full_overlapping_sentences(self):
        words = make_words("A b c. D e f. G h i.")
        start, end = snap_to_sentence_bounds(words, 1.2, 1.6)
        assert start == words[3].start
        assert end == words[5].end

    def test_shrink_finds_largest_contained_sentence(self):
        words = make_words("A b c. D e f. G h i.")
        # each sentence spans ~1.12s; window [0, 2.4] contains the first two
        start, end = snap_to_sentence_bounds(words, 0.0, 2.4, prefer_expand=False)
        assert (end - start) == pytest.approx(1.12, rel=1e-3)

    def test_never_mid_word_when_expand(self):
        words = make_words("The quick brown fox jumps. Over the lazy dog now.")
        start, end = snap_to_sentence_bounds(words, 1.0, 1.3)
        # boundaries must coincide with actual word times, never mid-word
        assert start in {w.start for w in words}
        assert end in {w.end for w in words}

    def test_empty_words_returns_input(self):
        assert snap_to_sentence_bounds([], 1.0, 2.0) == (1.0, 2.0)

    def test_window_before_all_words(self):
        words = make_words("Only one sentence exists here.")
        start, end = snap_to_sentence_bounds(words, -5.0, -1.0)
        assert start == words[0].start
        assert end == words[-1].end

    def test_start_never_negative(self):
        words = [Word(word="x", start=-0.5, end=0.0)]
        start, _ = snap_to_sentence_bounds(words, -1.0, 1.0)
        assert start >= 0.0


class TestMergeOverlapping:
    def test_merges(self):
        merged = merge_overlapping([(0, 2), (1, 3), (5, 6)])
        assert merged == [(0, 3), (5, 6)]

    def test_empty(self):
        assert merge_overlapping([]) == []

    def test_touching_ranges_merge(self):
        assert merge_overlapping([(0, 1), (1, 2)]) == [(0, 2)]


class TestTrimWords:
    def test_trim(self):
        words = make_words("a b c d e")
        trimmed = trim_words_to_range(words, 0.4, 1.6)
        assert [w.word for w in trimmed] == ["b", "c", "d"]

    def test_empty(self):
        assert trim_words_to_range([], 0, 1) == []
