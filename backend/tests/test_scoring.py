"""Tests for candidate generation, multi-signal scoring, and selection."""

from __future__ import annotations

from app.services import audio_analysis
from app.services.llm_analyzer import SentenceScore
from app.services.scoring import (
    Candidate,
    build_description,
    build_hashtags,
    build_sentence_groups,
    build_title,
    deduplicate,
    enforce_non_overlap,
    filter_by_categories,
    generate_candidates,
    reorder_for_hook,
    score_candidates,
    select_top_n,
    _text_similarity,
)
from app.services.segments import Word


def make_words(text: str, step: float = 0.4) -> list[Word]:
    ws, t = [], 0.0
    for w in text.split():
        ws.append(Word(word=w, start=t, end=t + step * 0.8))
        t += step
    return ws


class TestSentenceGroups:
    def test_groups_have_index_and_timing(self):
        words = make_words("Hello there. Another sentence. Third one here now.")
        groups = build_sentence_groups(words)
        assert [g["index"] for g in groups] == [0, 1, 2]
        assert groups[0]["start"] == 0.0
        # "Hello there." is two words; the group ends with "there."
        assert groups[0]["end"] == words[1].end
        assert "Hello" in groups[0]["text"]

    def test_empty(self):
        assert build_sentence_groups([]) == []

    def test_long_sentence_is_split(self):
        # one unbroken run of 20 words at normal speech pace (0.4s/word = 8s)
        words = make_words(" ".join(f"word{i}" for i in range(20)))
        groups = build_sentence_groups(words, max_seconds=2.0)
        assert len(groups) >= 4
        for g in groups:
            assert g["end"] - g["start"] <= 2.0 + 1e-6, g


class TestGenerateCandidates:
    def test_respects_duration_band(self):
        sentences = [
            {"index": i, "start": i * 2.0, "end": i * 2.0 + 1.5, "text": f"s{i}"}
            for i in range(20)
        ]
        cands = generate_candidates(sentences, [], dur_min=5.0, dur_max=10.0)
        assert len(cands) > 0
        for start, end, _ in cands:
            dur = end - start
            assert 5.0 <= dur <= 10.0, dur

    def test_starts_on_sentence_boundary(self):
        sentences = [
            {"index": i, "start": i * 2.0, "end": i * 2.0 + 1.5, "text": f"s{i}"}
            for i in range(10)
        ]
        cands = generate_candidates(sentences, [], dur_min=3.0, dur_max=8.0)
        starts = {round(c[0], 2) for c in cands}
        sentence_starts = {round(s["start"], 2) for s in sentences}
        assert starts.issubset(sentence_starts)

    def test_empty_sentences(self):
        assert generate_candidates([], [], dur_min=5, dur_max=10) == []


class TestScoring:
    def _setup(self):
        text = "First sentence here. Second one is much better and funnier. Third one ends it."
        words = make_words(text)
        sentences = build_sentence_groups(words)
        llm_index = {
            0: SentenceScore(
                sentence_index=0,
                category="story",
                category_confidence=0.5,
                hook_strength=0.3,
                emotion=0.2,
                idea_completeness=0.6,
                standalone_clarity=0.5,
                quoteability=0.3,
                reason="opener",
                keywords=["first"],
            ),
            1: SentenceScore(
                sentence_index=1,
                category="funny",
                category_confidence=0.9,
                hook_strength=0.9,
                emotion=0.8,
                idea_completeness=0.8,
                standalone_clarity=0.85,
                quoteability=0.8,
                reason="strong punchline",
                keywords=["funny", "better"],
                emojis=["😂"],
            ),
            2: SentenceScore(
                sentence_index=2,
                category="story",
                category_confidence=0.4,
                hook_strength=0.2,
                emotion=0.1,
                idea_completeness=0.9,
                standalone_clarity=0.6,
                quoteability=0.2,
                reason="resolution",
            ),
        }
        return words, sentences, llm_index

    def test_scores_and_ranks(self):
        words, sentences, llm_index = self._setup()
        cands = generate_candidates(sentences, words, dur_min=0.5, dur_max=5.0)
        scored = score_candidates(
            cands, sentences, words, llm_index, [0.7], None, []
        )
        assert len(scored) == len(cands)
        # every clip is 0..1
        for c in scored:
            assert 0.0 <= c.viral_score <= 1.0
            for v in c.scores.values():
                assert 0.0 <= v <= 1.0
        # descending order
        scores = [c.viral_score for c in scored]
        assert scores == sorted(scores, reverse=True)
        # the clip containing the strong middle sentence wins
        best = scored[0]
        assert "funny" in (best.category_primary, best.category_secondary)
        assert any("punchline" in r for r in best.reasons)

    def test_scores_in_range_without_audio_features(self):
        words, sentences, llm_index = self._setup()
        cands = generate_candidates(sentences, words, dur_min=0.5, dur_max=5.0)
        scored = score_candidates(cands, sentences, words, llm_index, [0.5], None, [])
        assert all(0.0 <= c.viral_score <= 1.0 for c in scored)

    def test_with_audio_features(self):
        words, sentences, llm_index = self._setup()
        feats = audio_analysis.AudioFeatures(duration=6.0)
        feats.rms = [0.1] * 40 + [0.9] * 10 + [0.1] * 10
        feats.silence = [True] * 5 + [False] * 55
        feats.reaction_prob = [0.0] * 30 + [0.8] * 20 + [0.0] * 10
        cands = generate_candidates(sentences, words, dur_min=0.5, dur_max=5.0)
        scored = score_candidates(cands, sentences, words, llm_index, [0.6], feats, [])
        assert all(c.scores["laughter"] >= 0.0 for c in scored)
        # at least one candidate picks up the laughter burst
        assert any(c.scores["laughter"] > 0.1 for c in scored)


class TestSelection:
    def _cand(self, start: float, end: float, score: float, text="", cat="funny") -> Candidate:
        return Candidate(
            start=start,
            end=end,
            duration=end - start,
            text=text or f"clip at {start}",
            words=[],
            sentence_indices=[],
            scores={},
            viral_score=score,
            category_primary=cat,
        )

    def test_dedup_removes_near_identical(self):
        cands = [
            self._cand(0, 5, 0.9, "the quick brown fox jumps over the lazy dog today"),
            self._cand(1, 6, 0.85, "the quick brown fox jumps over the lazy dog today"),
            self._cand(20, 25, 0.8, "completely different content about pizza"),
        ]
        kept = deduplicate(cands)
        assert len(kept) == 2
        assert kept[0].viral_score == 0.9

    def test_enforce_non_overlap(self):
        cands = [self._cand(0, 5, 0.9), self._cand(3, 8, 0.85), self._cand(10, 15, 0.8)]
        kept = enforce_non_overlap(cands)
        assert [(c.start, c.end) for c in kept] == [(0, 5), (10, 15)]

    def test_category_filter(self):
        cands = [self._cand(0, 5, 0.9, cat="funny"), self._cand(10, 15, 0.8, cat="serious")]
        assert len(filter_by_categories(cands, ["funny"])) == 1
        assert len(filter_by_categories(cands, [])) == 2

    def test_select_top_n_respects_limit_and_score(self):
        cands = [self._cand(i * 10, i * 10 + 5, 0.9 - i * 0.1) for i in range(6)]
        # third candidate dips below the default 0.35 floor
        cands.append(self._cand(100, 105, 0.05))
        top = select_top_n(cands, top_n=3)
        assert len(top) == 3
        assert top[0].viral_score == 0.9


class TestSimilarity:
    def test_identical(self):
        assert _text_similarity("a b c d", "a b c d") == 1.0

    def test_disjoint(self):
        assert _text_similarity("a b", "c d") == 0.0

    def test_empty(self):
        assert _text_similarity("", "a") == 0.0


class TestTextGeneration:
    def _cand(self, words, **kw):
        defaults = dict(
            start=0.0,
            end=2.0,
            duration=2.0,
            text=" ".join(w.word for w in words),
            words=words,
            sentence_indices=[],
            scores={},
            viral_score=0.85,
            category_primary="funny",
            category_secondary="",
            keywords=["marketing", "growth"],
            emojis=[],
        )
        defaults.update(kw)
        return Candidate(**defaults)

    def test_title(self):
        words = make_words("This is the best marketing tip nobody talks about.")
        title = build_title(self._cand(words))
        assert 0 < len(title) <= 100
        assert title.split()[0] == "This"

    def test_title_truncated_when_long(self):
        words = make_words(" ".join(f"word{i}" for i in range(30)))
        title = build_title(self._cand(words))
        assert len(title) <= 100

    def test_description(self):
        words = make_words("Short clip text.")
        desc = build_description(self._cand(words, reasons=["Great hook"]))
        assert "Short clip text." in desc
        assert "Great hook" in desc

    def test_hashtags(self):
        words = make_words("x y z.")
        tags = build_hashtags(self._cand(words))
        assert "#fyp" in tags
        assert "#funny" in tags
        assert "#marketing" in tags
        assert len(tags.split()) <= 10

    def test_hook_reorder(self):
        sentences = [
            {"index": 0, "start": 0.0, "end": 1.0, "text": "a"},
            {"index": 1, "start": 1.0, "end": 2.0, "text": "b"},
            {"index": 2, "start": 2.0, "end": 3.0, "text": "c"},
        ]
        llm = {
            0: SentenceScore(sentence_index=0, hook_strength=0.2, standalone_clarity=0.8),
            1: SentenceScore(sentence_index=1, hook_strength=0.9, standalone_clarity=0.8),
            2: SentenceScore(sentence_index=2, hook_strength=0.1, standalone_clarity=0.8),
        }
        reordered = reorder_for_hook(sentences, llm, [0, 1, 2])
        assert reordered[0] == 1

    def test_hook_reorder_skipped_when_best_is_first(self):
        llm = {
            0: SentenceScore(sentence_index=0, hook_strength=0.9, standalone_clarity=0.9),
            1: SentenceScore(sentence_index=1, hook_strength=0.2, standalone_clarity=0.9),
        }
        assert reorder_for_hook([], llm, [0, 1]) == [0, 1]
