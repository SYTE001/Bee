"""Tests for the precision@k evaluation metrics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.precision_at_k import (
    Label,
    Prediction,
    Span,
    best_iou,
    evaluate,
    iou,
    load_labels,
    precision_at_k,
    recall,
)


def _pred(start: float, end: float, *, rank: int, score: float = 50.0, category: str = "") -> Prediction:
    return Prediction(Span(start, end), score=score, category=category, rank=rank, title=f"clip {rank}")


def _label(start: float, end: float, *, category: str = "") -> Label:
    return Label(Span(start, end), category=category)


# --------------------------------------------------------------------------
# IoU
# --------------------------------------------------------------------------
class TestIoU:
    def test_identical(self):
        assert iou(Span(0, 10), Span(0, 10)) == pytest.approx(1.0)

    def test_disjoint(self):
        assert iou(Span(0, 10), Span(20, 30)) == 0.0

    def test_adjacent(self):
        # touching spans share a zero-length intersection
        assert iou(Span(0, 10), Span(10, 20)) == 0.0

    def test_partial(self):
        assert iou(Span(0, 10), Span(5, 15)) == pytest.approx(5 / 15)

    def test_contained(self):
        # a 2s clip inside a 10s label
        assert iou(Span(4, 6), Span(0, 10)) == pytest.approx(0.2)

    def test_degenerate(self):
        assert iou(Span(5, 5), Span(0, 10)) == 0.0


# --------------------------------------------------------------------------
# precision@k
# --------------------------------------------------------------------------
class TestPrecisionAtK:
    labels = [_label(0, 10), _label(20, 30), _label(40, 50)]

    def test_all_hits(self):
        preds = [_pred(0, 10, rank=1), _pred(20, 30, rank=2), _pred(40, 50, rank=3)]
        assert precision_at_k(preds, self.labels, 3, iou_threshold=0.3) == 1.0

    def test_top_k_misses(self):
        preds = [_pred(100, 110, rank=1), _pred(0, 10, rank=2)]
        assert precision_at_k(preds, self.labels, 1, iou_threshold=0.3) == 0.0
        assert precision_at_k(preds, self.labels, 2, iou_threshold=0.3) == 0.5

    def test_threshold_boundaries(self):
        # IoU exactly 0.2 with a 10s label (2s clip in the middle)
        preds = [_pred(4, 6, rank=1)]
        assert precision_at_k(preds, self.labels, 1, iou_threshold=0.3) == 0.0
        assert precision_at_k(preds, self.labels, 1, iou_threshold=0.2) == 1.0

    def test_empty_predictions(self):
        assert precision_at_k([], self.labels, 5, iou_threshold=0.3) == 0.0

    def test_empty_labels(self):
        preds = [_pred(0, 10, rank=1)]
        assert precision_at_k(preds, [], 1, iou_threshold=0.3) == 0.0

    def test_zero_and_negative_k(self):
        preds = [_pred(0, 10, rank=1)]
        assert precision_at_k(preds, self.labels, 0, iou_threshold=0.3) == 0.0

    def test_k_larger_than_predictions(self):
        # only 2 predictions: @10 counts the 2 that exist
        preds = [_pred(0, 10, rank=1), _pred(100, 110, rank=2)]
        assert precision_at_k(preds, self.labels, 10, iou_threshold=0.3) == 0.5

    def test_category_tightens_match(self):
        labels = [_label(0, 10, category="insight")]
        match = _pred(0, 10, rank=1, category="insight")
        mismatch = _pred(0, 10, rank=1, category="funny")
        assert precision_at_k([match], labels, 1) == 1.0
        assert precision_at_k([mismatch], labels, 1) == 0.0

    def test_unlabelled_category_is_lenient(self):
        labels = [_label(0, 10)]
        pred = _pred(0, 10, rank=1, category="funny")
        assert precision_at_k([pred], labels, 1) == 1.0


# --------------------------------------------------------------------------
# recall
# --------------------------------------------------------------------------
class TestRecall:
    labels = [_label(0, 10), _label(20, 30)]

    def test_full_coverage(self):
        preds = [_pred(0, 10, rank=1), _pred(20, 30, rank=2)]
        assert recall(preds, self.labels, iou_threshold=0.3) == 1.0

    def test_partial_coverage(self):
        preds = [_pred(0, 10, rank=1)]
        assert recall(preds, self.labels, iou_threshold=0.3) == 0.5

    def test_no_coverage(self):
        preds = [_pred(100, 110, rank=1)]
        assert recall(preds, self.labels, iou_threshold=0.3) == 0.0

    def test_no_labels(self):
        assert recall([_pred(0, 10, rank=1)], [], iou_threshold=0.3) == 0.0


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
class TestEvaluate:
    def test_report_shape(self):
        preds = [_pred(0, 10, rank=1, score=88.0), _pred(100, 110, rank=2, score=40.0)]
        report = evaluate(preds, [_label(0, 10)], ks=(1, 2), iou_threshold=0.3)
        assert report["n_labels"] == 1
        assert report["n_predictions"] == 2
        assert report["precision_at_k"]["1"] == 1.0
        assert report["precision_at_k"]["2"] == 0.5
        assert report["recall"] == 1.0
        assert len(report["hits"]) == 2
        assert report["hits"][0]["hit"] is True
        assert report["hits"][1]["hit"] is False

    def test_predictions_sorted_by_rank(self):
        preds = [_pred(100, 110, rank=2), _pred(0, 10, rank=1)]
        report = evaluate(preds, [_label(0, 10)], ks=(1,))
        assert report["hits"][0]["rank"] == 1
        assert report["precision_at_k"]["1"] == 1.0

    def test_best_iou(self):
        preds = [_pred(4, 6, rank=1)]
        assert best_iou(preds[0], [_label(0, 10)]) == pytest.approx(0.2)
        assert best_iou(preds[0], []) == 0.0


# --------------------------------------------------------------------------
# labels file parsing
# --------------------------------------------------------------------------
class TestLoadLabels:
    def test_parses_projects(self, tmp_path: Path):
        path = tmp_path / "labels.json"
        path.write_text(
            json.dumps(
                {
                    "projects": [
                        {
                            "source": "ep01.mp4",
                            "labels": [
                                {"start": 1, "end": 5, "category": "insight"},
                                {"start": 10, "end": 20, "note": "laugh"},
                            ],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        labels = load_labels(path)
        assert list(labels) == ["ep01.mp4"]
        assert len(labels["ep01.mp4"]) == 2
        first = labels["ep01.mp4"][0]
        assert first.span.start == 1.0 and first.span.end == 5.0
        assert first.category == "insight"
        assert labels["ep01.mp4"][1].category == ""

    def test_missing_source_is_skipped(self, tmp_path: Path):
        path = tmp_path / "labels.json"
        path.write_text(json.dumps({"projects": [{"labels": []}]}), encoding="utf-8")
        assert load_labels(path) == {}
