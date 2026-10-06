"""Offline evaluation: precision@k of clip selection vs human labels.

Ground truth is a JSON file of hand-picked moments:

```json
{
  "projects": [
    {
      "source": "podcast_ep01.mp4",
      "labels": [
        {"start": 12.0, "end": 45.0, "category": "insight"},
        {"start": 61.0, "end": 88.0, "category": "funny"}
      ]
    }
  ]
}
```

A selected clip counts as a hit when its temporal IoU with some label clears
`--iou` (0.3 by default — a clip that covers the gist of the moment, not
necessarily its exact frame range). Category is checked only when both the
label and the clip carry one.

Usage:
    python -m eval.precision_at_k --labels labels.json
    python -m eval.precision_at_k --labels labels.json --k 1,3,5,10 --iou 0.25
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from app.core.logging import get_logger
from app.db.models.clip import Clip
from app.db.models.project import Project
from app.db.session import SessionLocal

log = get_logger("eval")


# --------------------------------------------------------------------------
# Data shapes
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Span:
    """A time range [start, end) in seconds."""

    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(self.end - self.start, 0.0)


@dataclass(frozen=True)
class Label:
    """A human-labelled viral moment."""

    span: Span
    category: str = ""
    note: str = ""


@dataclass(frozen=True)
class Prediction:
    """A clip the system selected, normalised for scoring."""

    span: Span
    score: float = 0.0
    category: str = ""
    rank: int = 0
    title: str = ""


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def iou(a: Span, b: Span) -> float:
    """Temporal intersection-over-union of two spans."""
    inter = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    union = max(a.end, b.end) - min(a.start, b.start)
    if union <= 0:
        return 0.0
    return inter / union


def matches(prediction: Prediction, labels: list[Label], *, iou_threshold: float) -> bool:
    """True when the prediction overlaps any label past the threshold."""
    return any(
        iou(prediction.span, label.span) >= iou_threshold
        and _category_ok(prediction, label)
        for label in labels
    )


def _category_ok(prediction: Prediction, label: Label) -> bool:
    # Category only tightens the match when the label specifies one.
    if not label.category:
        return True
    if not prediction.category:
        return True
    return prediction.category.lower() == label.category.lower()


def precision_at_k(
    predictions: list[Prediction],
    labels: list[Label],
    k: int,
    *,
    iou_threshold: float = 0.3,
) -> float:
    """Fraction of the top-k predictions that are hits (0..1)."""
    if k <= 0 or not predictions:
        return 0.0
    top = predictions[:k]
    if not top:
        return 0.0
    hits = sum(1 for p in top if matches(p, labels, iou_threshold=iou_threshold))
    return hits / len(top)


def recall(predictions: list[Prediction], labels: list[Label], *, iou_threshold: float = 0.3) -> float:
    """Fraction of labels covered by at least one prediction."""
    if not labels:
        return 0.0
    covered = sum(
        1
        for label in labels
        if any(
            iou(p.span, label.span) >= iou_threshold and _category_ok(p, label)
            for p in predictions
        )
    )
    return covered / len(labels)


def best_iou(prediction: Prediction, labels: list[Label]) -> float:
    """Highest IoU of a prediction against any label (0 when there are none)."""
    if not labels:
        return 0.0
    return max(iou(prediction.span, label.span) for label in labels)


def evaluate(
    predictions: list[Prediction],
    labels: list[Label],
    *,
    ks: tuple[int, ...] = (1, 3, 5, 10),
    iou_threshold: float = 0.3,
) -> dict:
    """Full report for one project."""
    preds = sorted(predictions, key=lambda p: p.rank)
    return {
        "n_labels": len(labels),
        "n_predictions": len(preds),
        "iou_threshold": iou_threshold,
        "precision_at_k": {str(k): precision_at_k(preds, labels, k, iou_threshold=iou_threshold) for k in ks},
        "recall": recall(preds, labels, iou_threshold=iou_threshold),
        "mean_best_iou": (
            sum(best_iou(p, labels) for p in preds) / len(preds) if preds else 0.0
        ),
        "hits": [
            {
                "rank": p.rank,
                "title": p.title,
                "start": p.span.start,
                "end": p.span.end,
                "score": p.score,
                "best_iou": round(best_iou(p, labels), 3),
                "hit": matches(p, labels, iou_threshold=iou_threshold),
            }
            for p in preds
        ],
    }


# --------------------------------------------------------------------------
# Ground-truth loading
# --------------------------------------------------------------------------
def load_labels(path: Path) -> dict[str, list[Label]]:
    """Parse the labels file into {source_name: [Label, ...]}."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, list[Label]] = {}
    for project in raw.get("projects", []):
        source = project.get("source") or project.get("source_name") or ""
        labels = [
            Label(
                span=Span(float(lb["start"]), float(lb["end"])),
                category=str(lb.get("category") or ""),
                note=str(lb.get("note") or ""),
            )
            for lb in project.get("labels", [])
        ]
        if source:
            out[source] = labels
    return out


# --------------------------------------------------------------------------
# Prediction loading (from the ClipForge DB)
# --------------------------------------------------------------------------
def predictions_from_db(source: str, project_id: str | None = None) -> tuple[str, list[Prediction]]:
    """Read the ranked clips stored for a project, keyed by source name."""
    db = SessionLocal()
    try:
        query = db.query(Project)
        if project_id:
            query = query.filter(Project.id == project_id)
        else:
            query = query.filter(Project.source_name == source)
        project = query.order_by(Project.created_at.desc()).first()
        if project is None:
            raise RuntimeError(f"no project found for source {source!r}")
        clips = (
            db.query(Clip)
            .filter(Clip.project_id == project.id)
            .order_by(Clip.rank.asc())
            .all()
        )
        preds = [
            Prediction(
                span=Span(clip.start, clip.end),
                score=clip.viral_score,
                category=clip.category_primary,
                rank=clip.rank,
                title=clip.title,
            )
            for clip in clips
        ]
        return project.source_name, preds
    finally:
        db.close()


def predictions_from_json(path: Path) -> list[Prediction]:
    """Load predictions from a plain JSON list (for headless scorer testing)."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        Prediction(
            span=Span(float(p["start"]), float(p["end"])),
            score=float(p.get("score") or p.get("viral_score") or 0.0),
            category=str(p.get("category") or ""),
            rank=int(p.get("rank") or i + 1),
            title=str(p.get("title") or ""),
        )
        for i, p in enumerate(raw)
    ]


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def _format_pct(v: float) -> str:
    return f"{v * 100:5.1f}%"


def print_report(source: str, report: dict) -> None:
    print(f"\n{'=' * 72}\n{source}\n{'=' * 72}")
    print(f"labels={report['n_labels']}  predictions={report['n_predictions']}  "
          f"IoU>= {report['iou_threshold']}")
    p_at_k = report["precision_at_k"]
    ks = "  ".join(f"@{k}={_format_pct(v)}" for k, v in p_at_k.items())
    print(f"precision {ks}")
    print(f"recall    {_format_pct(report['recall'])}   "
          f"mean best IoU {report['mean_best_iou']:.3f}")
    print("\n  rank  score  best IoU  hit  title")
    for h in report["hits"]:
        flag = "yes" if h["hit"] else " no"
        print(f"  {h['rank']:>4}  {h['score']:>5.1f}  {h['best_iou']:>8.3f}  "
              f"{flag:>3}  {h['title'][:44]}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, type=Path, help="ground-truth JSON file")
    parser.add_argument(
        "--k", default="1,3,5,10",
        help="comma-separated k values for precision@k",
    )
    parser.add_argument("--iou", type=float, default=0.3, help="IoU threshold for a hit")
    parser.add_argument("--project", default=None, help="evaluate only this project id")
    parser.add_argument(
        "--predictions", type=Path, default=None,
        help="evaluate this JSON prediction list against the first project's labels",
    )
    args = parser.parse_args(argv)

    ks = tuple(int(x) for x in str(args.k).split(",") if x.strip())
    labels_by_source = load_labels(args.labels)
    if not labels_by_source:
        print("labels file has no projects", file=sys.stderr)
        return 2

    if args.predictions:
        source, preds = (next(iter(labels_by_source)), predictions_from_json(args.predictions))
        report = evaluate(preds, labels_by_source[source], ks=ks, iou_threshold=args.iou)
        print_report(source, report)
        return 0

    db = SessionLocal()
    try:
        query = db.query(Project)
        if args.project:
            query = query.filter(Project.id == args.project)
        projects = query.order_by(Project.created_at.desc()).all()
    finally:
        db.close()

    evaluated = 0
    macro: list[dict] = []
    for project in projects:
        labels = labels_by_source.get(project.source_name)
        if labels is None:
            continue
        _, preds = predictions_from_db(project.source_name, project.id)
        report = evaluate(preds, labels, ks=ks, iou_threshold=args.iou)
        print_report(project.source_name, report)
        macro.append(report)
        evaluated += 1

    if evaluated == 0:
        print(
            "no project matched a labelled source name; "
            "labelled sources: " + ", ".join(sorted(labels_by_source)),
            file=sys.stderr,
        )
        return 3

    # macro-average across evaluated projects
    print(f"\n{'=' * 72}\nMacro average over {evaluated} project(s)\n{'=' * 72}")
    avg_p = {
        k: sum(r["precision_at_k"][k] for r in macro) / len(macro)
        for k in map(str, ks)
    }
    avg_recall = sum(r["recall"] for r in macro) / len(macro)
    avg_iou = sum(r["mean_best_iou"] for r in macro) / len(macro)
    print(f"precision {'  '.join(f'@{k}={_format_pct(v)}' for k, v in avg_p.items())}")
    print(f"recall    {_format_pct(avg_recall)}   mean best IoU {avg_iou:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
