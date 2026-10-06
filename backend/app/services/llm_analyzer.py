"""LLM-powered transcript analysis (Anthropic Claude).

Sends transcript chunks to Claude with a strict JSON schema, asking for moment
classification (FUNNY / SERIOUS / INSIGHT / EMOTIONAL / CONTROVERSIAL / STORY /
ACTIONABLE), plus hook strength, idea completeness, standalone clarity, and
quote-ability for each sentence group.

Robustness:
  - exponential-backoff retries (rate limit, overloaded, transient 5xx)
  - response is validated against the schema; malformed JSON is retried, then
    a heuristic fallback fills the gap so the pipeline never stalls
  - results are cached per transcript hash so re-runs are free
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from app.core.config import settings
from app.core.logging import get_logger

log = get_logger("llm")

CATEGORIES = [
    "funny",
    "serious",
    "insight",
    "emotional",
    "controversial",
    "story",
    "actionable",
]

SYSTEM_PROMPT = """You are a viral-content analyst for short-form video (TikTok/Reels/Shorts).
You receive a chunk of a podcast/interview/webinar transcript with sentence indices and timestamps.

Analyze it and return ONLY a JSON object matching this schema (no markdown, no commentary):

{
  "sentence_scores": [
    {
      "sentence_index": <int>,
      "category": "funny" | "serious" | "insight" | "emotional" | "controversial" | "story" | "actionable" | null,
      "category_confidence": <0..1>,
      "hook_strength": <0..1>,
      "idea_completeness": <0..1>,
      "standalone_clarity": <0..1>,
      "emotion": <0..1>,
      "quoteability": <0..1>,
      "reason": "<one short clause explaining the judgement>",
      "keywords": ["<important words to highlight>"],
      "emojis": ["<0-3 contextual emoji>"]
    }
  ],
  "overall_interest": <0..1>,
  "summary": "<2 sentence summary of the chunk>"
}

Scoring guidance:
- hook_strength: would the FIRST 3 seconds make a scroller stop? quotable, surprising, or contrarian openings score high.
- idea_completeness: the passage has a beginning, middle, and end (not cut off).
- standalone_clarity: understandable without earlier context.
- quoteability: worth clipping as a single quotable moment.
- Category must be the DOMINANT type of that sentence; null only if truly nothing."""


@dataclass
class SentenceScore:
    sentence_index: int
    category: str | None = None
    category_confidence: float = 0.0
    hook_strength: float = 0.0
    idea_completeness: float = 0.0
    standalone_clarity: float = 0.0
    emotion: float = 0.0
    quoteability: float = 0.0
    reason: str = ""
    keywords: list[str] = field(default_factory=list)
    emojis: list[str] = field(default_factory=list)


@dataclass
class ChunkAnalysis:
    overall_interest: float = 0.0
    summary: str = ""
    sentences: list[SentenceScore] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "overall_interest": self.overall_interest,
            "summary": self.summary,
            "sentences": [s.__dict__ for s in self.sentences],
        }

    @classmethod
    def from_dict(cls, d: dict) -> ChunkAnalysis:
        return cls(
            overall_interest=float(d.get("overall_interest", 0.0)),
            summary=str(d.get("summary", "")),
            sentences=[SentenceScore(**s) for s in d.get("sentences", [])],
        )


class LLMAnalysisError(RuntimeError):
    pass


def client():  # type: ignore[no-untyped-def]
    """Lazily build an Anthropic client; raises if unconfigured."""
    if settings.LLM_PROVIDER == "none" or not settings.ANTHROPIC_API_KEY:
        raise LLMAnalysisError(
            "LLM analysis is disabled — set ANTHROPIC_API_KEY in .env "
            "(falling back to heuristic scoring)"
        )
    try:
        from anthropic import Anthropic
    except ImportError as exc:  # pragma: no cover - env dependent
        raise LLMAnalysisError("anthropic package not installed") from exc
    kwargs: dict = {"api_key": settings.ANTHROPIC_API_KEY, "timeout": settings.LLM_TIMEOUT_SECONDS}
    if settings.LLM_BASE_URL:
        # Anthropic Messages-compatible endpoints (proxies, alternative providers)
        kwargs["base_url"] = settings.LLM_BASE_URL
    return Anthropic(**kwargs)


def _backoff_delay(attempt: int) -> float:
    return min(2.0 ** attempt, 16.0)


def _call_with_retry(messages: list[dict], max_retries: int) -> str:
    """Call Claude, retrying on transient errors with exponential backoff."""
    api = client()
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            resp = api.messages.create(
                model=settings.LLM_MODEL,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=messages,
            )
            return "".join(
                block.text
                for block in resp.content
                if getattr(block, "text", None)
            )
        except Exception as exc:  # noqa: BLE001 - classify retryability
            last_err = exc
            name = type(exc).__name__
            retryable = "rate_limit" in name.lower() or "overloaded" in name.lower()
            if attempt < max_retries and retryable:
                delay = _backoff_delay(attempt)
                log.warning("LLM call failed (%s); retrying in %.1fs", name, delay)
                time.sleep(delay)
                continue
            raise LLMAnalysisError(f"LLM call failed after {attempt + 1} attempt(s): {exc}") from exc
    raise LLMAnalysisError(f"LLM call failed: {last_err}")


def _extract_json(text: str) -> dict | None:
    """Pull a JSON object out of a model response that may include prose."""
    text = text.strip()
    # strip ```json fences if present
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "sentence_scores" in obj:
            return obj
    except json.JSONDecodeError:
        pass
    # last resort: first {...} block
    start = text.find("{")
    end = text.rfind("}")
    if 0 <= start < end:
        try:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            return None
    return None


def analyze_chunk(sentences: list[dict], chunk_index: int = 0) -> ChunkAnalysis:
    """Analyze one list of {index, start, end, text} sentence dicts."""
    if not sentences:
        return ChunkAnalysis()

    transcript = "\n".join(
        f"[{s['index']}] ({s['start']:.1f}-{s['end']:.1f}) {s['text']}" for s in sentences
    )
    messages = [
        {
            "role": "user",
            "content": f"Analyze this transcript chunk (part {chunk_index + 1}):\n\n{transcript}",
        }
    ]

    try:
        raw = _call_with_retry(messages, settings.LLM_MAX_RETRIES)
    except LLMAnalysisError as exc:
        log.warning("LLM unavailable, using heuristics for chunk %d: %s", chunk_index, exc)
        return heuristic_analysis(sentences)

    obj = _extract_json(raw)
    if obj is None:
        log.warning("LLM returned unparseable JSON; retrying once with a stricter prompt")
        messages.append({"role": "assistant", "content": raw})
        messages.append(
            {
                "role": "user",
                "content": "That was not valid JSON matching the schema. Reply with ONLY the JSON object.",
            }
        )
        try:
            raw = _call_with_retry(messages, 1)
            obj = _extract_json(raw)
        except LLMAnalysisError:
            obj = None
    if obj is None:
        log.warning("LLM JSON still unparseable; using heuristics for chunk %d", chunk_index)
        return heuristic_analysis(sentences)

    try:
        return _validate(obj, len(sentences))
    except (KeyError, TypeError, ValueError) as exc:
        log.warning("malformed LLM output (%s); using heuristics", exc)
        return heuristic_analysis(sentences)


def _validate(obj: dict, n_sentences: int) -> ChunkAnalysis:
    """Coerce the LLM payload into ChunkAnalysis, clamping all scores."""
    out = ChunkAnalysis(
        overall_interest=_clamp(obj.get("overall_interest", 0.0)),
        summary=str(obj.get("summary", ""))[:600],
    )
    seen = set()
    for raw in obj.get("sentence_scores", []):
        try:
            idx = int(raw["sentence_index"])
        except (KeyError, TypeError, ValueError):
            continue
        if idx in seen:
            continue
        seen.add(idx)
        cat = raw.get("category")
        if isinstance(cat, str) and cat.lower() not in CATEGORIES:
            cat = None
        out.sentences.append(
            SentenceScore(
                sentence_index=idx,
                category=cat.lower() if isinstance(cat, str) else None,
                category_confidence=_clamp(raw.get("category_confidence", 0.0)),
                hook_strength=_clamp(raw.get("hook_strength", 0.0)),
                idea_completeness=_clamp(raw.get("idea_completeness", 0.0)),
                standalone_clarity=_clamp(raw.get("standalone_clarity", 0.0)),
                emotion=_clamp(raw.get("emotion", 0.0)),
                quoteability=_clamp(raw.get("quoteability", 0.0)),
                reason=str(raw.get("reason", ""))[:200],
                keywords=[str(k)[:40] for k in (raw.get("keywords") or [])[:12]],
                emojis=[e for e in (raw.get("emojis") or []) if _is_emoji(e)][:4],
            )
        )
    if not out.sentences:
        raise ValueError("no valid sentence_scores")
    return out


def _clamp(v, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        return max(lo, min(hi, float(v)))
    except (TypeError, ValueError):
        return 0.0


_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF]", re.UNICODE
)


def _is_emoji(s) -> bool:
    return isinstance(s, str) and bool(_EMOJI_RE.search(s)) and len(s) <= 8


# --------------------------------------------------------------------------
# Heuristic fallback — used when no LLM key is configured
# --------------------------------------------------------------------------
_HYPE_WORDS = {
    "en": ["never", "actually", "secret", "nobody", "always", "worst", "best", "crazy",
           "insane", "shocking", "truth", "nobody talks", "plot twist", "wait"],
    "id": ["tidak pernah", "sebenarnya", "rahasia", "ternyata", "paling", "gila",
           "shocking", "benar", "justru", "padahal"],
}
_QUESTION_WORDS = {"who", "what", "why", "how", "when", "where", "siapa", "apa", "kenapa", "bagaimana"}


def heuristic_analysis(sentences: list[dict]) -> ChunkAnalysis:
    """Rule-based scoring when the LLM is unavailable — keeps the pipeline moving."""
    scored: list[SentenceScore] = []
    for s in sentences:
        text = str(s["text"]).lower()
        words = set(re.findall(r"[a-z']+", text))
        lang = "id" if any(c.isalpha() for c in text) and "tidak" in text else "en"
        hype = sum(1 for w in _HYPE_WORDS.get(lang, []) if w in text)
        has_q = any(text.startswith(q) for q in _QUESTION_WORDS) or "?" in str(s["text"])
        length = len(str(s["text"]).split())
        digits = sum(1 for c in str(s["text"]) if c.isdigit())

        scored.append(
            SentenceScore(
                sentence_index=int(s["index"]),
                category="insight" if digits or hype else ("story" if length > 20 else None),
                category_confidence=0.35,
                hook_strength=min(0.4 + 0.2 * hype + (0.2 if has_q else 0.0), 1.0),
                idea_completeness=0.5 if length > 8 else 0.25,
                standalone_clarity=0.4 + (0.2 if hype else 0.0),
                emotion=min(0.3 + 0.15 * hype, 1.0),
                quoteability=min(0.3 + 0.1 * hype + (0.15 if digits else 0.0), 1.0),
                reason="heuristic: keyword/punctuation signals",
                keywords=sorted(w for w in words if len(w) > 5)[:5],
                emojis=[],
            )
        )
    return ChunkAnalysis(
        overall_interest=0.45,
        summary="(heuristic analysis — no LLM configured)",
        sentences=scored,
    )


# --------------------------------------------------------------------------
# Chunking + caching
# --------------------------------------------------------------------------
def chunk_sentences(
    sentences: list[dict],
    *,
    max_chunk_seconds: float | None = None,
    max_chunk_chars: int = 12000,
) -> list[list[dict]]:
    """Group sentences into LLM-sized chunks (roughly N minutes of speech)."""
    target = max_chunk_seconds or settings.LLM_CHUNK_SECONDS
    chunks: list[list[dict]] = []
    current: list[dict] = []
    current_chars = 0
    for s in sentences:
        text_len = len(s["text"])
        if current and (
            (s["end"] - current[0]["start"]) > target or current_chars + text_len > max_chunk_chars
        ):
            chunks.append(current)
            current = []
            current_chars = 0
        current.append(s)
        current_chars += text_len
    if current:
        chunks.append(current)
    return chunks


def cache_key(sentences: list[dict]) -> str:
    h = hashlib.sha256()
    for s in sentences:
        h.update(f"{s['index']}:{s['start']}:{s['end']}:{s['text']}".encode("utf-8"))
    h.update(settings.LLM_MODEL.encode())
    return h.hexdigest()[:24]


def load_cached(path: str | Path, key: str) -> ChunkAnalysis | None:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if data.get("key") != key:
        return None
    try:
        return ChunkAnalysis.from_dict(data["analysis"])
    except (KeyError, TypeError):
        return None


def save_cached(path: str | Path, key: str, analysis: ChunkAnalysis) -> None:
    try:
        Path(path).write_text(
            json.dumps({"key": key, "analysis": analysis.to_dict()}, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError:  # pragma: no cover
        log.warning("could not write LLM cache")
