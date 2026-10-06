"""ASS (Advanced SubStation Alpha) subtitle generator.

Produces libass-compatible .ass files with per-word karaoke timing, active-word
highlighting, keyword coloring, contextual emoji, and per-style animations.

Design notes:
  - Each rendered "line" holds `words_per_line` words and is one Dialogue event.
  - Active-word highlighting uses ASS karaoke (`\\k`): the style's secondary
    color is the "not yet spoken" color and the primary color is the "spoken"
    color, so the currently spoken word is always visually distinct.
  - Per-word animations (pop/bounce/typewriter/glow/...) are emitted as
    override blocks inside the line so libass runs them with exact timing.
  - Everything is escaped: words are plain text, colors are converted from
    #RRGGBB to ASS's BGR hex.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from app.core.logging import get_logger
from app.services.subtitle_styles import StyleSchema, load_all_styles

log = get_logger("subtitle")

# Words too short to stand alone on their own line
_MIN_WORD_LEN = 1
# Profanity list (extendable); masked when censoring is enabled
_PROFANITY = {
    "fuck", "shit", "bitch", "asshole", "dick", "cunt", "piss", "bastard", "slut", "whore",
}
_PROFANITY_ID = {
    "bangsat", "kontol", "memek", "anjing", "goblok", "tolol", "bajingan", "kampret",
}


@dataclass
class SubtitleWord:
    """A word as used by the subtitle engine."""

    text: str
    start: float
    end: float
    speaker: str | None = None
    keyword: bool = False
    emoji: str | None = None


def hex_to_ass(color: str, default: str = "&H00FFFFFF") -> str:
    """Convert '#RRGGBB' to ASS BGR format '&H00BBGGRR'."""
    if not color:
        return default
    v = color.strip().lstrip("#")
    if len(v) != 6 or not all(c in "0123456789abcdefABCDEF" for c in v):
        return default
    r, g, b = v[0:2], v[2:4], v[4:6]
    return f"&H00{b}{g}{r}".upper()


def ass_time(seconds: float) -> str:
    """Format seconds as ASS timestamp `H:MM:SS.cc`."""
    if seconds < 0:
        seconds = 0.0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours}:{minutes:02d}:{secs:05.2f}"


def escape_text(text: str) -> str:
    """Escape characters that have special meaning in ASS."""
    return (
        text.replace("\\", "\\\\")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("\n", "\\N")
    )


_WORD_SPLIT_RE = re.compile(r"^(\W*)([\w'-]+)(\W*)$", re.UNICODE)


def censor_word(word: str) -> str:
    """Replace a profanity with asterisks, keeping first/last letter + punctuation."""
    match = _WORD_SPLIT_RE.match(word)
    if not match:
        return word
    prefix, core, suffix = match.groups()
    if core.lower() in _PROFANITY or core.lower() in _PROFANITY_ID:
        if len(core) <= 1:
            return word
        return f"{prefix}{core[0]}{'*' * (len(core) - 2)}{core[-1]}{suffix}"
    return word


class AssBuilder:
    """Builds an .ass file from timed words."""

    def __init__(
        self,
        style: StyleSchema,
        *,
        canvas_width: int = 1080,
        canvas_height: int = 1920,
        censor: bool = False,
        keywords: list[str] | None = None,
        emojis: dict[str, str] | None = None,
        scale: float = 1.0,
    ) -> None:
        self.style = style
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.censor = censor
        self.scale = scale  # user font-size multiplier
        self.keyword_set = {k.lower() for k in (keywords or [])}
        self.emoji_map = emojis or {}
        self.lines: list[str] = []  # completed Dialogue lines
        self.emoji_lines: list[str] = []

    # ------------------------------------------------------------------ build
    def add_words(self, words: list[SubtitleWord]) -> None:
        """Append a run of timed words to the subtitle stream."""
        if not words:
            return
        for group in self._group_words(words):
            self._emit_line(group)

    def _group_words(self, words: list[SubtitleWord]) -> list[list[SubtitleWord]]:
        """Split words into renderable lines honoring words_per_line + width."""
        style = self.style
        groups: list[list[SubtitleWord]] = []
        current: list[SubtitleWord] = []
        current_chars = 0
        for w in words:
            text = self._display_text(w)
            if current and (
                len(current) >= style.words_per_line
                or current_chars + len(text) > style.max_line_chars
            ):
                groups.append(current)
                current = []
                current_chars = 0
            current.append(w)
            current_chars += len(text) + 1
        if current:
            groups.append(current)
        return groups

    def _display_text(self, w: SubtitleWord) -> str:
        text = w.text
        if self.censor:
            text = censor_word(text)
        if self.style.uppercase:
            text = text.upper()
        return text

    def _emit_line(self, group: list[SubtitleWord]) -> None:
        style = self.style
        start = group[0].start
        # hold the last word a beat so single-word lines are readable
        end = max(group[-1].end, start + 0.35)

        parts: list[str] = []
        for w in group:
            parts.append(self._word_block(w, line_start=start))
        body = "".join(parts)
        if not body:
            return

        fade = self._fade_block(start, end)
        line = (
            f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Main,,0,0,0,,"
            f"{fade}{body}"
        )
        self.lines.append(line)

        if style.emoji and any(w.emoji for w in group):
            emoji_text = "".join(w.emoji for w in group if w.emoji)
            self.emoji_lines.append(
                f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Emoji,,0,0,0,,"
                f"{fade}{{\\an{style.alignment}}}{{\\fad({style.fade_in_ms},{style.fade_out_ms})}}"
                f"{escape_text(emoji_text)}"
            )

    def _word_block(self, w: SubtitleWord, line_start: float) -> str:
        """Build the override + text block for a single word."""
        style = self.style
        text = escape_text(self._display_text(w))
        if not text:
            return ""

        duration_cs = max(int((w.end - w.start) * 100), 1)
        tags: list[str] = [f"\\k{duration_cs}"]

        # keyword coloring
        stripped = w.text.strip(".,!?;:\"'()[]").lower()
        if style.keyword_color and stripped in self.keyword_set:
            tags.append(f"\\1c{hex_to_ass(style.keyword_color)}")
            if style.keyword_bold:
                tags.append("\\b1")

        # per-style animation
        tags.extend(self._animation_tags(w, line_start))

        return "{" + "".join(tags) + "}" + text + " "

    def _animation_tags(self, w: SubtitleWord, line_start: float) -> list[str]:
        style = self.style
        intensity = style.animation_intensity
        anim = style.animation

        if anim == "pop":
            peak = int(100 + 45 * intensity)
            half = max(int(80 * intensity), 40)
            return [
                f"\\fscx{peak}\\fscy{peak}",
                f"\\t(0,{half},\\fscx100\\fscy100)",
            ]
        if anim == "bounce":
            up = int(100 + 40 * intensity)
            return [
                "\\fscy70",
                f"\\t(0,{max(int(50 * intensity), 20)},\\fscy{up})",
                f"\\t({max(int(50 * intensity), 20)},{max(int(110 * intensity), 50)},\\fscy100)",
            ]
        if anim == "typewriter":
            # reveal this word only once its time slot arrives
            return ["\\alpha&HFF&", f"\\t({max(int((w.start - line_start) * 1000), 0)},"
                    f"{max(int((w.start - line_start) * 1000), 0) + 40},\\alpha&H00&)"]
        if anim == "glow":
            blur = min(int(6 * intensity), 12)
            return [
                f"\\blur{max(blur // 2, 1)}",
                f"\\t(0,{max(int(120 * intensity), 60)},\\blur{blur})",
                f"\\t({max(int(120 * intensity), 60)},{max(int(240 * intensity), 120)},\\blur{max(blur // 2, 1)})",
            ]
        if anim == "shake":
            amp = int(6 * intensity)
            return [
                f"\\frx{amp}\\fry{-amp // 2}",
                f"\\t(0,{max(int(90 * intensity), 40)},\\frx0\\fry0)",
            ]
        if anim == "slide":
            dist = int(60 * intensity)
            return [f"\\move(-{dist},0,0,0,{max(int(150 * intensity), 70)},0)"]
        if anim == "rainbow":
            # cycle hue-ish colors across the word's duration
            return [f"\\t(0,{max(duration_of(w), 30)},\\1c{hex_to_ass(style.secondary_color)})"]
        return []

    def _fade_block(self, start: float, end: float) -> str:
        style = self.style
        if style.fade_in_ms <= 0 and style.fade_out_ms <= 0:
            return ""
        return f"{{\\fad({style.fade_in_ms},{style.fade_out_ms})}}"

    # ------------------------------------------------------------------ output
    def build(self) -> str:
        """Render the complete .ass document."""
        style = self.style
        font_size = int(style.font_size * self.scale)
        sections = [
            self._script_info(),
            self._styles(font_size),
            self._events(),
        ]
        return "\n".join(sections) + "\n"

    def _script_info(self) -> str:
        return (
            "[Script Info]\n"
            "; Generated by ClipForge\n"
            "ScriptType: v4.00+\n"
            f"PlayResX: {self.canvas_width}\n"
            f"PlayResY: {self.canvas_height}\n"
            "WrapStyle: 2\n"
            "ScaledBorderAndShadow: yes\n"
            "YCbCr Matrix: TV.709\n"
        )

    def _styles(self, font_size: int) -> str:
        s = self.style
        align = s.alignment
        if s.position == "center":
            align = 5
        elif s.position == "top":
            align = 8
        margin_v = s.margin_v
        if s.position == "center":
            margin_v = 0
        elif s.position == "top":
            margin_v = max(s.margin_v // 2, 60)

        header = (
            "[V4+ Styles]\n"
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
            "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        )
        common = (
            f"0,0,{s.scale_x},{s.scale_y},{s.spacing},{s.rotation},{s.border_style},"
            f"{s.outline},{s.shadow},{align},{s.margin_l},{s.margin_r},{margin_v},1"
        )
        main = (
            f"Style: Main,{s.font},{font_size},"
            f"{hex_to_ass(s.primary_color, '&H00FFFFFF')},"
            f"{hex_to_ass(s.secondary_color, '&H0000FFD6')},"
            f"{hex_to_ass(s.outline_color, '&H00000000')},"
            f"{hex_to_ass(s.back_color, '&H00000000')},"
            f"{-1 if s.bold else 0},{-1 if s.italic else 0},{common}\n"
        )
        # Emoji style: same geometry, no outline, sits above the main text
        emoji = (
            f"Style: Emoji,{s.font},{int(font_size * 0.8)},"
            "&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,"
            f"0,0,0,0,100,100,0,0,1,2,0,{align},{s.margin_l},{s.margin_r},{max(margin_v + int(font_size * 1.6), margin_v)},1\n"
        )
        return header + main + emoji

    def _events(self) -> str:
        header = (
            "[Events]\n"
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, Effect, Text\n"
        )
        return header + "\n".join(self.lines + self.emoji_lines)

    # ------------------------------------------------------------------ write
    def write(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.build(), encoding="utf-8")
        log.info("wrote ASS: %s (%d line(s))", out.name, len(self.lines))
        return out


def duration_of(w: SubtitleWord) -> int:
    """Word duration in centiseconds."""
    return max(int((w.end - w.start) * 100), 1)


# --------------------------------------------------------------------------
# High level API
# --------------------------------------------------------------------------
def build_ass(
    words: list[dict],
    style_id: str,
    out_path: str | Path,
    *,
    canvas_width: int = 1080,
    canvas_height: int = 1920,
    censor: bool = False,
    keywords: list[str] | None = None,
    emojis: list[str] | None = None,
    font_scale: float = 1.0,
    words_per_line: int | None = None,
    position: str | None = None,
) -> Path:
    """Build an .ass file from word dicts (as stored on a Clip).

    `words` items: {word, start, end, speaker?, confidence?}
    """
    style = _resolve_style(style_id)
    if words_per_line is not None:
        style = _override(style, words_per_line=words_per_line)
    if position is not None and position in ("bottom", "center", "top"):
        style = _override(style, position=position)

    sub_words = [
        SubtitleWord(
            text=str(w.get("word", "")),
            start=float(w.get("start", 0.0)),
            end=float(w.get("end", 0.0)),
            speaker=w.get("speaker"),
        )
        for w in words
        if str(w.get("word", "")).strip()
    ]
    if not sub_words:
        raise ValueError("no words to render subtitles from")

    # keyword + emoji enrichment
    kw = {k.lower() for k in (keywords or [])}
    emoji_pool = list(emojis or [])
    for i, w in enumerate(sub_words):
        core = w.text.strip(".,!?;:\"'()[]").lower()
        w.keyword = core in kw
        if style.emoji and emoji_pool:
            w.emoji = emoji_pool[i % len(emoji_pool)]

    builder = AssBuilder(
        style,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        censor=censor,
        keywords=keywords,
        scale=font_scale,
    )
    builder.add_words(sub_words)
    return builder.write(out_path)


def _resolve_style(style_id: str) -> StyleSchema:
    """Load a preset, falling back to the classic default."""
    styles = load_all_styles()
    if style_id in styles:
        return styles[style_id]
    if "classic_bold_white" in styles:
        log.warning("style '%s' not found; using classic_bold_white", style_id)
        return styles["classic_bold_white"]
    return StyleSchema(id="fallback", name="Fallback")


def _override(style: StyleSchema, **changes) -> StyleSchema:
    """Return a copy of `style` with fields overridden."""
    data = style.__dict__.copy()
    data.update(changes)
    return StyleSchema(**data)


def list_styles() -> list[dict]:
    """Id + name for the frontend dropdown."""
    return [{"id": s.id, "name": s.name} for s in load_all_styles().values()]
