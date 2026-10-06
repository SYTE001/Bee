"""Subtitle style presets.

Each preset is a JSON file in this directory. The loader validates every file
against `StyleSchema`; adding a new style is just dropping in a new JSON — see
`docs/SUBTITLE_STYLES.md`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path

from app.core.logging import get_logger

log = get_logger("styles")

# app/services/subtitle_styles.py -> backend/styles/
STYLES_DIR = Path(__file__).resolve().parent.parent.parent / "styles"

VALID_ANIMATIONS = {
    "none",
    "pop",
    "typewriter",
    "bounce",
    "glow",
    "slide",
    "shake",
    "rainbow",
}
VALID_POSITIONS = {"bottom", "center", "top"}


@dataclass
class StyleSchema:
    """One subtitle preset. Field names map 1:1 to the JSON file."""

    id: str
    name: str
    font: str = "Arial"
    font_size: int = 54
    primary_color: str = "#FFFFFF"
    secondary_color: str = "#FFE600"
    outline_color: str = "#000000"
    back_color: str = "#000000"
    outline: float = 3.0
    shadow: float = 1.5
    bold: bool = True
    italic: bool = False
    uppercase: bool = False
    alignment: int = 2  # ASS numpad alignment: 2 = bottom center
    margin_v: int = 210  # clear of TikTok's right-side buttons + caption area
    margin_l: int = 40
    margin_r: int = 40
    words_per_line: int = 3
    max_line_chars: int = 30
    animation: str = "none"
    animation_intensity: float = 1.0  # 0..2 multiplier for effect strength
    keyword_color: str = ""
    keyword_bold: bool = True
    emoji: bool = False
    position: str = "bottom"
    spacing: float = 1.0
    border_style: int = 1  # 1 = outline, 3 = opaque box behind text
    scale_x: float = 100.0
    scale_y: float = 100.0
    rotation: float = 0.0
    # optional lead-in/out fade in milliseconds
    fade_in_ms: int = 60
    fade_out_ms: int = 60
    tags: dict = field(default_factory=dict)  # style-specific extras

    def validate(self) -> list[str]:
        """Return a list of validation problems (empty == OK)."""
        problems: list[str] = []
        if not self.id or not self.name:
            problems.append("id and name are required")
        if self.animation not in VALID_ANIMATIONS:
            problems.append(f"unknown animation: {self.animation}")
        if self.position not in VALID_POSITIONS:
            problems.append(f"unknown position: {self.position}")
        if not (5 <= self.font_size <= 200):
            problems.append("font_size out of range")
        if not (0.0 <= self.animation_intensity <= 4.0):
            problems.append("animation_intensity must be in 0..4")
        if not (1 <= self.alignment <= 9):
            problems.append("alignment must be 1..9")
        for cname in ("primary_color", "secondary_color", "outline_color", "back_color"):
            value = getattr(self, cname)
            if value and not _is_hex_color(value):
                problems.append(f"{cname} is not a valid #RRGGBB color: {value}")
        if self.words_per_line < 1 or self.words_per_line > 8:
            problems.append("words_per_line must be 1..8")
        return problems


_VALID_FIELD_NAMES = {f.name for f in fields(StyleSchema)}


def _is_hex_color(value: str) -> bool:
    if not isinstance(value, str):
        return False
    v = value.strip().lstrip("#")
    return len(v) == 6 and all(c in "0123456789abcdefABCDEF" for c in v)


def load_style(path: str | Path) -> StyleSchema:
    """Load + validate one preset JSON, raising ValueError on bad content."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"preset {path} must be a JSON object")
    unknown = set(data) - _VALID_FIELD_NAMES
    if unknown:
        log.warning("preset %s has unknown keys: %s", path, ", ".join(sorted(unknown)))
    try:
        style = StyleSchema(**{k: v for k, v in data.items() if k in _VALID_FIELD_NAMES})
    except TypeError as exc:
        raise ValueError(f"preset {path}: {exc}") from exc
    problems = style.validate()
    if problems:
        raise ValueError(f"preset {path} invalid: {'; '.join(problems)}")
    return style


def load_all_styles(directory: str | Path | None = None) -> dict[str, StyleSchema]:
    """Load every preset in `directory`; skips (and logs) invalid files."""
    d = Path(directory or STYLES_DIR)
    styles: dict[str, StyleSchema] = {}
    if not d.is_dir():
        log.warning("styles directory missing: %s", d)
        return styles
    for file in sorted(d.glob("*.json")):
        try:
            style = load_style(file)
            styles[style.id] = style
        except (json.JSONDecodeError, ValueError, OSError) as exc:
            log.error("skipping invalid style preset %s: %s", file.name, exc)
    log.info("loaded %d subtitle styles from %s", len(styles), d)
    return styles


def get_style(style_id: str, directory: str | Path | None = None) -> StyleSchema | None:
    """Look up one preset by id (falls back to the classic default)."""
    styles = load_all_styles(directory)
    return styles.get(style_id) or styles.get("classic_bold_white")


def list_style_metadata(directory: str | Path | None = None) -> list[dict]:
    """Id + name for the UI dropdown."""
    return [{"id": s.id, "name": s.name} for s in load_all_styles(directory).values()]
