"""Tests for the ASS subtitle generator and style presets."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services import subtitle
from app.services.subtitle import (
    AssBuilder,
    SubtitleWord,
    ass_time,
    build_ass,
    censor_word,
    escape_text,
    hex_to_ass,
)
from app.services.subtitle_styles import (
    STYLES_DIR,
    load_all_styles,
    load_style,
)

SAMPLE_WORDS = [
    {"word": "Nobody", "start": 0.0, "end": 0.35},
    {"word": "talks", "start": 0.35, "end": 0.7},
    {"word": "about", "start": 0.7, "end": 1.0},
    {"word": "this", "start": 1.0, "end": 1.25},
    {"word": "secret.", "start": 1.25, "end": 1.8},
    {"word": "It", "start": 2.0, "end": 2.15},
    {"word": "is", "start": 2.15, "end": 2.3},
    {"word": "crazy.", "start": 2.3, "end": 2.9},
]


class TestColorConversion:
    def test_rgb_to_bgr(self):
        assert hex_to_ass("#FF0000") == "&H000000FF"
        assert hex_to_ass("#00FF00") == "&H0000FF00"
        assert hex_to_ass("#0000FF") == "&H00FF0000"
        assert hex_to_ass("#FFFFFF") == "&H00FFFFFF"

    def test_invalid_falls_back(self):
        assert hex_to_ass("") == "&H00FFFFFF"
        assert hex_to_ass("nope") == "&H00FFFFFF"
        assert hex_to_ass("#1234567") == "&H00FFFFFF"


class TestAssTime:
    def test_formats(self):
        assert ass_time(0.0) == "0:00:00.00"
        assert ass_time(1.5) == "0:00:01.50"
        assert ass_time(65.25) == "0:01:05.25"
        assert ass_time(3661.0) == "1:01:01.00"

    def test_negative_clamped(self):
        assert ass_time(-5.0) == "0:00:00.00"


class TestEscape:
    def test_escapes_special(self):
        assert escape_text("a{b}c") == "a\\{b\\}c"
        assert escape_text("back\\slash") == "back\\\\slash"


class TestCensor:
    def test_masks_known_words(self):
        assert censor_word("fuck") == "f**k"
        assert censor_word("shit!") == "s**t!"
        assert censor_word("SHIT") == "S**T"

    def test_leaves_clean_words(self):
        assert censor_word("hello") == "hello"

    def test_single_letter_untouched(self):
        assert censor_word("a") == "a"


class TestAssBuilder:
    def test_minimal_output_structure(self, tmp_path):
        out = build_ass(SAMPLE_WORDS, "classic_bold_white", tmp_path / "out.ass")
        text = out.read_text(encoding="utf-8")
        assert "[Script Info]" in text
        assert "[V4+ Styles]" in text
        assert "[Events]" in text
        assert "Dialogue:" in text

    def test_karaoke_timing_per_word(self, tmp_path):
        out = build_ass(SAMPLE_WORDS, "classic_bold_white", tmp_path / "out.ass")
        text = out.read_text(encoding="utf-8")
        # word 0 is 0.35s = 35 centiseconds
        assert "\\k35" in text
        # word 5 "It" is 0.15s -> round(0.15 * 100) == 15, int() == 14
        assert "\\k14" in text or "\\k15" in text

    def test_line_timings_match_words(self, tmp_path):
        words = [SubtitleWord(text="hello", start=1.0, end=1.5)]
        builder = AssBuilder(load_style(STYLES_DIR / "classic_bold_white.json"))
        builder.add_words(words)
        text = builder.build()
        assert "0:00:01.00" in text
        # single word is held to at least 0.35s
        assert "0:00:01.50" in text

    def test_words_per_line_grouping(self, tmp_path):
        out = build_ass(SAMPLE_WORDS, "classic_bold_white", tmp_path / "out.ass")
        text = out.read_text(encoding="utf-8")
        dialogue = [l for l in text.splitlines() if l.startswith("Dialogue")]
        # 8 words at 3 per line => 3 lines (3+3+2)
        assert len(dialogue) == 3

    def test_uppercase_style_applied(self, tmp_path):
        out = build_ass(SAMPLE_WORDS, "hormozi", tmp_path / "out.ass")
        text = out.read_text(encoding="utf-8")
        assert "NOBODY" in text

    def test_keyword_highlight(self, tmp_path):
        out = build_ass(
            SAMPLE_WORDS,
            "hormozi",
            tmp_path / "out.ass",
            keywords=["secret"],
        )
        text = out.read_text(encoding="utf-8")
        # keyword color = #FFE600 -> BGR &H0000E6FF
        assert "&H0000E6FF" in text

    def test_censor_mode(self, tmp_path):
        bad = [{"word": "fuck", "start": 0.0, "end": 0.4}, {"word": "world", "start": 0.4, "end": 0.8}]
        out = build_ass(bad, "classic_bold_white", tmp_path / "out.ass", censor=True)
        text = out.read_text(encoding="utf-8")
        assert "f**k" in text
        assert "fuck" not in text

    def test_emoji_style_emits_emoji_layer(self, tmp_path):
        out = build_ass(
            SAMPLE_WORDS,
            "emoji_auto",
            tmp_path / "out.ass",
            emojis=["🔥", "💡"],
        )
        text = out.read_text(encoding="utf-8")
        assert "Style: Emoji" in text
        assert "🔥" in text

    def test_empty_words_raises(self, tmp_path):
        with pytest.raises(ValueError):
            build_ass([], "classic_bold_white", tmp_path / "out.ass")

    def test_unknown_style_falls_back(self, tmp_path):
        out = build_ass(SAMPLE_WORDS, "does_not_exist", tmp_path / "out.ass")
        assert out.exists()

    def test_position_override(self, tmp_path):
        out = build_ass(
            SAMPLE_WORDS, "classic_bold_white", tmp_path / "out.ass", position="center"
        )
        text = out.read_text(encoding="utf-8")
        assert "\\an5" not in text  # alignment expressed in the Style line, not per-line

    def test_canvas_resolution(self, tmp_path):
        out = build_ass(
            SAMPLE_WORDS,
            "classic_bold_white",
            tmp_path / "out.ass",
            canvas_width=720,
            canvas_height=1280,
        )
        text = out.read_text(encoding="utf-8")
        assert "PlayResX: 720" in text
        assert "PlayResY: 1280" in text

    def test_animation_tags_present(self, tmp_path):
        # pop style should emit scale + transform tags
        out = build_ass(SAMPLE_WORDS, "hormozi", tmp_path / "out.ass")
        text = out.read_text(encoding="utf-8")
        assert "\\fscx" in text
        assert "\\t(" in text

    def test_fonts_scaled(self, tmp_path):
        out = build_ass(
            SAMPLE_WORDS, "classic_bold_white", tmp_path / "out.ass", font_scale=1.5
        )
        text = out.read_text(encoding="utf-8")
        assert "Style: Main,Arial Black,78" in text

    def test_main_style_has_all_23_fields(self, tmp_path):
        """A short style row shifts every later field and breaks rendering.

        Regression: ScaleX/ScaleY were once omitted, so ScaleY picked up the
        rotation value and libass drew zero-height (invisible) text.
        """
        out = build_ass(SAMPLE_WORDS, "classic_bold_white", tmp_path / "out.ass")
        style_line = next(
            line
            for line in out.read_text(encoding="utf-8").splitlines()
            if line.startswith("Style: Main,")
        )
        fields = style_line[len("Style: ") :].split(",")
        assert len(fields) == 23, f"expected 23 style fields, got {len(fields)}"
        # ScaleX/ScaleY must stay at the schema default, not 1.0/0.0.
        assert fields[11] == "100.0" and fields[12] == "100.0"

    def test_every_preset_style_row_is_well_formed(self, tmp_path):
        for style_id in (sid for sid in load_all_styles()):
            out = build_ass(SAMPLE_WORDS, style_id, tmp_path / f"{style_id}.ass")
            for line in out.read_text(encoding="utf-8").splitlines():
                if line.startswith("Style: "):
                    assert len(line[len("Style: ") :].split(",")) == 23, (style_id, line)

    def test_rendered_text_is_visible(self, tmp_path):
        """Burn the ASS with real ffmpeg and confirm pixels are drawn.

        Field-count bugs produce a syntactically valid file that ffmpeg
        accepts while drawing nothing, so the only reliable check is pixels.
        """
        import subprocess

        try:
            import numpy as np  # type: ignore[import-untyped]
        except ImportError:  # pragma: no cover
            pytest.skip("numpy not installed")

        ass = build_ass(
            [{"word": "hello", "start": 0.0, "end": 2.0}],
            "classic_bold_white",
            tmp_path / "vis.ass",
        )
        esc = str(ass).replace("\\", "/").replace(":", "\\:")
        video = tmp_path / "vis.mp4"
        subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                "-i", "color=c=black:s=1080x1920:d=2:r=30",
                "-vf", f"subtitles='{esc}'",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video),
            ],
            check=True,
            capture_output=True,
        )
        frame = tmp_path / "vis.gray"
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-ss", "1", "-i", str(video),
             "-frames:v", "1", "-pix_fmt", "gray", "-f", "rawvideo", str(frame)],
            check=True,
            capture_output=True,
        )
        arr = np.frombuffer(frame.read_bytes(), dtype=np.uint8).reshape(1920, 1080)
        assert arr.max() > 120, "subtitles rendered no visible text"


class TestStylePresets:
    def test_all_fifteen_load(self):
        styles = load_all_styles()
        assert len(styles) >= 15
        expected = {
            "classic_bold_white", "hormozi", "beast", "karaoke_glow", "minimal_clean",
            "boxed_highlight", "typewriter", "bounce_pop", "gradient_fire",
            "comic_impact", "neon_cyberpunk", "retro_yellow", "handwritten",
            "podcast_lower_third", "emoji_auto",
        }
        assert expected.issubset(set(styles))

    def test_every_preset_is_valid(self):
        for file in sorted(STYLES_DIR.glob("*.json")):
            style = load_style(file)
            problems = style.validate()
            assert problems == [], f"{file.name}: {problems}"

    def test_ids_match_filenames(self):
        for file in STYLES_DIR.glob("*.json"):
            data = json.loads(file.read_text(encoding="utf-8"))
            assert data["id"] == file.stem

    def test_invalid_preset_rejected(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"id": "bad", "name": "Bad", "animation": "nonsense"}))
        with pytest.raises(ValueError):
            load_style(bad)

    def test_malformed_json_skipped(self, tmp_path):
        broken = tmp_path / "broken.json"
        broken.write_text("{not json")
        styles = load_all_styles(tmp_path)
        assert "broken" not in styles
