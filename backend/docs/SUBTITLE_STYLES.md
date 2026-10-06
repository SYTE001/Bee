# Subtitle styles

Every subtitle preset is a JSON file in [`../styles/`](../styles). The pipeline
picks one per clip (user-chosen or defaulted), the clip editor can switch it,
and the live preview in the browser renders the same font, colours, outline
and word grouping the burn-in will produce.

**Adding a style = dropping one JSON file into `backend/styles/`. No code
changes, no restart.** The loader validates it on first use and logs any
problems.

## Minimal preset

```json
{
  "id": "my_style",
  "name": "My Style",
  "font": "Arial Black",
  "font_size": 54,
  "primary_color": "#FFFFFF",
  "secondary_color": "#FFE600",
  "outline_color": "#000000",
  "back_color": "#000000",
  "outline": 3,
  "shadow": 1.5,
  "bold": true,
  "words_per_line": 3,
  "animation": "pop",
  "position": "bottom"
}
```

Every field has a default, so only `id` and `name` are strictly required.

## Field reference

| Field | Type | Range / values | Meaning |
|---|---|---|---|
| `id` | string | unique, snake_case | Stable identifier stored on each clip |
| `name` | string | — | Shown in the upload page and clip editor |
| `font` | string | a font family | Used for both the ASS burn-in and the browser preview. Install the font in the container/system if it isn't a web-safe family |
| `font_size` | int | 5–200 | ASS points on the 1080×1920 canvas |
| `primary_color` | `#RRGGBB` | hex | Spoken-word colour |
| `secondary_color` | `#RRGGBB` | hex | Not-yet-spoken colour (karaoke highlight) |
| `outline_color` | `#RRGGBB` | hex | Stroke colour |
| `back_color` | `#RRGGBB` | hex | Shadow / box colour |
| `outline` | float | ≥ 0 | Stroke width in points |
| `shadow` | float | ≥ 0 | Drop-shadow blur; 0 disables |
| `bold` / `italic` | bool | — | Font weight / style |
| `uppercase` | bool | — | Transform the text |
| `alignment` | int | 1–9 | ASS numpad layout: 1 bottom-left … 2 bottom-centre … 9 top-right |
| `position` | string | `bottom` \| `center` \| `top` | Vertical placement (also drives the preview) |
| `margin_v` / `margin_l` / `margin_r` | int | ≥ 0 | Pixels from the canvas edge. `margin_v: 210` clears TikTok's right-side buttons and caption area |
| `words_per_line` | int | 1–8 | Words per subtitle line (each line = one Dialogue event) |
| `max_line_chars` | int | ≥ 8 | Soft cap; longer lines wrap |
| `animation` | string | `none` `pop` `typewriter` `bounce` `glow` `slide` `shake` `rainbow` | Per-word effect |
| `animation_intensity` | float | 0–4 | Effect strength multiplier |
| `keyword_color` | `#RRGGBB` | hex or empty | Highlight colour for detected keywords; empty disables |
| `keyword_bold` | bool | — | Bold keywords |
| `emoji` | bool | — | Append a contextual emoji to keywords |
| `border_style` | int | 1 \| 3 | 1 = outline, 3 = opaque box behind the text |
| `scale_x` / `scale_y` | float | > 0 | ASS text scaling (percent) |
| `spacing` | float | ≥ 0 | Letter spacing in points |
| `rotation` | float | degrees | Line rotation |
| `fade_in_ms` / `fade_out_ms` | int | ≥ 0 | Lead-in/out fade; 0 disables |
| `tags` | object | — | Free-form extras, currently unused by the built-in renderer |

## Notes

- **Karaoke highlighting.** Each line is emitted with ASS `\k` timing: the
  current word renders in `primary_color`, the rest in `secondary_color`. Pick
  a pair with enough contrast that both read against the video.
- **Two colours must differ.** If `primary_color == secondary_color` the
  active word is indistinguishable.
- **Validation.** `StyleSchema.validate()` checks ranges, colours, animation
  and position names. Unknown JSON keys are logged as a warning (not an error)
  so older presets keep loading.
- **Fonts.** The burn-in needs the font installed for libass/ffmpeg; the
  Docker image ships DejaVu and Liberation. `font: "Arial Black"` and
  `Impact` are common presets — on Linux those resolve through fontconfig
  substitution, so preview the result on the target platform.
