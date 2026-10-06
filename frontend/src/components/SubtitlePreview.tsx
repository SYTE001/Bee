import { useMemo } from "react";

import type { SubtitleStyle } from "../api/types";

interface Props {
  style: SubtitleStyle;
  /** Words to display; defaults to a sample hook line. */
  words?: string[];
  /** Words that should render in the style's keyword colour. */
  keywords?: string[];
  /** Index of the currently spoken word (rendered in the secondary colour). */
  activeIndex?: number;
  /** Preview canvas width in px. The real render is 1080px wide. */
  width?: number;
  className?: string;
}

const DEFAULT_WORDS = ["the", "secret", "is", "consistency", "over", "talent"];
const DEFAULT_KEYWORDS = ["secret", "consistency"];

/**
 * Browser-side stand-in for the ASS burn-in: renders the style's font, colours,
 * outline, shadow, position and word grouping the way libass will, so the clip
 * editor can show what the subtitle looks like before a (slow) re-render.
 *
 * The geometry is scaled from the 1080x1920 output canvas to the preview width,
 * so margin_v / font_size map 1:1 to their real pixel values.
 */
export function SubtitlePreview({
  style,
  words = DEFAULT_WORDS,
  keywords = DEFAULT_KEYWORDS,
  activeIndex = 1,
  width = 234,
  className,
}: Props) {
  const scale = width / 1080;

  // mirror the subtitle engine: `words_per_line` words per Dialogue event
  const lines = useMemo(() => {
    const per = Math.max(1, style.words_per_line);
    const out: string[][] = [];
    for (let i = 0; i < words.length && out.length < 2; i += per) {
      out.push(words.slice(i, i + per));
    }
    return out;
  }, [words, style.words_per_line]);

  const alignClass =
    style.alignment % 3 === 1
      ? "text-left"
      : style.alignment % 3 === 0
        ? "text-right"
        : "text-center";

  const verticalClass =
    style.position === "top"
      ? "items-start"
      : style.position === "center"
        ? "items-center"
        : "items-end";

  const pad =
    style.position === "top"
      ? { paddingTop: style.margin_v * scale }
      : style.position === "center"
        ? {}
        : { paddingBottom: style.margin_v * scale };

  const wordStyle = (active: boolean, isKeyword: boolean): React.CSSProperties => {
    const color = isKeyword && style.keyword_color
      ? style.keyword_color
      : active
        ? style.secondary_color
        : style.primary_color;
    return {
      color,
      fontFamily: `"${style.font}", "Arial Black", system-ui, sans-serif`,
      fontSize: style.font_size * scale,
      fontWeight: (isKeyword && style.keyword_bold) || style.bold ? 800 : 400,
      fontStyle: style.italic ? "italic" : "normal",
      textTransform: style.uppercase ? "uppercase" : "none",
      WebkitTextStrokeWidth: `${Math.max(style.outline, 0) * scale}px`,
      WebkitTextStrokeColor: style.outline_color,
      // ASS drop shadow: offset + blur in the style's back colour
      textShadow: style.shadow
        ? `0 ${style.shadow * 2 * scale}px ${style.shadow * 4 * scale}px ${style.back_color}`
        : undefined,
      letterSpacing: style.spacing ? `${style.spacing * scale}px` : undefined,
      display: "inline-block",
    };
  };

  return (
    <div
      className={`relative overflow-hidden rounded-xl bg-gradient-to-br from-forge-panel2 to-black ${verticalClass} flex flex-col ${alignClass} ${className ?? ""}`}
      style={{ width, aspectRatio: "9 / 16", ...pad }}
    >
      {/* stand-in for the video track */}
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(60%_40%_at_30%_25%,rgba(139,92,246,0.18),transparent)]" />

      {/* TikTok safe-zone hint: the right rail where buttons sit */}
      <div className="pointer-events-none absolute right-1 top-1/4 flex w-[9%] flex-col gap-2 rounded-full bg-white/5 p-1">
        <div className="aspect-square rounded-full bg-white/15" />
        <div className="aspect-square rounded-full bg-white/15" />
        <div className="aspect-square rounded-full bg-white/15" />
      </div>

      <div className="relative z-10 w-full px-2" style={{ paddingLeft: style.margin_l * scale, paddingRight: style.margin_r * scale }}>
        {lines.map((line, li) => (
          <p key={li} className="leading-tight">
            {line.map((word, wi) => {
              const gi = li * Math.max(1, style.words_per_line) + wi;
              const isKeyword = keywords.includes(word.toLowerCase());
              return (
                <span key={wi} style={wordStyle(gi === activeIndex, isKeyword)}>
                  {style.emoji && isKeyword ? `${word} 🔥` : word}
                  {wi < line.length - 1 ? " " : ""}
                </span>
              );
            })}
          </p>
        ))}
      </div>

      {style.animation !== "none" && (
        <span className="absolute bottom-1.5 left-2 rounded bg-black/60 px-1.5 py-0.5 text-[9px] uppercase tracking-wide text-white/70">
          {style.animation} anim
        </span>
      )}
    </div>
  );
}
