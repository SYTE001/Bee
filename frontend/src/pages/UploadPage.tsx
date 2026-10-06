import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";

import { api, ApiError } from "../api/client";
import type { LayoutMode, SourceType } from "../api/types";
import { Notice } from "../components/ui";

const CATEGORIES = [
  "funny",
  "serious",
  "insight",
  "emotional",
  "controversial",
  "story",
  "actionable",
];

const DURATION_PRESETS: { label: string; min: number; max: number }[] = [
  { label: "Short (15–30s)", min: 15, max: 30 },
  { label: "Medium (30–60s)", min: 30, max: 60 },
  { label: "Long (60–90s)", min: 60, max: 90 },
];

const ACCEPTED = ".mp4,.mov,.mkv,.avi,.webm,.m4v,.mpg,.mpeg,.wmv,.flv";

export function UploadPage() {
  const navigate = useNavigate();
  const inputRef = useRef<HTMLInputElement>(null);

  const [sourceType, setSourceType] = useState<SourceType>("upload");
  const [file, setFile] = useState<File | null>(null);
  const [url, setUrl] = useState("");
  const [language, setLanguage] = useState("auto");
  const [clipCount, setClipCount] = useState(10);
  const [durationPreset, setDurationPreset] = useState(0);
  const [categories, setCategories] = useState<string[]>([]);
  const [fps, setFps] = useState(60);
  const [layout, setLayout] = useState<LayoutMode>("auto");
  const [subtitleStyle, setSubtitleStyle] = useState("classic_bold_white");
  const [styles, setStyles] = useState<{ id: string; name: string }[]>(FALLBACK_STYLES);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [dragging, setDragging] = useState(false);

  // Load the live style list from the backend (falls back on network error).
  useEffect(() => {
    api
      .listStyles()
      .then((list) => {
        if (list.length) {
          setStyles(list.map((s) => ({ id: s.id, name: s.name })));
          if (!list.some((s) => s.id === subtitleStyle)) {
            setSubtitleStyle(list[0].id);
          }
        }
      })
      .catch(() => {
        /* keep the bundled fallback list */
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const preset = DURATION_PRESETS[durationPreset];

  const submit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      setError(null);
      if (sourceType === "upload" && !file) {
        setError("Choose a video file to upload.");
        return;
      }
      if (sourceType === "youtube" && !url.trim()) {
        setError("Paste a YouTube URL.");
        return;
      }
      setBusy(true);
      try {
        const name =
          sourceType === "upload" && file
            ? file.name
            : url.trim().split("/").pop() || "youtube-video";
        const project = await api.createProject({
          source_type: sourceType,
          source_name: name,
          source_url: sourceType === "youtube" ? url.trim() : undefined,
          language,
          target_dur_min: preset.min,
          target_dur_max: preset.max,
          target_clip_count: clipCount,
          desired_categories: categories,
          subtitle_style: subtitleStyle,
          target_fps: fps,
          layout_mode: layout,
        });

        if (sourceType === "upload" && file) {
          await api.uploadMedia(project.id, file, true);
        } else {
          await api.fetchYoutube(project.id);
        }
        navigate(`/projects/${project.id}`);
      } catch (err) {
        setError(
          err instanceof ApiError
            ? typeof err.detail === "string"
              ? err.detail
              : JSON.stringify(err.detail)
            : "request failed — is the backend running?",
        );
      } finally {
        setBusy(false);
      }
    },
    [file, url, sourceType, language, preset, clipCount, categories, fps, layout, subtitleStyle, navigate],
  );

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files?.[0];
    if (f) setFile(f);
  }, []);

  return (
    <div className="mx-auto max-w-3xl px-4 py-10">
      <h1 className="text-3xl font-bold">Create viral clips</h1>
      <p className="mt-2 text-forge-muted">
        Upload a podcast, interview, webinar, or vlog — ClipForge finds the best
        moments and turns them into vertical, subtitled clips.
      </p>

      <Notice kind="warn" >
        ⚠️ Copyright: only process content that you own or are licensed to
        redistribute. You are responsible for the videos you upload.
      </Notice>

      <form onSubmit={submit} className="card mt-6 flex flex-col gap-6">
        {/* source */}
        <div className="flex gap-2">
          {(["upload", "youtube"] as SourceType[]).map((t) => (
            <button
              key={t}
              type="button"
              onClick={() => setSourceType(t)}
              className={`btn flex-1 ${
                sourceType === t ? "btn-primary" : "btn-ghost"
              }`}
            >
              {t === "upload" ? "Upload file" : "YouTube URL"}
            </button>
          ))}
        </div>

        {sourceType === "upload" ? (
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            onClick={() => inputRef.current?.click()}
            className={`cursor-pointer rounded-xl border-2 border-dashed p-8 text-center transition-colors ${
              dragging
                ? "border-forge-accent bg-forge-accent/10"
                : "border-forge-border hover:border-forge-accent/60"
            }`}
          >
            <input
              ref={inputRef}
              type="file"
              accept={ACCEPTED}
              className="hidden"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            />
            {file ? (
              <div>
                <p className="font-semibold">{file.name}</p>
                <p className="text-sm text-forge-muted">
                  {(file.size / 1024 / 1024).toFixed(1)} MB · click to replace
                </p>
              </div>
            ) : (
              <div>
                <p className="font-semibold">Drop a video here or click to browse</p>
                <p className="text-sm text-forge-muted">
                  MP4, MOV, MKV, WebM — up to 4 GB
                </p>
              </div>
            )}
          </div>
        ) : (
          <label className="block">
            <span className="label">YouTube URL</span>
            <input
              className="input"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="https://www.youtube.com/watch?v=…"
              type="url"
            />
            <span className="mt-1 block text-xs text-forge-muted">
              Requires yt-dlp on the backend server.
            </span>
          </label>
        )}

        {/* options grid */}
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          <label className="block">
            <span className="label">Language</span>
            <select className="input" value={language} onChange={(e) => setLanguage(e.target.value)}>
              <option value="auto">Auto-detect</option>
              <option value="en">English</option>
              <option value="id">Bahasa Indonesia</option>
              <option value="es">Español</option>
              <option value="de">Deutsch</option>
              <option value="fr">Français</option>
              <option value="ja">日本語</option>
              <option value="ko">한국어</option>
              <option value="zh">中文</option>
            </select>
          </label>

          <label className="block">
            <span className="label">Clip length</span>
            <select
              className="input"
              value={durationPreset}
              onChange={(e) => setDurationPreset(Number(e.target.value))}
            >
              {DURATION_PRESETS.map((p, i) => (
                <option key={p.label} value={i}>
                  {p.label}
                </option>
              ))}
            </select>
          </label>

          <label className="block">
            <span className="label">Number of clips: {clipCount}</span>
            <input
              type="range"
              min={3}
              max={25}
              value={clipCount}
              onChange={(e) => setClipCount(Number(e.target.value))}
              className="mt-3 w-full accent-forge-accent"
            />
          </label>

          <label className="block">
            <span className="label">Output FPS</span>
            <select className="input" value={fps} onChange={(e) => setFps(Number(e.target.value))}>
              <option value={30}>30 fps (native)</option>
              <option value={60}>60 fps (interpolated)</option>
              <option value={90}>90 fps (interpolated)</option>
            </select>
            {fps > 30 && (
              <span className="mt-1 block text-xs text-yellow-300/80">
                Source footage is usually 30 fps — higher rates are interpolated.
              </span>
            )}
          </label>

          <label className="block">
            <span className="label">Reframe layout</span>
            <select className="input" value={layout} onChange={(e) => setLayout(e.target.value as LayoutMode)}>
              <option value="auto">Auto (recommended)</option>
              <option value="single_focus">Single focus — crop to speaker</option>
              <option value="split_screen">Split screen — 2 speakers</option>
              <option value="fit_blur">Fit + blurred background</option>
              <option value="fill_zoom">Fill + dynamic zoom</option>
            </select>
          </label>

          <label className="block">
            <span className="label">Subtitle style</span>
            <select className="input" value={subtitleStyle} onChange={(e) => setSubtitleStyle(e.target.value)}>
              {styles.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.name}
                </option>
              ))}
            </select>
          </label>
        </div>

        {/* categories */}
        <div>
          <span className="label">Moment categories (optional)</span>
          <div className="flex flex-wrap gap-2">
            {CATEGORIES.map((c) => {
              const active = categories.includes(c);
              return (
                <button
                  key={c}
                  type="button"
                  onClick={() =>
                    setCategories((prev) =>
                      prev.includes(c) ? prev.filter((x) => x !== c) : [...prev, c],
                    )
                  }
                  className={`badge cursor-pointer border ${
                    active
                      ? "border-forge-accent bg-forge-accent/30 text-violet-200"
                      : "border-forge-border text-forge-muted hover:border-forge-accent/50"
                  }`}
                >
                  {c}
                </button>
              );
            })}
          </div>
        </div>

        {error && <Notice kind="warn">{error}</Notice>}

        <button type="submit" className="btn-primary w-full py-3" disabled={busy}>
          {busy ? "Processing…" : "Generate clips"}
        </button>
      </form>
    </div>
  );
}

// Used only if the style list cannot be fetched from the backend.
const FALLBACK_STYLES: { id: string; name: string }[] = [
  { id: "classic_bold_white", name: "Classic Bold White" },
  { id: "hormozi", name: "Hormozi" },
  { id: "beast", name: "Beast" },
  { id: "karaoke_glow", name: "Karaoke Glow" },
  { id: "minimal_clean", name: "Minimal Clean" },
  { id: "boxed_highlight", name: "Boxed Highlight" },
  { id: "typewriter", name: "Typewriter" },
  { id: "bounce_pop", name: "Bounce Pop" },
  { id: "gradient_fire", name: "Gradient Fire" },
  { id: "comic_impact", name: "Comic Impact" },
  { id: "neon_cyberpunk", name: "Neon Cyberpunk" },
  { id: "retro_yellow", name: "Retro Yellow" },
  { id: "handwritten", name: "Handwritten" },
  { id: "podcast_lower_third", name: "Podcast Lower-Third" },
  { id: "emoji_auto", name: "Emoji Auto" },
];
