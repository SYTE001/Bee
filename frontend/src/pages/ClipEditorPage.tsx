import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";

import { api, ApiError, authUrl } from "../api/client";
import type {
  Clip,
  ClipWord,
  LayoutMode,
  Project,
  SubtitleStyle,
} from "../api/types";
import { Notice, ScoreRing, formatDuration } from "../components/ui";
import { SubtitlePreview } from "../components/SubtitlePreview";

const FPS_OPTIONS = [30, 60, 90];

const LAYOUTS: { value: LayoutMode; label: string }[] = [
  { value: "auto", label: "Auto (recommended)" },
  { value: "single_focus", label: "Single focus — crop to speaker" },
  { value: "split_screen", label: "Split screen — 2 speakers" },
  { value: "fit_blur", label: "Fit + blurred background" },
  { value: "fill_zoom", label: "Fill + dynamic zoom" },
];

/**
 * Clip editor: trim, subtitle style with live preview, layout, fps and manual
 * transcript edits. Saving marks the clip for re-render; the render itself is
 * queued asynchronously.
 */
export function ClipEditorPage() {
  const { projectId, clipId } = useParams<{ projectId: string; clipId: string }>();

  const [project, setProject] = useState<Project | null>(null);
  const [clip, setClip] = useState<Clip | null>(null);
  const [styles, setStyles] = useState<SubtitleStyle[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // form state
  const [title, setTitle] = useState("");
  const [start, setStart] = useState(0);
  const [end, setEnd] = useState(1);
  const [transcript, setTranscript] = useState("");
  const [styleId, setStyleId] = useState("classic_bold_white");
  const [layout, setLayout] = useState<LayoutMode>("auto");
  const [fps, setFps] = useState(30);

  const [saving, setSaving] = useState(false);
  const [saveMsg, setSaveMsg] = useState<string | null>(null);
  const [rendering, setRendering] = useState(false);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      if (!projectId || !clipId) return;
      try {
        const [p, s] = await Promise.all([
          api.getProject(projectId),
          api.listStyles(),
        ]);
        if (cancelled) return;
        setProject(p);
        setStyles(s);
        const c = p.clips?.find((x) => x.id === clipId);
        if (!c) {
          setError("clip not found");
          return;
        }
        setClip(c);
        setTitle(c.title);
        setStart(c.start);
        setEnd(c.end);
        setTranscript(c.transcript);
        setStyleId(c.subtitle_style);
        setLayout((c.layout_mode as LayoutMode) ?? "auto");
        setFps(c.target_fps);
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "failed to load clip");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    void load();
    return () => {
      cancelled = true;
    };
  }, [projectId, clipId]);

  const activeStyle = useMemo(
    () => styles.find((s) => s.id === styleId),
    [styles, styleId],
  );

  // words for the live preview (regenerated from the edited transcript)
  const previewWords = useMemo(() => {
    const raw = transcript.trim();
    return raw ? raw.split(/\s+/) : ["edit", "the", "transcript"];
  }, [transcript]);

  const dirty = useMemo(() => {
    if (!clip) return false;
    return (
      title !== clip.title ||
      start !== clip.start ||
      end !== clip.end ||
      transcript.trim() !== clip.transcript.trim() ||
      styleId !== clip.subtitle_style ||
      layout !== clip.layout_mode ||
      fps !== clip.target_fps
    );
  }, [clip, title, start, end, transcript, styleId, layout, fps]);

  /**
   * Rebuild the word list after a transcript edit: keep timings for words whose
   * index survived, and spread any new/removed words evenly across the clip.
   */
  const rebuildWords = useCallback(
    (text: string): ClipWord[] => {
      const tokens = text.trim().split(/\s+/).filter(Boolean);
      const oldWords = clip?.words ?? [];
      const span = Math.max(end - start, 0.1);
      const per = span / Math.max(tokens.length, 1);
      return tokens.map((word, i) => {
        const old = oldWords[i];
        if (old && old.word === word) {
          return { ...old, start: clamp(old.start, start, end), end: clamp(old.end, start, end) };
        }
        return { word, start: start + i * per, end: start + (i + 1) * per };
      });
    },
    [clip, start, end],
  );

  const save = async () => {
    if (!projectId || !clipId || !clip) return;
    if (end <= start) {
      setSaveMsg("End must be after start.");
      return;
    }
    setSaving(true);
    setSaveMsg(null);
    try {
      const textChanged = transcript.trim() !== (clip.transcript?.trim() ?? "");
      const updated = await api.updateClip(projectId, clipId, {
        title,
        start,
        end,
        subtitle_style: styleId,
        layout_mode: layout,
        target_fps: fps,
        ...(textChanged ? { words: rebuildWords(transcript) } : {}),
      });
      setClip(updated);
      setSaveMsg(
        "Saved. The output changed, so the previous render was invalidated.",
      );
    } catch (err) {
      setSaveMsg(
        err instanceof ApiError ? String(err.detail) : "failed to save clip",
      );
    } finally {
      setSaving(false);
    }
  };

  const render = async () => {
    if (!projectId || !clipId) return;
    setRendering(true);
    setError(null);
    try {
      await api.renderClip(projectId, clipId, true);
      // poll until the render lands
      pollRef.current = setInterval(async () => {
        const p = await api.getProject(projectId);
        const fresh = p.clips?.find((c) => c.id === clipId);
        if (!fresh) return;
        setClip(fresh);
        if (fresh.rendered || fresh.render_error) {
          stopPolling();
          setRendering(false);
        }
      }, 2500);
    } catch (err) {
      setRendering(false);
      setError(err instanceof ApiError ? String(err.detail) : "failed to queue render");
    }
  };

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);
  useEffect(() => stopPolling, [stopPolling]);

  if (loading) {
    return (
      <div className="mx-auto max-w-4xl px-4 py-10">
        <p className="text-forge-muted">Loading clip…</p>
      </div>
    );
  }

  if (error || !clip || !project) {
    return (
      <div className="mx-auto max-w-4xl px-4 py-10">
        <Notice kind="warn">{error ?? "clip not found"}</Notice>
        <Link to={`/projects/${projectId}`} className="btn-primary mt-4">
          Back to project
        </Link>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-5xl px-4 py-10">
      <div className="flex items-center gap-3 text-sm text-forge-muted">
        <Link to={`/projects/${projectId}`} className="hover:text-violet-300">
          ← {project.source_name}
        </Link>
        <span>/</span>
        <span>Clip #{clip.rank}</span>
      </div>

      <div className="mt-4 grid gap-6 lg:grid-cols-[320px_1fr]">
        {/* ---------------------------------------------------------- preview */}
        <div className="space-y-4">
          {activeStyle ? (
            <SubtitlePreview
              style={activeStyle}
              words={previewWords}
              width={288}
              className="mx-auto ring-1 ring-forge-border"
            />
          ) : (
            <div className="card">Loading styles…</div>
          )}
          <p className="text-center text-xs text-forge-muted">
            Live preview of the <strong>{activeStyle?.name}</strong> style — the
            burn-in uses the same font, colours, outline and word grouping.
          </p>

          {clip.rendered && clip.video_url ? (
            <div className="card !p-3">
              <video
                src={authUrl(clip.video_url)}
                controls
                playsInline
                className="mx-auto max-h-[420px] rounded-lg bg-black"
              />
              <p className="mt-2 text-center text-xs text-forge-muted">
                Current render ·{" "}
                <a
                  href={authUrl(clip.video_url)}
                  download
                  className="text-violet-300 underline"
                >
                  download mp4
                </a>
              </p>
            </div>
          ) : (
            <div className="card text-center text-sm text-forge-muted">
              {clip.render_error ? (
                <span className="text-red-400">Last render failed: {clip.render_error}</span>
              ) : (
                "Not rendered yet"
              )}
            </div>
          )}
        </div>

        {/* ----------------------------------------------------------- editor */}
        <form
          className="card space-y-5"
          onSubmit={(e) => {
            e.preventDefault();
            void save();
          }}
        >
          <div className="flex items-start justify-between gap-3">
            <h1 className="text-xl font-bold">Edit clip</h1>
            <ScoreRing score={clip.viral_score} size={56} />
          </div>

          <label className="block">
            <span className="label">Title</span>
            <input
              className="input"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              maxLength={120}
            />
          </label>

          <div>
            <span className="label">Trim</span>
            <TrimBar
              start={start}
              end={end}
              duration={project.duration ?? end}
              onStart={setStart}
              onEnd={setEnd}
            />
            <div className="mt-2 flex items-center justify-between text-sm text-forge-muted">
              <span>
                {formatDuration(start)} – {formatDuration(end)}
              </span>
              <span className={end - start > 90 ? "text-yellow-300" : ""}>
                {(end - start).toFixed(1)}s
              </span>
            </div>
          </div>

          <div>
            <label className="block">
              <span className="label">Transcript</span>
              <textarea
                className="input min-h-[110px] font-mono text-sm"
                value={transcript}
                onChange={(e) => setTranscript(e.target.value)}
              />
            </label>
            <span className="mt-1 block text-xs text-forge-muted">
              Edits re-split the words and spread new timings evenly across the
              trimmed range.
            </span>
          </div>

          <div className="grid gap-4 sm:grid-cols-2">
            <label className="block">
              <span className="label">Subtitle style</span>
              <select
                className="input"
                value={styleId}
                onChange={(e) => setStyleId(e.target.value)}
              >
                {styles.map((s) => (
                  <option key={s.id} value={s.id}>
                    {s.name}
                  </option>
                ))}
              </select>
            </label>

            <label className="block">
              <span className="label">Reframe layout</span>
              <select
                className="input"
                value={layout}
                onChange={(e) => setLayout(e.target.value as LayoutMode)}
              >
                {LAYOUTS.map((l) => (
                  <option key={l.value} value={l.value}>
                    {l.label}
                  </option>
                ))}
              </select>
            </label>
          </div>

          <label className="block">
            <span className="label">Output FPS</span>
            <select
              className="input"
              value={fps}
              onChange={(e) => setFps(Number(e.target.value))}
            >
              {FPS_OPTIONS.map((f) => (
                <option key={f} value={f}>
                  {f} fps{f > 30 ? " (interpolated)" : " (native)"}
                </option>
              ))}
            </select>
          </label>

          {saveMsg && <Notice kind={saveMsg.startsWith("Saved") ? "info" : "warn"}>{saveMsg}</Notice>}
          {error && <Notice kind="warn">{error}</Notice>}

          <div className="flex flex-wrap gap-3">
            <button
              type="submit"
              className="btn-primary px-5 py-2.5 disabled:opacity-40"
              disabled={saving || !dirty}
            >
              {saving ? "Saving…" : dirty ? "Save changes" : "All changes saved"}
            </button>
            <button
              type="button"
              className="rounded-lg border border-forge-border px-5 py-2.5 text-sm text-forge-muted transition-colors hover:border-forge-accent/50 hover:text-violet-200 disabled:opacity-40"
              disabled={rendering || dirty}
              title={dirty ? "Save before rendering" : undefined}
              onClick={() => void render()}
            >
              {rendering ? "Rendering…" : clip.rendered ? "Re-render" : "Render clip"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Trim bar — two handles over the source timeline
// ---------------------------------------------------------------------------
function TrimBar({
  start,
  end,
  duration,
  onStart,
  onEnd,
}: {
  start: number;
  end: number;
  duration: number;
  onStart: (v: number) => void;
  onEnd: (v: number) => void;
}) {
  const max = Math.max(duration, end);
  const pct = (v: number) => `${(v / max) * 100}%`;
  const step = 0.1;

  return (
    <div>
      <div className="relative h-10 rounded-lg bg-forge-panel2">
        <div
          className="absolute inset-y-0 rounded-lg bg-forge-accent/30 ring-1 ring-forge-accent"
          style={{ left: pct(start), width: `${((end - start) / max) * 100}%` }}
        />
        <div
          className="absolute inset-y-0 w-2 -translate-x-1/2 cursor-ew-resize rounded-full bg-violet-300"
          style={{ left: pct(start) }}
        />
        <div
          className="absolute inset-y-0 w-2 -translate-x-1/2 cursor-ew-resize rounded-full bg-violet-300"
          style={{ left: pct(end) }}
        />
      </div>
      <div className="mt-3 grid grid-cols-2 gap-3">
        <label className="flex items-center gap-2 text-sm">
          <span className="text-forge-muted">start</span>
          <input
            type="number"
            className="input"
            min={0}
            max={end - step}
            step={step}
            value={start}
            onChange={(e) => onStart(Math.min(Number(e.target.value), end - step))}
          />
        </label>
        <label className="flex items-center gap-2 text-sm">
          <span className="text-forge-muted">end</span>
          <input
            type="number"
            className="input"
            min={start + step}
            max={max}
            step={step}
            value={end}
            onChange={(e) => onEnd(Math.max(Number(e.target.value), start + step))}
          />
        </label>
      </div>
    </div>
  );
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}
