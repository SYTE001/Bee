import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { api, ApiError, authUrl } from "../api/client";
import type { Clip, Project } from "../api/types";
import { CategoryBadge, ScoreRing, formatDuration } from "./ui";

interface Props {
  project: Project;
  /** Ask the parent to re-fetch the project after a render completes. */
  onRefresh?: () => void;
}

/**
 * Grid of ranked clips with per-clip render/download/edit actions and a batch
 * toolbar (render all, download ZIP). Rendering is asynchronous — once a clip
 * is queued we poll until it finishes or errors.
 */
export function ClipGrid({ project, onRefresh }: Props) {
  const clips: Clip[] = project.clips ?? [];

  if (clips.length === 0) {
    if (project.status === "completed" || project.status === "failed") {
      return (
        <div className="card text-center">
          <p className="text-forge-muted">No clips were found for this project.</p>
          <p className="mt-2 text-sm text-forge-muted">
            Moment detection and ranking did not produce any clip inside the
            requested duration band — try a longer video or a different preset.
          </p>
          {project.status === "failed" && (
            <Link to="/upload" className="btn-primary mt-4">
              Try another video
            </Link>
          )}
        </div>
      );
    }
    return (
      <div className="card text-center text-forge-muted">
        Clips appear here once the pipeline finishes analyzing your video.
      </div>
    );
  }

  return (
    <div>
      <BatchToolbar project={project} clips={clips} onRefresh={onRefresh} />
      <div className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {clips.map((clip) => (
          <ClipCard key={clip.id} projectId={project.id} clip={clip} onRefresh={onRefresh} />
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Batch toolbar
// ---------------------------------------------------------------------------
function BatchToolbar({
  project,
  clips,
  onRefresh,
}: {
  project: Project;
  clips: Clip[];
  onRefresh?: () => void;
}) {
  const rendered = clips.filter((c) => c.rendered).length;
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);

  const renderAll = async () => {
    setBusy(true);
    setMsg(null);
    try {
      await api.renderAllClips(project.id, false);
      setMsg("Queued all clips for rendering — this takes a while.");
      onRefresh?.();
    } catch (err) {
      setMsg(err instanceof ApiError ? String(err.detail) : "failed to queue rendering");
    } finally {
      setBusy(false);
    }
  };

  const downloadZip = () => {
    if (rendered === 0) {
      setMsg("Render at least one clip first.");
      return;
    }
    api.downloadZip(project.id);
  };

  return (
    <div className="card flex flex-wrap items-center gap-3 !py-3">
      <span className="text-sm font-semibold">
        {rendered} / {clips.length} rendered
      </span>
      <div className="ml-auto flex flex-wrap gap-2">
        <button
          className="btn-primary !px-3 !py-1.5 text-sm disabled:opacity-40"
          disabled={busy || rendered === clips.length}
          onClick={() => void renderAll()}
        >
          {busy
            ? "Queuing…"
            : rendered === clips.length
              ? "All rendered"
              : rendered === 0
                ? "Render all"
                : `Render remaining ${clips.length - rendered}`}
        </button>
        <button
          className="btn-primary !px-3 !py-1.5 text-sm disabled:opacity-40"
          disabled={rendered === 0}
          onClick={downloadZip}
        >
          Download ZIP
        </button>
      </div>
      {msg && <p className="basis-full text-xs text-forge-muted">{msg}</p>}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Clip card
// ---------------------------------------------------------------------------
function ClipCard({
  projectId,
  clip,
  onRefresh,
}: {
  projectId: string;
  clip: Clip;
  onRefresh?: () => void;
}) {
  const [rendering, setRendering] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // While this clip is queued for rendering, poll until it settles.
  useEffect(() => {
    if (!rendering) return;
    const id = setInterval(async () => {
      try {
        const p = await api.getProject(projectId);
        const fresh = p.clips?.find((c) => c.id === clip.id);
        if (!fresh) {
          clearInterval(id);
          setRendering(false);
          return;
        }
        if (fresh.rendered || fresh.render_error) {
          clearInterval(id);
          setRendering(false);
          onRefresh?.();
        }
      } catch {
        /* keep polling; the parent's own error handling covers persistent failures */
      }
    }, 2500);
    return () => clearInterval(id);
  }, [rendering, projectId, clip.id, onRefresh]);

  const render = async () => {
    setError(null);
    setRendering(true);
    try {
      await api.renderClip(projectId, clip.id, !clip.rendered);
    } catch (err) {
      setRendering(false);
      setError(err instanceof ApiError ? String(err.detail) : "failed to queue render");
    }
  };

  return (
    <div className="card flex flex-col gap-3 transition-transform hover:scale-[1.02]">
      <div className="flex items-start justify-between gap-3">
        <span className="text-sm font-semibold text-forge-muted">
          #{clip.rank} · {formatDuration(clip.start)}–{formatDuration(clip.end)}
        </span>
        <ScoreRing score={clip.viral_score} />
      </div>

      <div className="-mx-4 -mt-2 aspect-video overflow-hidden rounded-t-xl bg-black">
        {clip.thumb_url ? (
          <img
            src={authUrl(clip.thumb_url)}
            alt={clip.title}
            className="h-full w-full object-cover"
            loading="lazy"
          />
        ) : (
          <div className="grid h-full place-items-center text-xs text-white/30">no preview yet</div>
        )}
      </div>

      <h3 className="line-clamp-2 font-bold">{clip.title || "Untitled clip"}</h3>

      <div className="flex flex-wrap gap-2">
        <CategoryBadge category={clip.category_primary} />
        {clip.category_secondary && clip.category_secondary !== clip.category_primary && (
          <CategoryBadge category={clip.category_secondary} />
        )}
      </div>

      {clip.reasons && clip.reasons.length > 0 && (
        <ul className="space-y-1 text-xs text-forge-muted">
          {clip.reasons.slice(0, 2).map((r, i) => (
            <li key={i}>• {r}</li>
          ))}
        </ul>
      )}

      <p className="line-clamp-2 text-sm text-forge-muted">{clip.transcript}</p>

      {error && <p className="text-xs text-red-400">{error}</p>}
      {clip.render_error && !error && (
        <p className="text-xs text-red-400">Render failed: {clip.render_error}</p>
      )}

      <div className="mt-auto flex items-center justify-between gap-2 pt-2">
        <span className="text-xs text-forge-muted">
          {clip.duration.toFixed(1)}s · {clip.target_fps}fps
        </span>
        <div className="flex gap-2">
          <Link
            to={`/projects/${projectId}/clips/${clip.id}`}
            className="rounded-lg border border-forge-border px-3 py-1.5 text-xs text-forge-muted transition-colors hover:border-forge-accent/50 hover:text-violet-200"
          >
            Edit
          </Link>
          {clip.rendered && clip.video_url ? (
            <a
              href={authUrl(clip.video_url)}
              download
              className="btn-primary !px-3 !py-1.5 text-xs"
            >
              Download
            </a>
          ) : (
            <button
              className="btn-primary !px-3 !py-1.5 text-xs disabled:opacity-40"
              disabled={rendering}
              onClick={() => void render()}
            >
              {rendering ? "Rendering…" : clip.render_error ? "Re-render" : "Render"}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
