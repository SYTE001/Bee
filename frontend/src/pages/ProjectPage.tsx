import { useParams } from "react-router-dom";

import { api } from "../api/client";
import { useProject } from "../hooks/useProject";
import { Notice, StatusBadge, formatDuration } from "../components/ui";
import { PIPELINE_STAGES } from "../api/types";
import { ClipGrid } from "../components/ClipGrid";

export function ProjectPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const { project, events, connected, error, refresh } = useProject(projectId);

  if (error) {
    return (
      <div className="mx-auto max-w-4xl px-4 py-10">
        <Notice kind="warn">{error}</Notice>
      </div>
    );
  }

  if (!project) {
    return (
      <div className="mx-auto max-w-4xl px-4 py-10">
        <p className="text-forge-muted">Loading project…</p>
      </div>
    );
  }

  const failed = project.status === "failed";
  const done = project.status === "completed";

  const activeStageIdx = PIPELINE_STAGES.findIndex((s) => s.key === project.status);
  // How far the tracker should show as "reached"
  const reachedIdx = done
    ? PIPELINE_STAGES.length
    : activeStageIdx >= 0
      ? activeStageIdx
      : PIPELINE_STAGES.length - 1; // unknown status: assume near the end

  return (
    <div className="mx-auto max-w-5xl px-4 py-10">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold">{project.source_name}</h1>
          <div className="mt-1 flex flex-wrap gap-3 text-sm text-forge-muted">
            {project.duration ? <span>{formatDuration(project.duration)}</span> : null}
            {project.width ? (
              <span>
                {project.width}×{project.height}@{project.fps?.toFixed(0)}fps
              </span>
            ) : null}
            {project.detected_language ? (
              <span>language: {project.detected_language}</span>
            ) : null}
            {project.num_speakers ? <span>{project.num_speakers} speakers</span> : null}
          </div>
        </div>
        <StatusBadge status={project.status} />
      </div>

      {failed && project.error && (
        <div className="mt-4">
          <Notice kind="warn">
            <strong>Pipeline failed:</strong> {project.error}
            <div className="mt-2">
              Completed stages are cached — re-running resumes from the failure
              point.{" "}
              <button
                className="text-violet-300 underline"
                onClick={() => void api.startProject(project.id)}
              >
                Retry
              </button>
            </div>
          </Notice>
        </div>
      )}

      {/* stage tracker */}
      <div className="card mt-6">
        <div className="mb-3 flex items-center justify-between text-sm">
          <span className="font-semibold">Pipeline progress</span>
          <span className="text-forge-muted">
            {connected ? "live" : "polling"} · {Math.round(project.stage_progress * 100)}%
          </span>
        </div>
        <div className="mb-4 h-2 w-full overflow-hidden rounded-full bg-forge-panel2">
          <div
            className={`h-full rounded-full transition-all duration-500 ${
              failed
                ? "bg-red-500"
                : "bg-gradient-to-r from-forge-accent to-forge-accent2"
            }`}
            style={{ width: `${Math.max(3, project.stage_progress * 100)}%` }}
          />
        </div>
        <ol className="grid grid-cols-2 gap-2 sm:grid-cols-3">
          {PIPELINE_STAGES.map((stage, idx) => {
            const reached = idx < reachedIdx || done;
            const current = idx === activeStageIdx && !done && !failed;
            return (
              <li
                key={stage.key}
                className={`flex items-center gap-2 rounded-lg px-3 py-2 text-sm ${
                  current
                    ? "bg-forge-accent/20 text-violet-200"
                    : reached
                      ? "text-emerald-300"
                      : "text-forge-muted"
                }`}
              >
                <span className="grid h-5 w-5 place-items-center rounded-full border border-current text-xs">
                  {reached ? "✓" : current ? "•" : idx + 1}
                </span>
                {stage.label}
              </li>
            );
          })}
        </ol>
        {project.stage_message && (
          <p className="mt-3 text-sm text-forge-muted">{project.stage_message}</p>
        )}
      </div>

      {/* live event log */}
      {events.length > 0 && (
        <details className="card mt-4">
          <summary className="cursor-pointer text-sm font-semibold">
            Event log ({events.length})
          </summary>
          <div className="mt-3 max-h-48 space-y-1 overflow-auto font-mono text-xs text-forge-muted">
            {events
              .slice()
              .reverse()
              .map((e, i) => (
                <div key={i}>
                  <span className="text-forge-accent">[{e.type}]</span>{" "}
                  {typeof e.data?.message === "string"
                    ? e.data.message
                    : JSON.stringify(e.data)}
                </div>
              ))}
          </div>
        </details>
      )}

      {/* clips */}
      <div className="mt-8">
        <h2 className="mb-4 text-xl font-bold">Clips</h2>
        <ClipGrid project={project} onRefresh={refresh} />
      </div>
    </div>
  );
}
