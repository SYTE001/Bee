import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { api, ApiError } from "../api/client";
import type { Project } from "../api/types";
import { Notice, StatusBadge, formatDuration } from "../components/ui";

export function ProjectsPage() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const res = await api.listProjects();
      setProjects(res.items);
      setError(null);
    } catch (err) {
      setError(
        err instanceof ApiError ? String(err.detail ?? "request failed") : "network error",
      );
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  // Keep statuses fresh while any project is still processing.
  useEffect(() => {
    const active = projects.some(
      (p) => p.status !== "completed" && p.status !== "failed",
    );
    if (!active) return;
    const id = setInterval(() => void load(), 4000);
    return () => clearInterval(id);
  }, [projects, load]);

  const onDelete = useCallback(
    async (id: string) => {
      if (!confirm("Delete this project and all its clips?")) return;
      try {
        await api.deleteProject(id);
        setProjects((prev) => prev.filter((p) => p.id !== id));
      } catch (err) {
        setError(err instanceof ApiError ? String(err.detail) : "delete failed");
      }
    },
    [],
  );

  if (loading) {
    return (
      <div className="mx-auto max-w-5xl px-4 py-10">
        <p className="text-forge-muted">Loading projects…</p>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-5xl px-4 py-10">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">Your projects</h1>
        <Link to="/upload" className="btn-primary">
          + New project
        </Link>
      </div>

      {error && (
        <div className="mt-4">
          <Notice kind="warn">{error}</Notice>
        </div>
      )}

      {projects.length === 0 ? (
        <div className="card mt-8 text-center">
          <p className="text-forge-muted">
            No projects yet.{" "}
            <Link to="/upload" className="text-violet-400 hover:underline">
              Upload your first video
            </Link>{" "}
            to get started.
          </p>
        </div>
      ) : (
        <div className="mt-6 grid grid-cols-1 gap-4 md:grid-cols-2">
          {projects.map((p) => (
            <div key={p.id} className="card flex flex-col gap-3">
              <div className="flex items-start justify-between gap-3">
                <Link
                  to={`/projects/${p.id}`}
                  className="font-semibold hover:text-violet-300"
                >
                  {p.source_name}
                </Link>
                <StatusBadge status={p.status} />
              </div>

              <div className="flex flex-wrap gap-3 text-sm text-forge-muted">
                <span>{p.source_type === "youtube" ? "YouTube" : "Upload"}</span>
                {p.duration ? <span>{formatDuration(p.duration)}</span> : null}
                {p.width ? (
                  <span>
                    {p.width}×{p.height}
                  </span>
                ) : null}
                {p.detected_language ? <span>lang: {p.detected_language}</span> : null}
                {p.clips?.length ? <span>{p.clips.length} clips</span> : null}
              </div>

              {p.status === "failed" && p.error && (
                <p className="line-clamp-2 text-sm text-red-300">{p.error}</p>
              )}
              {p.status !== "failed" && p.status !== "completed" && p.stage_message && (
                <p className="line-clamp-1 text-sm text-violet-300">{p.stage_message}</p>
              )}

              <div className="mt-auto flex justify-end gap-2 pt-2">
                <Link to={`/projects/${p.id}`} className="btn-ghost !py-1.5">
                  Open
                </Link>
                <button
                  onClick={() => onDelete(p.id)}
                  className="btn-ghost !py-1.5 text-red-300 hover:border-red-500"
                >
                  Delete
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
