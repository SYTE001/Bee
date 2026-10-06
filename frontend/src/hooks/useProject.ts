import { useCallback, useEffect, useRef, useState } from "react";

import { api } from "../api/client";
import type { PipelineEvent, Project } from "../api/types";

interface State {
  project: Project | null;
  events: PipelineEvent[];
  connected: boolean;
  error: string | null;
}

/**
 * Polls a project and subscribes to its SSE stream so the UI shows pipeline
 * progress in real time. Falls back to polling alone when SSE is unavailable.
 */
export function useProject(projectId: string | undefined) {
  const [state, setState] = useState<State>({
    project: null,
    events: [],
    connected: false,
    error: null,
  });
  const unsubscribe = useRef<(() => void) | null>(null);
  const [refreshTick, setRefreshTick] = useState(0);

  const refresh = useCallback(() => {
    setRefreshTick((t) => t + 1);
  }, []);

  const fetchProject = useCallback(async () => {
    if (!projectId) return;
    try {
      const p = await api.getProject(projectId);
      setState((s) => ({ ...s, project: p, error: null }));
      return p;
    } catch (err) {
      setState((s) => ({
        ...s,
        error: err instanceof Error ? err.message : "failed to load project",
      }));
      return null;
    }
  }, [projectId]);

  // Re-fetch on demand (a child action finished a render, saved an edit, …).
  useEffect(() => {
    if (projectId) void fetchProject();
  }, [projectId, refreshTick, fetchProject]);

  useEffect(() => {
    if (!projectId) return;
    // SSE needs the same origin — in dev the vite proxy covers /api
    unsubscribe.current = api.subscribeEvents(projectId, (evt) => {
      setState((s) => {
        const events = [...s.events, evt].slice(-200);
        let project = s.project;
        if (evt.type === "stage" && project) {
          const data = evt.data as {
            status?: string;
            message?: string;
            progress?: number;
          };
          project = {
            ...project,
            status: (data.status as Project["status"]) ?? project.status,
            stage_message: data.message ?? project.stage_message,
            stage_progress: data.progress ?? project.stage_progress,
          };
        }
        return { ...s, events, project, connected: true };
      });
    });

    const es = unsubscribe.current;
    return () => {
      es?.();
      unsubscribe.current = null;
    };
  }, [projectId, fetchProject]);

  // Poll as a safety net (SSE can silently drop); stop when finished.
  useEffect(() => {
    if (!projectId) return;
    const id = setInterval(async () => {
      const p = await fetchProject();
      if (p && (p.status === "completed" || p.status === "failed")) {
        clearInterval(id);
      }
    }, 3000);
    return () => clearInterval(id);
  }, [projectId, fetchProject]);

  return { ...state, refresh };
}
