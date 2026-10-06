// Thin fetch wrapper with JWT handling and typed helpers.

import type {
  Clip,
  ClipUpdate,
  PipelineEvent,
  Project,
  ProjectCreate,
  SubtitleStyle,
  TokenResponse,
  User,
} from "./types";

const TOKEN_KEY = "clipforge_token";

export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}

export function setToken(token: string | null): void {
  if (token) localStorage.setItem(TOKEN_KEY, token);
  else localStorage.removeItem(TOKEN_KEY);
}

/**
 * Append the JWT to a URL that the browser loads without our headers —
 * `<video>`, `<img>`, and `window.location` downloads. Mirrors the SSE route,
 * which accepts `?token=` for the same reason.
 */
export function authUrl(path: string): string {
  const token = getToken();
  if (!token) return path;
  const sep = path.includes("?") ? "&" : "?";
  return `${path}${sep}token=${encodeURIComponent(token)}`;
}

export class ApiError extends Error {
  status: number;
  detail: unknown;
  constructor(status: number, detail: unknown) {
    super(typeof detail === "string" ? detail : `HTTP ${status}`);
    this.status = status;
    this.detail = detail;
  }
}

async function request<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const headers = new Headers(options.headers || {});
  const token = getToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);

  if (options.body instanceof FormData) {
    // let the browser set the multipart boundary
  } else if (options.body) {
    headers.set("Content-Type", "application/json");
  }

  const res = await fetch(path, { ...options, headers });
  if (!res.ok) {
    let detail: unknown = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? body;
    } catch {
      /* non-JSON error */
    }
    if (res.status === 401) setToken(null);
    throw new ApiError(res.status, detail);
  }
  if (res.status === 204) return undefined as T;
  const text = await res.text();
  return (text ? JSON.parse(text) : undefined) as T;
}

export const api = {
  // ---------------------------------------------------------------- auth
  register: (email: string, password: string, display_name?: string) =>
    request<TokenResponse>("/api/auth/register", {
      method: "POST",
      body: JSON.stringify({ email, password, display_name }),
    }),
  login: (email: string, password: string) =>
    request<TokenResponse>("/api/auth/login", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    }),
  me: () => request<User>("/api/auth/me"),

  // ---------------------------------------------------------------- projects
  createProject: (payload: ProjectCreate) =>
    request<Project>("/api/projects", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  listProjects: (limit = 50, offset = 0) =>
    request<{ items: Project[]; total: number }>(
      `/api/projects?limit=${limit}&offset=${offset}`,
    ),
  getProject: (id: string) => request<Project>(`/api/projects/${id}`),
  deleteProject: (id: string) =>
    request<void>(`/api/projects/${id}`, { method: "DELETE" }),
  startProject: (id: string) =>
    request<Project>(`/api/projects/${id}/start`, { method: "POST" }),

  uploadMedia: async (projectId: string, file: File, autoStart = true) => {
    const form = new FormData();
    form.append("file", file);
    form.append("auto_start", String(autoStart));
    return request<Project>(`/api/projects/${projectId}/upload`, {
      method: "POST",
      body: form,
    });
  },

  fetchYoutube: (projectId: string) =>
    request<Project>(`/api/projects/${projectId}/youtube`, { method: "POST" }),

  // ---------------------------------------------------------------- clips
  listClips: (projectId: string) =>
    request<Clip[]>(`/api/projects/${projectId}/clips`),
  updateClip: (projectId: string, clipId: string, payload: ClipUpdate) =>
    request<Clip>(`/api/projects/${projectId}/clips/${clipId}`, {
      method: "PATCH",
      body: JSON.stringify(payload),
    }),
  renderClip: (projectId: string, clipId: string, force = false) =>
    request<{ task_id: string }>(
      `/api/projects/${projectId}/clips/${clipId}/render?force=${force}`,
      { method: "POST" },
    ),
  renderAllClips: (projectId: string, force = false) =>
    request<{ task_id: string }>(
      `/api/projects/${projectId}/render?force=${force}`,
      { method: "POST" },
    ),
  downloadZipUrl: (projectId: string) =>
    `/api/projects/${projectId}/download.zip`,

  /** Navigate to the ZIP download (the browser handles the file save). */
  downloadZip: (projectId: string) => {
    window.location.href = authUrl(api.downloadZipUrl(projectId));
  },

  // -------------------------------------------------------------- styles
  listStyles: () => request<SubtitleStyle[]>("/api/subtitle-styles"),

  // ---------------------------------------------------------------- sse
  /**
   * Subscribe to the project's SSE stream. Returns an unsubscribe function.
   * `onEvent` receives parsed events; EventSource handles auto-reconnect.
   */
  subscribeEvents: (
    projectId: string,
    onEvent: (evt: PipelineEvent) => void,
  ): (() => void) => {
    const token = getToken();
    // EventSource cannot set headers; pass the token as a query param.
    const url = `/api/projects/${projectId}/events?token=${encodeURIComponent(token ?? "")}`;
    const es = new EventSource(url);

    const handle = (type: string) => (raw: MessageEvent) => {
      try {
        const parsed = JSON.parse(raw.data) as PipelineEvent;
        onEvent({ ...parsed, type: parsed.type || type });
      } catch {
        /* ignore malformed payloads */
      }
    };
    ["stage", "progress", "log", "error", "done", "clip"].forEach((t) =>
      es.addEventListener(t, handle(t) as EventListener),
    );
    return () => es.close();
  },
};
