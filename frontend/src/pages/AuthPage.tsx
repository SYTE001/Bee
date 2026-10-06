import { useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";

import { ApiError } from "../api/client";
import { useAuth } from "../hooks/useAuth";
import { Notice } from "../components/ui";

type Mode = "login" | "register";

export function AuthPage({ mode }: { mode: Mode }) {
  const { login, register } = useAuth();
  const navigate = useNavigate();
  const [params] = useSearchParams();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const next = params.get("next") ?? "/projects";

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      if (mode === "register") {
        await register(email, password, name || undefined);
      } else {
        await login(email, password);
      }
      navigate(next);
    } catch (err) {
      const detail =
        err instanceof ApiError
          ? typeof err.detail === "string"
            ? err.detail
            : "invalid credentials"
          : "network error";
      setError(detail);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mx-auto flex max-w-md flex-col gap-6 px-4 py-16">
      <div>
        <h1 className="text-3xl font-bold">
          {mode === "register" ? "Create your account" : "Welcome back"}
        </h1>
        <p className="mt-2 text-forge-muted">
          {mode === "register"
            ? "Start turning long videos into viral clips."
            : "Log in to manage your ClipForge projects."}
        </p>
      </div>

      <form onSubmit={submit} className="card flex flex-col gap-4">
        {mode === "register" && (
          <label className="block">
            <span className="label">Display name (optional)</span>
            <input
              className="input"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Jane Creator"
              maxLength={120}
            />
          </label>
        )}
        <label className="block">
          <span className="label">E-mail</span>
          <input
            className="input"
            type="email"
            required
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="you@example.com"
          />
        </label>
        <label className="block">
          <span className="label">Password</span>
          <input
            className="input"
            type="password"
            required
            minLength={6}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="••••••••"
          />
        </label>

        {error && <Notice kind="warn">{error}</Notice>}

        <button type="submit" className="btn-primary w-full" disabled={busy}>
          {busy
            ? "Please wait…"
            : mode === "register"
              ? "Create account"
              : "Log in"}
        </button>
      </form>

      <p className="text-center text-sm text-forge-muted">
        {mode === "register" ? "Already have an account?" : "New to ClipForge?"}{" "}
        <a
          href={mode === "register" ? "/login" : "/register"}
          className="text-violet-400 hover:underline"
        >
          {mode === "register" ? "Log in" : "Sign up"}
        </a>
      </p>
    </div>
  );
}
