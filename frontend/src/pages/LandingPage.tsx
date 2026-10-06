import { Link } from "react-router-dom";

import { useAuth } from "../hooks/useAuth";

const FEATURES = [
  {
    title: "Multi-signal moment detection",
    body: "Laughter, energy spikes, punchlines, emotion, and LLM semantic analysis combine into one explainable viral score.",
  },
  {
    title: "9:16 reframing with face tracking",
    body: "Crops follow the active speaker with smoothed motion, with split-screen, blurred-fit, and dynamic-zoom layouts.",
  },
  {
    title: "15 animated subtitle styles",
    body: "Word-by-word karaoke highlights, Hormozi-style pops, neon glow, emoji auto-annotation, and more.",
  },
  {
    title: "Hook-first clipping",
    body: "Every clip snaps to sentence boundaries, and the strongest sentence can be promoted to the first three seconds.",
  },
];

export function LandingPage() {
  const { user } = useAuth();

  return (
    <div className="mx-auto max-w-5xl px-4 py-20">
      <div className="text-center">
        <span className="badge border border-forge-border bg-forge-panel2 text-violet-300">
          Long video → vertical viral clips
        </span>
        <h1 className="mt-6 text-5xl font-bold tracking-tight">
          Turn podcasts into{" "}
          <span className="bg-gradient-to-r from-forge-accent to-forge-accent2 bg-clip-text text-transparent">
            TikTok-ready clips
          </span>
        </h1>
        <p className="mx-auto mt-5 max-w-2xl text-lg text-forge-muted">
          ClipForge analyzes podcasts, interviews, webinars, and vlogs — then
          finds the best moments, reframes them to 9:16, and burns in animated
          subtitles automatically.
        </p>
        <div className="mt-8 flex justify-center gap-3">
          <Link to={user ? "/upload" : "/register"} className="btn-primary px-6 py-3">
            {user ? "Upload a video" : "Get started free"}
          </Link>
          <Link to={user ? "/projects" : "/login"} className="btn-ghost px-6 py-3">
            {user ? "My projects" : "Log in"}
          </Link>
        </div>
      </div>

      <div className="mt-20 grid grid-cols-1 gap-4 md:grid-cols-2">
        {FEATURES.map((f) => (
          <div key={f.title} className="card">
            <h3 className="font-semibold">{f.title}</h3>
            <p className="mt-2 text-sm text-forge-muted">{f.body}</p>
          </div>
        ))}
      </div>

      <div className="card mt-12 text-center text-sm text-forge-muted">
        ⚠️ Please only process content you own or are licensed to redistribute.
        Respect creators' copyrights.
      </div>
    </div>
  );
}
