import type { ReactNode } from "react";

const CATEGORY_COLORS: Record<string, string> = {
  funny: "bg-yellow-500/20 text-yellow-300",
  serious: "bg-blue-500/20 text-blue-300",
  insight: "bg-emerald-500/20 text-emerald-300",
  emotional: "bg-pink-500/20 text-pink-300",
  controversial: "bg-red-500/20 text-red-300",
  story: "bg-purple-500/20 text-purple-300",
  actionable: "bg-cyan-500/20 text-cyan-300",
};

export function CategoryBadge({ category }: { category: string }) {
  const key = category.toLowerCase();
  const cls = CATEGORY_COLORS[key] ?? "bg-forge-border text-forge-muted";
  return <span className={`badge ${cls}`}>{category || "unclassified"}</span>;
}

export function ScoreRing({ score, size = 64 }: { score: number; size?: number }) {
  const radius = size / 2 - 6;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference * (1 - Math.max(0, Math.min(100, score)) / 100);
  const color = score >= 75 ? "#34d399" : score >= 55 ? "#fbbf24" : "#f87171";

  return (
    <div className="relative" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90">
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="#2a2a40"
          strokeWidth={5}
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth={5}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          style={{ transition: "stroke-dashoffset 0.6s ease" }}
        />
      </svg>
      <span
        className="absolute inset-0 grid place-items-center text-sm font-bold"
        style={{ color }}
      >
        {Math.round(score)}
      </span>
    </div>
  );
}

export function StatusBadge({ status }: { status: string }) {
  const done = status === "completed";
  const failed = status === "failed";
  const active = !done && !failed && status !== "pending";
  const cls = done
    ? "bg-emerald-500/20 text-emerald-300"
    : failed
      ? "bg-red-500/20 text-red-300"
      : active
        ? "bg-forge-accent/30 text-violet-300"
        : "bg-forge-border text-forge-muted";
  return (
    <span className={`badge ${cls}`}>
      {active && <span className="mr-1.5 h-1.5 w-1.5 animate-pulse rounded-full bg-current" />}
      {status.replace("_", " ")}
    </span>
  );
}

export function Notice({ children, kind = "info" }: { children: ReactNode; kind?: "info" | "warn" }) {
  const cls =
    kind === "warn"
      ? "border-yellow-500/40 bg-yellow-500/10 text-yellow-200"
      : "border-forge-accent/40 bg-forge-accent/10 text-violet-100";
  return (
    <div className={`rounded-lg border p-3 text-sm ${cls}`}>{children}</div>
  );
}

export function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}
