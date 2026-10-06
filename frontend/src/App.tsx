import { Navigate, Route, Routes } from "react-router-dom";

import { useAuth } from "./hooks/useAuth";
import { Navbar } from "./components/Navbar";
import { AuthPage } from "./pages/AuthPage";
import { ClipEditorPage } from "./pages/ClipEditorPage";
import { LandingPage } from "./pages/LandingPage";
import { ProjectPage } from "./pages/ProjectPage";
import { ProjectsPage } from "./pages/ProjectsPage";
import { UploadPage } from "./pages/UploadPage";

function Protected({ children }: { children: React.ReactNode }) {
  const { user, loading } = useAuth();
  if (loading) {
    return (
      <div className="py-20 text-center text-forge-muted">Loading…</div>
    );
  }
  if (!user) return <Navigate to="/login" replace />;
  return <>{children}</>;
}

export function App() {
  return (
    <div className="flex min-h-screen flex-col">
      <Navbar />
      <main className="flex-1">
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/login" element={<AuthPage mode="login" />} />
          <Route path="/register" element={<AuthPage mode="register" />} />
          <Route
            path="/upload"
            element={
              <Protected>
                <UploadPage />
              </Protected>
            }
          />
          <Route
            path="/projects"
            element={
              <Protected>
                <ProjectsPage />
              </Protected>
            }
          />
          <Route
            path="/projects/:projectId"
            element={
              <Protected>
                <ProjectPage />
              </Protected>
            }
          />
          <Route
            path="/projects/:projectId/clips/:clipId"
            element={
              <Protected>
                <ClipEditorPage />
              </Protected>
            }
          />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
      <footer className="border-t border-forge-border py-6 text-center text-xs text-forge-muted">
        ClipForge · only process content you own or are licensed to redistribute
      </footer>
    </div>
  );
}
