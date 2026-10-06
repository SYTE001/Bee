import { Link } from "react-router-dom";

import { useAuth } from "../hooks/useAuth";

export function Navbar() {
  const { user, logout } = useAuth();

  return (
    <header className="sticky top-0 z-20 border-b border-forge-border bg-forge-bg/80 backdrop-blur">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3">
        <Link to="/" className="flex items-center gap-2 font-bold">
          <span className="grid h-8 w-8 place-items-center rounded-lg bg-gradient-to-br from-forge-accent to-forge-accent2 text-sm">
            CF
          </span>
          <span className="text-lg tracking-tight">ClipForge</span>
        </Link>

        <nav className="flex items-center gap-4 text-sm text-forge-muted">
          {user ? (
            <>
              <Link to="/projects" className="hover:text-forge-text">
                Projects
              </Link>
              <Link to="/upload" className="btn-primary !px-3 !py-1.5">
                New clip
              </Link>
              <div className="flex items-center gap-2">
                <span className="hidden sm:inline">{user.email}</span>
                <button onClick={logout} className="btn-ghost !px-3 !py-1.5">
                  Log out
                </button>
              </div>
            </>
          ) : (
            <>
              <Link to="/login" className="hover:text-forge-text">
                Log in
              </Link>
              <Link to="/register" className="btn-primary !px-3 !py-1.5">
                Sign up
              </Link>
            </>
          )}
        </nav>
      </div>
    </header>
  );
}
