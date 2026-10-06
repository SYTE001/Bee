"""Local filesystem storage.

All artifacts live under STORAGE_DIR/<project_id>/<filename>. Only relative
paths are persisted in the DB; absolute resolution happens here.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from app.core.config import settings
from app.core.logging import get_logger

log = get_logger("storage")


class Storage:
    """Thin wrapper around the local artifacts directory."""

    def __init__(self, base_dir: str | Path | None = None) -> None:
        self.base = Path(base_dir or settings.STORAGE_DIR).resolve()
        self.base.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ paths
    def project_dir(self, project_id: str) -> Path:
        d = self.base / project_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def clips_dir(self, project_id: str) -> Path:
        d = self.project_dir(project_id) / "clips"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def abspath(self, rel_path: str) -> Path:
        """Resolve a stored relative path to an absolute filesystem path."""
        p = (self.base / rel_path).resolve()
        # Prevent traversal outside base
        if self.base not in p.parents and p != self.base:
            raise ValueError(f"unsafe path outside storage: {rel_path}")
        return p

    def relpath(self, abs_path: str | Path) -> str:
        """Inverse of `abspath`."""
        return str(Path(abs_path).resolve().relative_to(self.base)).replace("\\", "/")

    # ------------------------------------------------------------------ files
    def save_upload(self, project_id: str, filename: str, data: bytes) -> Path:
        """Write raw upload bytes into the project dir with a safe name."""
        safe = self._safe_name(filename)
        target = self.project_dir(project_id) / f"source_{safe}"
        target.write_bytes(data)
        log.info("saved upload %s (%d bytes)", target, len(data))
        return target

    def exists(self, rel_path: str | None) -> bool:
        return bool(rel_path) and self.abspath(rel_path).exists()

    def open(self, rel_path: str, mode: str = "rb"):
        return open(self.abspath(rel_path), mode)

    def size(self, rel_path: str) -> int:
        return self.abspath(rel_path).stat().st_size

    # ------------------------------------------------------------------ cleanup
    def delete_project(self, project_id: str) -> None:
        """Delete a whole project directory (best effort)."""
        d = self.base / project_id
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
            log.info("deleted project storage %s", d)

    def cleanup_old(self, max_age_days: int) -> int:
        """Delete project dirs untouched for `max_age_days`. Returns count."""
        import time

        cutoff = time.time() - max_age_days * 86400
        removed = 0
        for child in self.base.iterdir():
            if child.is_dir() and child.stat().st_mtime < cutoff:
                shutil.rmtree(child, ignore_errors=True)
                removed += 1
        if removed:
            log.info("cleanup removed %d stale project dirs", removed)
        return removed

    @staticmethod
    def _safe_name(filename: str) -> str:
        """Strip path components & weird chars, keep the extension."""
        name = os.path.basename(filename)
        stem, ext = os.path.splitext(name)
        ext = ext.lower()[:8]
        keep = "-_." + "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        stem = "".join(c if c in keep else "_" for c in stem)[:120] or "video"
        return f"{stem}{ext}"


storage = Storage()
