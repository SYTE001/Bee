"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import auth, clips, projects
from app.core.config import settings
from app.core.logging import get_logger, setup_logging
from app.db.session import create_all
from app.services import task_runner

log = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    log.info("starting %s %s", settings.APP_NAME, settings.APP_VERSION)
    create_all()
    try:
        yield
    finally:
        task_runner.shutdown()
        log.info("shutdown complete")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(auth.router, prefix=settings.API_PREFIX)
    app.include_router(projects.router, prefix=settings.API_PREFIX)
    app.include_router(clips.router, prefix=settings.API_PREFIX)

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "app": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "task_backend": settings.TASK_BACKEND,
        }

    # Copyright reminder surfaced in the UI as well.
    @app.get("/")
    def root() -> dict:
        return {
            "name": settings.APP_NAME,
            "docs": "/docs",
            "notice": (
                "Only process content you own or are licensed to redistribute."
            ),
        }

    # Serve the built frontend (production single-container mode).
    import os

    static_dir = os.path.join(os.path.dirname(settings.STORAGE_DIR), "frontend")
    if os.path.isdir(static_dir):
        app.mount(
            "/",
            StaticFiles(directory=static_dir, html=True),
            name="frontend",
        )
    return app


app = create_app()
