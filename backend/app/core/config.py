"""Application configuration via environment variables (Pydantic v2 settings).

Everything is configurable through `.env` — no secrets are ever hardcoded.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, DotEnvSettingsSource, SettingsConfigDict

# Anchored to this file so the app finds `.env` no matter which working
# directory the server is launched from (`make dev` cds to backend/, but a
# bare `uvicorn app.main:app` from the repo root would otherwise silently
# miss every setting and fall back to code defaults).
BACKEND_DIR = Path(__file__).resolve().parents[2]
ENV_FILE = BACKEND_DIR / ".env"


class _LLMCredentialsDotenvSource(DotEnvSettingsSource):
    """Read LLM credentials from `.env` with priority over process env vars.

    A machine may export ANTHROPIC_API_KEY / ANTHROPIC_BASE_URL globally for
    other tooling. Pydantic ranks process env above the dotenv file, so those
    would shadow the project's `.env` and silently point the scorer at a
    different endpoint with a foreign key. Restricting the override to the LLM
    fields keeps Docker's `environment:` block authoritative for the rest
    (DATABASE_URL, TASK_BACKEND, ...).
    """

    _llm_fields = frozenset({"LLM_PROVIDER", "ANTHROPIC_API_KEY", "LLM_BASE_URL", "LLM_MODEL"})

    def __call__(self) -> dict:  # type: ignore[override]
        data = super().__call__()
        return {k: v for k, v in data.items() if k in self._llm_fields and v}



class Settings(BaseSettings):
    """Global settings, loaded from environment / `.env` file."""

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings):  # type: ignore[override]
        # Order = priority (first wins). The LLM credential source sits ahead of
        # env_settings so a machine-global ANTHROPIC_API_KEY can't shadow .env;
        # every other field still resolves from env_settings as normal.
        return (
            init_settings,
            _LLMCredentialsDotenvSource(settings_cls),
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )

    # ------------------------------------------------------------------ app
    APP_NAME: str = "ClipForge"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False
    API_PREFIX: str = "/api"

    # Allowed CORS origins (comma separated)
    CORS_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173"

    # ---------------------------------------------------------------- security
    SECRET_KEY: str = Field(default="change-me-in-production")
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
    MAX_JOBS_PER_DAY: int = 20  # per-user job quota

    # ---------------------------------------------------------------- storage
    STORAGE_DIR: str = "./data/storage"
    MODELS_DIR: str = "./data/models"
    MAX_UPLOAD_GB: float = 4.0
    CLEANUP_AFTER_DAYS: int = 7  # auto-clean finished projects older than N days

    # ---------------------------------------------------------------- database
    # PostgreSQL in docker; sqlite fallback keeps local dev zero-config.
    DATABASE_URL: str = "sqlite:///./data/clipforge.db"

    @field_validator("STORAGE_DIR", "MODELS_DIR", "DATABASE_URL", mode="after")
    @classmethod
    def _anchor_relative_paths(cls, v: str) -> str:
        # `.env` paths are written relative to backend/, but the process cwd
        # depends on how the server was launched. Anchor them to the backend
        # directory so `data/` always lands in the same place.
        if v.startswith("sqlite:///./"):
            return f"sqlite:///{BACKEND_DIR}/{v[len('sqlite:///./'):]}"
        if v.startswith("./"):
            return str(BACKEND_DIR / v[2:])
        return v

    # ---------------------------------------------------------------- tasks
    TASK_BACKEND: Literal["celery", "thread"] = "thread"
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"
    REDIS_URL: str = "redis://localhost:6379/0"
    WORKER_CONCURRENCY: int = 2

    # ---------------------------------------------------------------- ingest
    YT_DLP_FORMAT: str = "bv*[height<=1080]+ba/b[height<=1080]/b"

    # ---------------------------------------------------------------- whisper
    WHISPER_MODEL: str = "large-v3"
    WHISPER_DEVICE: str = "auto"  # auto | cpu | cuda
    WHISPER_COMPUTE_TYPE: str = "auto"  # auto | int8 | float16 | int8_float16
    WHISPER_BEAM_SIZE: int = 5
    TRANSCRIBE_CACHE: bool = True

    # ---------------------------------------------------------------- pyannote
    HF_TOKEN: str = ""  # required for pyannote speaker diarization
    DIARIZATION_ENABLED: bool = True

    # ---------------------------------------------------------------- LLM
    LLM_PROVIDER: Literal["anthropic", "none"] = "anthropic"
    ANTHROPIC_API_KEY: str = ""
    # Optional: point the Anthropic client at an Anthropic Messages-compatible
    # endpoint (e.g. a proxy or alternative provider). Empty = official API.
    LLM_BASE_URL: str = ""
    LLM_MODEL: str = "claude-sonnet-4-5"
    LLM_MAX_RETRIES: int = 4
    LLM_TIMEOUT_SECONDS: float = 120.0
    LLM_CHUNK_SECONDS: float = 600.0  # analyze transcript in ~10 min chunks

    # ---------------------------------------------------------------- scoring
    # Viral score = weighted ensemble of multi-signal scores (0..1 each).
    SCORE_WEIGHTS: str = (
        "llm_overall:0.30,hook:0.20,emotion:0.15,audio_energy:0.10,"
        "laughter:0.10,completeness:0.10,visual:0.05"
    )
    DEFAULT_TOP_N_CLIPS: int = 10
    MIN_CLIP_SCORE: float = 35.0

    # ---------------------------------------------------------------- render
    TARGET_LUFS: float = -14.0
    RENDER_HEIGHT: int = 1920
    RENDER_WIDTH: int = 1080
    PREFERRED_FPS: int = 60
    H264_CRF: int = 18
    H265_CRF: int = 22
    # libx264/libx265 presets. `medium` is the x264 default; on low-core CPUs
    # `veryfast`/`faster` trades a little efficiency for a large speedup.
    H264_PRESET: Literal[
        "ultrafast", "superfast", "veryfast", "faster", "fast",
        "medium", "slow", "slower", "veryslow",
    ] = "medium"
    H265_PRESET: Literal[
        "ultrafast", "superfast", "veryfast", "faster", "fast",
        "medium", "slow", "slower", "veryslow",
    ] = "medium"
    VIDEO_BITRATE_MAX: str = "16M"
    VIDEO_BITRATE_BUF: str = "24M"
    # auto prefers NVENC when a usable NVIDIA encoder is present, else libx264.
    VIDEO_CODEC: Literal["auto", "h264", "h265"] = "auto"
    NVENC_CQ: int = 20  # constant-quality target for the NVENC VBR rate control
    # Concurrent NVENC sessions: consumer GPUs run out of encoder slots (and
    # VRAM is shared with whisper) well before the CPU cap bites.
    NVENC_MAX_PARALLEL: int = 2
    # Burned-in extras. Leave WATERMARK_TEXT empty to skip the watermark.
    WATERMARK_TEXT: str = ""
    HOOK_TITLE_SECONDS: float = 3.5  # 0 disables the opening title overlay
    CENSOR_PROFANITY: bool = False
    AUTO_EMOJI: bool = True  # only applies to styles whose preset enables emoji
    # True motion interpolation (minterpolate) is slow; off = frame duplication.
    FRAME_INTERPOLATION: bool = False

    @field_validator("CORS_ORIGINS")
    @classmethod
    def _split_origins(cls, v: str) -> list[str]:
        return [o.strip() for o in v.split(",") if o.strip()]

    @property
    def score_weights(self) -> dict[str, float]:
        """Parse `SCORE_WEIGHTS` string into a dict."""
        weights: dict[str, float] = {}
        for pair in self.SCORE_WEIGHTS.split(","):
            pair = pair.strip()
            if not pair:
                continue
            key, _, raw = pair.partition(":")
            try:
                weights[key.strip()] = float(raw)
            except ValueError:
                continue
        return weights

    @property
    def cors_origin_list(self) -> list[str]:
        return self.CORS_ORIGINS  # type: ignore[return-value]  # validator returns list


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()


settings = get_settings()
