# ClipForge backend — API + pipeline worker.
#
# Build args:
#   BASE_IMAGE  nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 for GPU,
#               ubuntu:22.04 for CPU-only hosts.
#   ENABLE_ML   true (default) installs the heavy ML stack (whisper, torch,
#               pyannote). Set false for a slim API/worker image.
#
# ffmpeg is installed system-wide; CUDA images pick up NVENC from the driver.

ARG BASE_IMAGE=ubuntu:22.04
FROM ${BASE_IMAGE}

ARG ENABLE_ML=true

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    STORAGE_DIR=/data/storage \
    MODELS_DIR=/data/models \
    DATABASE_URL=sqlite:////data/clipforge.db

# NVIDIA runtime needs these capabilities for NVENC + CUDA
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
        ffmpeg \
        fonts-dejavu fonts-liberation \
        curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && python -m pip install --no-cache-dir --upgrade pip

WORKDIR /app

# Dependencies are copied first so source changes don't bust the layer cache.
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY requirements-ml.txt .
RUN if [ "$ENABLE_ML" = "true" ]; then \
        python -m pip install --no-cache-dir -r requirements-ml.txt; \
    else \
        echo "ML deps skipped (ENABLE_ML=false)"; \
    fi

COPY app ./app
COPY styles ./styles
COPY eval ./eval
COPY scripts ./scripts

RUN mkdir -p /data/storage /data/models
VOLUME ["/data"]

EXPOSE 8000

# docker-compose overrides this for the worker service.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
