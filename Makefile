# ClipForge developer shortcuts. Run from the repo root.
#
# Backend commands assume `cd backend`; the env vars there point at ./data.

PY      ?= python
PIP     ?= $(PY) -m pip
COMPOSE ?= docker compose

.PHONY: help install install-ml dev dev-frontend test test-fast lint build clean seed eval docker-up docker-up-gpu docker-down

help:
	@echo "ClipForge — common tasks"
	@echo ""
	@echo "  make install         install backend deps (no ML stack)"
	@echo "  make install-ml      also install whisper / torch / pyannote (GPU recommended)"
	@echo "  make dev             start the API on :8000 and the frontend on :5173"
	@echo "  make test            full backend test suite"
	@echo "  make test-fast       skip the whisper-model-dependent tests"
	@echo "  make build           production frontend build"
	@echo "  make seed            create a demo project with real rendered clips"
	@echo "  make eval            run precision@k against eval/labels.example.json"
	@echo "  make docker-up       full stack on CPU"
	@echo "  make docker-up-gpu   full stack with NVIDIA GPU + NVENC"
	@echo "  make docker-down     stop the stack"
	@echo "  make clean           remove build artifacts and local data"

install:
	cd backend && $(PIP) install -r requirements.txt
	cd frontend && npm install

install-ml: install
	cd backend && $(PIP) install -r requirements-ml.txt

# --- local dev: API (:8000) + vite (:5173), both with hot reload ----------
dev:
	cd backend && $(PY) -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000 &
	cd frontend && npx vite --port 5173

dev-frontend:
	cd frontend && npx vite --port 5173

# --- tests ----------------------------------------------------------------
test:
	cd backend && $(PY) -m pytest -q

# whisper needs a model download; skip it for a quick local loop
test-fast:
	cd backend && $(PY) -m pytest -q --ignore=tests/test_transcribe.py

# --- build / demo / eval ---------------------------------------------------
build:
	cd frontend && npm run build

seed:
	cd backend && $(PY) scripts/seed_demo.py

eval:
	cd backend && $(PY) -m eval.precision_at_k --labels eval/labels.example.json

# --- docker ----------------------------------------------------------------
docker-up:
	$(COMPOSE) up --build

docker-up-gpu:
	$(COMPOSE) -f docker-compose.yml -f docker-compose.gpu.yml up --build

docker-down:
	$(COMPOSE) down

clean:
	rm -rf frontend/dist frontend/node_modules
	rm -rf backend/data/storage backend/data/models backend/data/clipforge.db
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .pytest_cache -prune -exec rm -rf {} +
