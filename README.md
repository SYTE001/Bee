# ClipForge

Turn long videos — podcasts, interviews, webinars, vlogs — into vertical,
caption-ready clips for TikTok / Reels / Shorts.

Upload a video or paste a URL. ClipForge transcribes it with word-level
timestamps, detects speakers, finds the viral moments with a multi-signal
scorer, reframes 16:9 → 9:16 with face tracking, burns in animated subtitles,
and renders 1080×1920 mp4s you can download individually or as a ZIP.

> ⚠️ **Copyright.** Only process content you own or are licensed to
> redistribute. ClipForge shows this reminder in the upload flow and the API
> root; the responsibility for source material is yours.

---

## Table of contents

- [Pipeline](#pipeline)
- [Hardware requirements](#hardware-requirements)
- [Quick start (local)](#quick-start-local)
- [Docker](#docker)
- [Configuration](#configuration)
- [Subtitle styles](#subtitle-styles)
- [Evaluation (precision@k)](#evaluation-precisionk)
- [Project layout](#project-layout)
- [Testing](#testing)
- [How the pieces fit](#how-the-pieces-fit)

---

## Pipeline

Ten stages, each idempotent and resumable — completed stages are cached in the
artifact store, so a failure mid-run resumes from where it stopped.

| # | Stage | What happens |
|---|---|---|
| 1 | Ingest | Upload (multipart) or `yt-dlp` URL fetch; media metadata via ffprobe |
| 2 | Audio extraction | FFmpeg demux + loudness normalisation toward `TARGET_LUFS` |
| 3 | Transcription | faster-whisper (`large-v3` default) with word-level timestamps, cached |
| 4 | Diarisation | pyannote.audio speaker turns (needs `HF_TOKEN`; optional) |
| 5 | Analysis | Audio features (energy, pitch, speech rate, laughter) + LLM moment detection (Claude) + visual signals |
| 6 | Segmentation | Sentence groups → sliding windows snapped to sentence boundaries |
| 7 | Scoring | Weighted ensemble (see `SCORE_WEIGHTS`), dedup, non-overlap, Top-N |
| 8 | Reframing | Face detection per frame, smoothed crop (EMA), active-speaker focus |
| 9 | Subtitles | ASS/libass generation with karaoke timing, keyword highlight, emoji |
| 10 | Render | 1080×1920 H.264/H.265 (NVENC when available), thumbnail, ZIP export |

The clip editor can re-run stages 8–10 per clip with different style, layout,
fps or trim, without touching the analysis cache.

## Hardware requirements

| Component | Minimum | Recommended |
|---|---|---|
| CPU | 8 cores | 16+ cores (rendering is CPU-bound without NVENC) |
| RAM | 8 GB | 32 GB (large-v3 + diarisation are hungry) |
| GPU | none | NVIDIA 8 GB VRAM — used for whisper, and NVENC cuts render time ~5× |
| Disk | 5 GB | 50 GB (models + source media + renders) |
| FFmpeg | 6.0+ | with `libx264`/`libx265`; NVENC needs an NVIDIA driver |

A 60-minute podcast renders roughly 3–6 clips per minute of wall-clock on a
16-core CPU box; roughly 5× faster with NVENC. VRAM use is tracked per stage,
and the worker concurrency cap (`WORKER_CONCURRENCY`) keeps memory in check.

### Low-power laptops (integrated graphics, 2–4 cores)

The defaults above target a workstation. On something like a ThinkPad L570
(i5-7200U, Intel HD 620) the defaults — `large-v3`, 60 fps, `medium` preset —
would take hours per video and peg the CPU the whole time. It still runs fine
if you tune it; see [`backend/.env`](backend/.env) for a working profile
tuned for "a few clips a day from short videos". The knobs that matter:

| Setting | Laptop value | Why |
|---|---|---|
| `WHISPER_MODEL` | `base.en` | `large-v3` takes hours on 2 cores; `base.en` is near-realtime |
| `WHISPER_BEAM_SIZE` | `2` | Halves decode cost, accuracy stays good on clean speech |
| `DIARIZATION_ENABLED` | `false` | pyannote costs minutes of CPU; skip unless it's an interview |
| `PREFERRED_FPS` | `30` | 60 fps doubles encode time and TikTok downsamples anyway |
| `H264_PRESET` | `veryfast` | ~1.2–1.5× faster than `medium` (measured on a vertical 15 s render); the gap widens on fewer cores |
| `H264_CRF` | `20` | Small extra saving; invisible after platform re-encode |
| `DEFAULT_TOP_N_CLIPS` | `5` | No point ranking 10 clips if you only publish 2–3 |
| `WORKER_CONCURRENCY` | `2` | Matches the core count so the laptop stays responsive |

Keep videos under ~20 minutes at these settings. Everything else in the app —
the editor, subtitle preview, scoring, ZIP export — is cheap and works the
same regardless of hardware.

## Quick start (local)

Needs: Python 3.11, Node 20+, FFmpeg 6+ on `PATH`.

```bash
git clone <repo> clipforge && cd clipforge
cp backend/.env.example backend/.env      # then edit SECRET_KEY + tokens
make install                              # backend + frontend deps
make dev                                  # API :8000, frontend :5173
```

Open <http://localhost:5173>, register, upload a video (or paste a URL), and
watch the pipeline run. `TASK_BACKEND=thread` (the default) runs everything
in-process — no Redis needed.

<details>
<summary><strong>No <code>make</code>? (Windows)</strong></summary>

The Makefile targets are thin wrappers. Run the underlying commands:

```bash
# install
cd backend && python -m pip install -r requirements.txt && cd ../frontend && npm install

# dev (two terminals)
cd backend && python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
cd frontend && npx vite --port 5173

# test
cd backend && python -m pytest -q --ignore=tests/test_transcribe.py

# demo + eval
cd backend && python scripts/seed_demo.py
cd backend && python -m eval.precision_at_k --labels eval/labels.example.json
```

</details>

For the full ML stack (whisper, diarisation, LLM analysis):

```bash
make install-ml
```

and set in `backend/.env`:

```ini
WHISPER_MODEL=large-v3
HF_TOKEN=hf_...                    # accept the pyannote model conditions first
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
```

Any Anthropic Messages-compatible provider works too — point `LLM_BASE_URL`
at its base URL and set `ANTHROPIC_API_KEY`/`LLM_MODEL` to what that provider
issued, e.g. `LLM_BASE_URL=https://api.example.com/v1`.

Without `ANTHROPIC_API_KEY` the scorer falls back to keyword/heuristic moment
detection (`LLM_PROVIDER=none`) — lower quality, but it runs unattended.

**Fast smoke test** (no model downloads, no API keys):

```ini
WHISPER_MODEL=tiny
LLM_PROVIDER=none
DIARIZATION_ENABLED=false
```

## Docker

```bash
cp backend/.env.example backend/.env      # edit SECRET_KEY, HF_TOKEN, ANTHROPIC_API_KEY
make docker-up                            # CPU: api + worker + redis + db + frontend
```

Frontend on <http://localhost:8080>, API docs on <http://localhost:8000/docs>.

NVIDIA GPU (CUDA + NVENC):

```bash
make docker-up-gpu
```

The stack runs the same code with `TASK_BACKEND=celery`, a Postgres database,
and one worker container. Scale rendering horizontally by scaling the worker:

```bash
docker compose up --scale worker=3
```

## Configuration

Everything is environment-driven (`backend/.env`, copied from
`backend/.env.example`). Notable knobs:

| Var | Default | Effect |
|---|---|---|
| `TASK_BACKEND` | `thread` | `thread` = zero-infra; `celery` = Redis + worker containers |
| `WHISPER_MODEL` | `large-v3` | `tiny`/`base` for smoke tests |
| `WHISPER_DEVICE` | `auto` | `cuda` in the GPU compose profile |
| `LLM_PROVIDER` | `anthropic` | `none` disables Claude moment detection |
| `LLM_BASE_URL` | (empty) | Anthropic Messages-compatible endpoint (proxy/other provider); empty = official API |
| `SCORE_WEIGHTS` | see file | Viral-score ensemble; retune without code changes |
| `MIN_CLIP_SCORE` | `35` | Drop candidates below this viral score |
| `DEFAULT_TOP_N_CLIPS` | `10` | How many clips a project yields |
| `VIDEO_CODEC` | `auto` | `auto` picks NVENC when usable, else libx264 |
| `NVENC_MAX_PARALLEL` | `2` | Concurrent NVENC encodes (encoder slots + VRAM are the binding constraint) |
| `WATERMARK_TEXT` | (empty) | Burned into every clip when set |
| `HOOK_TITLE_SECONDS` | `3.5` | Opening title overlay; `0` disables |
| `CENSOR_PROFANITY` | `false` | Masks profanity in subtitles |
| `MAX_JOBS_PER_DAY` | `20` | Per-user pipeline quota |

No secrets are hardcoded anywhere; the repo ships no keys.

## Subtitle styles

15 presets ship in [`backend/styles/`](backend/styles). Adding one is a single
JSON file drop — no restart, no code. The clip editor shows a live preview
using the same font, colours, outline and word grouping the burn-in produces.

Full field reference: [backend/docs/SUBTITLE_STYLES.md](backend/docs/SUBTITLE_STYLES.md).

## Evaluation (precision@k)

`backend/eval/precision_at_k.py` scores selection quality against hand-picked
ground truth:

```json
{
  "projects": [
    {
      "source": "podcast_ep01.mp4",
      "labels": [
        {"start": 12.0, "end": 45.0, "category": "insight"},
        {"start": 61.0, "end": 88.0, "category": "funny"}
      ]
    }
  ]
}
```

A clip is a hit when its temporal IoU with a label clears the threshold
(default `0.3`) and the category matches when the label specifies one.

```bash
python -m eval.precision_at_k --labels my_labels.json
python -m eval.precision_at_k --labels my_labels.json --k 1,3,5 --iou 0.25
python -m eval.precision_at_k --labels my_labels.json --predictions clips.json
```

It prints per-project precision@k, recall, mean best IoU and a per-clip hit
table, then a macro average. `make seed && make eval` runs it against a demo
project end-to-end.

## Project layout

```
clipforge/
├── backend/
│   ├── app/
│   │   ├── api/routes/         auth, projects, clips (render/ZIP/editor)
│   │   ├── core/               config, logging, security (JWT)
│   │   ├── db/                 SQLAlchemy models + session
│   │   ├── pipelines/          the 10-stage video pipeline
│   │   ├── schemas/            Pydantic v2 request/response models
│   │   ├── services/           ingest, transcribe, diarize, scoring,
│   │   │                       reframe, subtitle, render, storage, tasks
│   │   └── workers/            celery app (production task backend)
│   ├── eval/                   precision@k + example labels
│   ├── styles/                 subtitle preset JSON files
│   ├── scripts/                seed_demo.py, smoke runs
│   ├── tests/                  pytest suite
│   └── docs/SUBTITLE_STYLES.md
├── frontend/                   React + Vite + TypeScript + Tailwind
├── docker-compose.yml          CPU stack
├── docker-compose.gpu.yml      NVIDIA override
└── Makefile                    install / dev / test / docker shortcuts
```

## Testing

```bash
make test-fast      # full suite minus the whisper-download test
make test           # everything
```

The suite exercises the real FFmpeg path (synthetic media generated in
fixtures), so filter plumbing — not just unit logic — is covered: ASS
generation with a pixel-visibility check, the reframing crop plans, render
output geometry, ZIP export, and the API surface.

## How the pieces fit

- **Frontend** talks to the API over REST; pipeline progress arrives over SSE
  (`/api/projects/{id}/events`) with polling as a fallback. Media URLs carry
  the JWT as a query parameter because `<video>`/`<img>`/navigation can't set
  headers.
- **API** writes project/clip rows and hands work to the task runner.
- **Task runner** dispatches either to a thread pool (dev) or Celery
  (production); both call the same pipeline entrypoint, so behaviour is
  identical.
- **Artifact store** keeps per-project JSON/media artifacts (transcript,
  diarisation, audio/visual features, crop plans) so a re-run only redoes what
  changed — including a single clip's render after an edit.
- **Scoring** is a configurable weighted ensemble; the eval script is how you
  decide whether a retune actually helped.

---

Built for people who'd rather ship clips than scrub timelines.
