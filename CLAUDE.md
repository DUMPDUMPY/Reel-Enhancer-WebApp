# CLAUDE.md

Local web app for short vertical clips (Reels/TikTok): upload from a phone over home Wi-Fi →
Thai speech-to-text subtitles (1–2 lines, timed to the real speech) → edit on the phone →
FFmpeg enhance + burn subtitles at 1080×1920 → download. User-facing text is Thai.

## Target environment

- Runs as a Docker Compose project in **Synology DSM Container Manager** (xpenology),
  Intel **Core Ultra 5 245K** (Arrow Lake), iGPU passed in via `/dev/dri`. No NVIDIA/CUDA.
- The user wants heavy work on the **iGPU, not the CPU**. Keep CPU paths as fallbacks only.
- Ollama runs in a separate DSM project (`ollama/ollama`, `OLLAMA_VULKAN=1`, port 11434);
  reached from here via `host.docker.internal` or the NAS IP.
- Phone is mainly **Android Chrome**; plain HTTP on the LAN, single user, no auth needed.

## Architecture

| Piece | Where | Notes |
|---|---|---|
| `whisper` service | `docker/whisper/` | whisper.cpp v1.9.5 `whisper-server`, built with Vulkan (needs `spirv-headers` to build). Downloads models into `./models` on first start. |
| `app` service | `Dockerfile`, `app/` | Python 3.14, FastAPI, FFmpeg + `intel-media-va-driver-non-free` (VAAPI). |

- `app/main.py` — routes. `app/jobs.py` — job folders (`/data/jobs/<id>/job.json`) and a
  **single-worker** queue; each task is a chain of nodes (`steps`) shown live in the UI.
- `app/pipeline.py` — task bodies + node definitions (`TRANSCRIBE_NODES`, `RENDER_NODES`, …).
  Wrap each step in `with jobs.step(job_id, key):`.
- `app/services/` — `transcriber` (whisper HTTP), `segmenter` (whisper JSON → cues),
  `thai_text` (width + wrapping), `proofreader` (Ollama), `subtitles` (.ass/.vtt),
  `renderer` (FFmpeg commands, iGPU capability probe, fallbacks), `media` (ffprobe).
- `app/bench.py` — Ollama model comparison. Runs are stored as jobs with `kind: "bench"`
  (excluded from `/api/jobs`), one node per model; built-in test set has an answer key.
  If you change the proofreading prompt or validation, re-check scores here.
- `app/static/` — vanilla HTML/CSS/JS, no build step, no CDN (must work offline on the LAN).

## Rules that are easy to break

- **Thai has no spaces.** Never let libass or an LLM wrap lines. Line breaks come from
  `thai_text.wrap_lines` (PyThaiNLP, hint names added to the dictionary), written as `\N`
  with `WrapStyle: 2`. Width counts base glyphs only (`display_width` skips marks above/below).
- **Timing comes only from whisper.** The Ollama proofreader may fix spelling inside a cue;
  it must not add/drop/merge cues. Its answer is JSON-schema constrained, validated by id,
  and offered as per-cue suggestions the user accepts. Ollama failure is a `warn` node, never a job error.
- **whisper.cpp + VAD**: segment times are mapped back to the original audio, token times are
  not. `segmenter._segment_tokens` rescales token times onto each segment — keep it.
- **Rendering**: VAAPI decode/scale/denoise/sharpen/encode on the iGPU, `hwdownload` only for
  pad/crop, fps and libass, then `hwupload`. Capabilities are probed with tiny test encodes
  (`renderer.detect_caps`); every render tries `gpu` → `gpu-swdec` → `cpu`. Arrow Lake may need
  `-low_power 1` for H.264 — the probe decides, don't hard-code it.
- Output spec: 1080×1920, H.264 High, 30 fps, VBR 12M / max 15M, AAC 192k, `+faststart`, bt709 tags.
- Never stretch: non-9:16 sources are padded or cropped (`RenderOptions.fit`).

## Commands

```sh
pip install -r requirements-dev.txt     # needs ffmpeg with libass on PATH
pytest                                  # includes a real CPU render; whisper/Ollama are mocked
DATA_DIR=./data RENDER_MODE=cpu uvicorn app.main:app --reload
docker compose up -d --build            # on the NAS
```

There is no iGPU in CI/dev containers: VAAPI and Vulkan paths can only be verified on the NAS
(`docker logs reel-whisper`, `docker exec reel-enhancer vainfo`, `GET /api/system`,
`POST /api/hwcheck`). Say so instead of claiming they work.

## Conventions

- Match the existing style: small modules, type hints, short docstrings explaining *why*.
- User-facing strings (UI, errors, node labels) in Thai; code, comments and commits in English.
- Config only through env vars in `app/config.py` (documented in README + `docker-compose.yml`).
- Update `CHANGELOG.md` (Unreleased section) with every user-visible change.
- Fonts in `fonts/` are OFL; keep the license files next to them.
