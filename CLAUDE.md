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
- `app/services/llm.py` — proofreading LLM providers: Ollama + OpenAI-compatible APIs keyed by env
  (`ZAI_API_KEY`, `TYPHOON_API_KEY`, `OPENROUTER_API_KEY`, `GEMINI_API_KEY`, `CUSTOM_LLM_*`).
  Model refs are `provider:model`; a name without a known prefix is an Ollama model.
  Cloud JSON mode is `json_object` (no schema), so validation + one retry in `proofreader` matter.
- `app/bench.py` — proofreading model comparison (Ollama and cloud). Runs are stored as jobs with `kind: "bench"`
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

- `app/services/stt_cloud.py` — ElevenLabs Scribe STT (`ELEVENLABS_API_KEY`), output converted to
  whisper verbose_json so `segmenter` is shared. Picked when the model id starts with `elevenlabs:`.

## Verified on the NAS (Core Ultra 5 245K, DSM Container Manager)

- whisper.cpp Vulkan: `ggml_vulkan: 0 = Intel(R) Graphics (ARL) (Intel open-source Mesa driver)`,
  `uma: 1 | fp16: 1 | int dot: 1`; the model loads on `Vulkan0`. `llvmpipe` is also listed and ignored.
- VAAPI: Intel iHD 25.2.3; H.264 High has `EncSlice` only (no `EncSliceLP`), so the probe picks
  `h264_vaapi` with `low_power=0`. `RENDER_MODE=auto` → `vaapi`.
- Container Manager rejects `cpus:` and doesn't create missing bind-mount folders (`data/`, `models/` ship in the repo).
- Silero VAD can find **no speech** in a song (singing over music) → whisper returns nothing;
  `transcribe_task` retries once without VAD.
- ElevenLabs Scribe with `language_code: "th"` works (returns `tha`). Thai `words[]` are single
  characters/clusters, not words — never split segments on pauses between them (`stt_cloud.to_whisper_json`).
- Cloud calls to Z.AI and ElevenLabs work from the NAS.
- Proofreading bench on the NAS (built-in set): `zai:glm-5.3` 75 (7/8, 1 wrong, 24 s; run before `reasoning_effort`
  was sent, see below); `sailor2:8b` via Ollama Vulkan on the iGPU 50 (5/8, 98.5 s, 4.9 tok/s, 2 rewrites
  rejected; once added "(Meow)" after a name). The 245K iGPU is small — expect ~1 min per 10-cue batch for 8B.

## Testing with real API keys

Keys are provided only as environment variables/secrets, never in chat: `ZAI_API_KEY`,
`ELEVENLABS_API_KEY`. Results so far (2026-10-25):
- **ElevenLabs Scribe — verified.** Test audio: `eleven_v3` TTS of the bench sentences
  (`eleven_multilingual_v2` rejects `language_code: th`). `language_code: "th"` and `keyterms` are
  accepted (response `language_code: "tha"`). 12/14 sentences exact (the 2 misses are TTS saying
  นะ/มาก for the slang), segment timing within ~0.1–0.16 s of the reference, ~3.5 s per 36 s clip.
  whisper.cpp large-v3-q5_0 (CPU, VAD) on the same audio: 5/14 exact, name errors (กุ้งไก่, โมจิว),
  without a hint its segments merge two sentences and cue timing is off by ~0.55 s.
- **Open:** Whisper's `HALLUCINATIONS` list drops real lines like "อย่าลืมกดติดตามกันด้วยนะ"
  (seen with the hint on). Needs real hallucination samples to tune `no_speech_prob`/`avg_logprob`.
- **Z.AI — verified (Coding Plan key).** Endpoint `/api/coding/paas/v4`; the plan covers only
  `glm-5.3` and `glm-5.3-flash` (other models → 1113/1314; Coding Plan keys are separate from
  Standard keys since 2026-10-16). GLM 5.3 always thinks: the Coding Plan endpoint silently ignores
  `thinking: disabled` (the Standard endpoint rejects it with 1210), so `llm` sends
  `reasoning_effort` (`ZAI_REASONING_EFFORT`, low/high/max, default high — same score as max, fewer tokens) up front for `glm-5.3*`.
  Built-in set, 3 runs each: glm-5.3 max 100 (15 s), high 100 (13 s), low 88/88/100 (5–6 s);
  glm-5.3-flash max 50/88/88, high 62/75/75, low 50–88. No over-edits of slang from either.
  Reference, CPU Ollama: sailor2:8b 62, gemma3:4b 50, Typhoon 2.5 4B 50, Typhoon 2.1 4B 38,
  qwen2.5:3b 0 (wrong edits).

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
