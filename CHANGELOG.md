# Changelog

All notable changes to this project are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Fixed
- Proofreading a long clip (e.g. a 3-minute song, ~40 cues) with an 8B Ollama model timed out after
  2 minutes. Cues are now sent in batches of `PROOFREAD_BATCH` (10) with per-batch progress on the
  node; a failed batch is skipped instead of losing the whole result. `OLLAMA_TIMEOUT` default 120 → 300 s per batch.
- Whisper with VAD on a song could return no subtitles at all (Silero VAD found no speech in the
  singing). If VAD finds nothing, the clip is now transcribed again without VAD, shown on the node.
- ElevenLabs subtitles no longer split inside a Thai word (e.g. `ไห` | `น` on a held note in a song).
  Scribe returns Thai as single characters, so pauses are now split per PyThaiNLP word by the segmenter.
- DSM Container Manager: removed the `cpus:` limit from `docker-compose.yml` (not supported there) and
  ship empty `data/` and `models/` folders, which must exist before the first start.

### Added
- **Delete jobs** from the recent-jobs list (🗑 per job, or "ลบงานทั้งหมด") to free disk space; each job
  shows its size on disk and the list header shows the total. Comparison runs can be deleted too.
- **Cloud-first defaults**: with `ELEVENLABS_API_KEY` set, `elevenlabs:scribe_v2` is the default
  speech-to-text model (override with `TRANSCRIBE_MODEL`); cloud providers are listed before Ollama, and
  "AI proofreading" is ticked by default when the default proofreading model is a cloud one.
- **Free Ollama RAM from the UI**: models Ollama keeps loaded are shown under the top bar with their
  size and a ⏹ stop button (or stop all), which unloads them right away (`/api/ollama/loaded`, `/api/ollama/unload`).
- **Speech-to-text via ElevenLabs Scribe** (`ELEVENLABS_API_KEY`): `elevenlabs:scribe_v2` appears in
  the model list; word timestamps are converted to the Whisper shape so segmenting, editing and
  rendering are unchanged. Hint words are sent as `keyterms` (`ELEVENLABS_KEYTERMS=0` to disable).
- **Proofreading via API key** so the NAS doesn't have to run an LLM: Z.AI GLM, Typhoon API,
  OpenRouter, Gemini or any OpenAI-compatible endpoint (`ZAI_API_KEY`, `TYPHOON_API_KEY`, ...).
  Models are referenced as `provider:model` (e.g. `zai:glm-4.7-flash`); plain names stay Ollama.
  `PROOFREAD_MODEL` sets the default. Dropdowns group models by provider; the comparison page can
  pit cloud models against local ones.
- **🧪 Model comparison page** (tab "ทดสอบโมเดล"): pick up to 8 Ollama models and run the
  same proofreading job on each, one node per model. Built-in Thai test set with an answer key
  (names, tone marks, ร/ล, homophones, plus slang that must not be "fixed") gives each model a
  score, fixed/over-edit counts, time, tokens/s, model load time and a "thinking" flag; or use
  the subtitles of a real job to compare suggestions side by side. Past runs are kept in a history list.
- Live **node progress** for every task: each step (extract audio, Whisper, segmenting,
  Ollama, .ass, iGPU check, render, save) shows its state, device (CPU / iGPU / Ollama),
  detail, per-step progress and a live elapsed timer; render fallback from iGPU to CPU is
  shown on the node.
- Ollama failures are shown as a warning node; the transcript is kept and the job still finishes.
- `CLAUDE.md` with project context and rules for AI-assisted development.

- Z.AI now uses the GLM Coding Plan endpoint (`https://api.z.ai/api/coding/paas/v4`) by default;
  set `ZAI_BASE_URL=https://api.z.ai/api/paas/v4` for a pay-as-you-go key. GLM 5.3 models, which
  always think, get `reasoning_effort` from `ZAI_REASONING_EFFORT` (default `max`); other models
  that refuse `thinking: disabled` are retried the same way. Default Z.AI model is now `glm-5.3`
  (100 on the built-in test set; `glm-5.3-flash` 50–88).
### Fixed
- Line breaks no longer split a word when PyThaiNLP tokenizes it oddly (e.g. ตั้ง|ใจมา): breaks
  where a dictionary word spans the gap are avoided.
- ElevenLabs transcripts no longer lose real lines such as "อย่าลืมกดติดตาม…": the Whisper
  outro-hallucination filter is skipped for Scribe segments.

- Ollama answers that drop or merge items are retried once (`proofreader.ask_validated`),
  in both proofreading and the comparison page; failed raw answers are kept in bench results.
- The comparison page unloads each model right after testing it (`keep_alive: 0`), so several
  models are never in memory at once (a 12B model was OOM-killed without this).
- Comparison score now also subtracts "wrong edits" (an error changed into a different error).

### Changed
- `proofreader.ask()` returns Ollama timing stats; `proofread()` builds on it.
- Job API returns `task.steps` and the server time (`now`) so the phone can show accurate timers.

## [0.1.0] - 2026-10-25

### Added
- Mobile-first web UI (FastAPI + vanilla JS): upload, hint words, live subtitle preview on the
  video, cue editor (text, timing, add/delete), style picker (font, size, colours, outline,
  box, bottom margin), render options (pad/crop, sharpen, denoise), download Master MP4.
- Speech-to-text with whisper.cpp server built for **Vulkan** (Intel iGPU), default model
  `ggml-large-v3-q5_0`, Silero VAD, models auto-downloaded to `./models`; switch models from the UI.
- Thai segmentation with PyThaiNLP: 1–2 line cues that never break inside a word or a hint
  name; filter for Whisper hallucinations over music/silence; token times remapped for VAD.
- Optional Ollama proofreading with JSON-schema output, validated and offered as suggestions.
- FFmpeg rendering on the iGPU via VAAPI (decode, scale, denoise, sharpen, H.264 encode),
  rotation and HDR handling, capability probe and automatic libx264 fallback.
- Output for Facebook Reels / TikTok: 1080×1920, H.264 High, 30 fps, 12–15 Mbps, AAC 192k.
- Docker Compose project for Synology DSM Container Manager; Kanit, Prompt and Sarabun fonts.
- Test suite including a real CPU render and the full API flow.
