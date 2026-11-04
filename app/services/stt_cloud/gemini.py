"""Cloud speech-to-text provider: Google Gemini 3.5 Transcribe (Interactions API).

Uses the same AI Studio key as the proofreading LLM (GEMINI_API_KEY). Word
timestamps are required for cue timing but the API rejects them together with
custom vocabulary, so hint words can't bias this model (see config.py).
Free tier: 10 requests/day per model, 3 requests/minute.
"""

import base64

import httpx

from app import config
from app.services.stt_cloud import _common

PREFIX = "gemini:"
_SOURCE = "gemini"

# Same AI Studio key as the proofreading LLM (llm.py reads it through env too;
# llm.py's GEMINI_BASE_URL is a different thing — the OpenAI-style endpoint).
def _key() -> str:
    return config.env("GEMINI_API_KEY", "")


def enabled() -> bool:
    return bool(_key())


def models() -> list[str]:
    return [PREFIX + config.GEMINI_STT_MODEL] if enabled() else []


def is_cloud(model: str | None) -> bool:
    return bool(model) and model.startswith(PREFIX)


def transcribe(wav_path, hint: str = "", model: str | None = None) -> dict:
    if not enabled():
        raise RuntimeError("ยังไม่ได้ใส่ GEMINI_API_KEY")
    model_id = (model or "").removeprefix(PREFIX) or config.GEMINI_STT_MODEL
    languages = [x.strip() for x in config.GEMINI_STT_LANGUAGES.split(",") if x.strip()]
    body = {
        "model": f"models/{model_id}",
        "store": False,
        "input": [{
            "type": "audio",
            "data": base64.b64encode(wav_path.read_bytes()).decode(),
            "mime_type": "audio/wav",
        }],
        "generation_config": {
            "transcription_config": {
                "language_codes": languages,
                "mode": {"type": "verbatim", "timestamp_granularities": ["word"]},
            },
        },
    }
    r = httpx.post(
        f"{config.GEMINI_STT_BASE_URL}/interactions",
        headers={"x-goog-api-key": _key()},
        json=body,
        timeout=config.WHISPER_TIMEOUT,
    )
    if r.status_code == 429:
        raise RuntimeError("Gemini เกินโควตาฟรี (10 ครั้ง/วันต่อโมเดล, 3 ครั้ง/นาที) — รอสักครู่หรือเปลี่ยนโมเดลถอดเสียง")
    if r.status_code >= 400:
        raise RuntimeError(f"Gemini ตอบ {r.status_code}: {r.text[:300]}")
    data = r.json()
    if data.get("status") == "failed":
        raise RuntimeError(f"Gemini ถอดเสียงไม่สำเร็จ: {data.get('errors')}")
    # The transcript text and the word annotations live on the model-output steps.
    words, text = [], []
    for step in data.get("steps") or []:
        for content in step.get("content") or []:
            if content.get("type") != "text":
                continue
            text.append(content.get("text") or "")
            for ann in content.get("annotations") or []:
                if ann.get("type") != "word_info":
                    continue
                start = _common.parse_offset(ann.get("start_offset"))
                end = _common.parse_offset(ann.get("end_offset"))
                if ann.get("text") and start is not None and end is not None:
                    words.append({"word": ann["text"], "start": start, "end": end})
    segments = _common.words_to_segments(words, _SOURCE, gap_split=1.0)
    return {"segments": segments, "text": "".join(text), "language": languages[0] if languages else "th"}
