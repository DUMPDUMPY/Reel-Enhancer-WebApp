"""Cloud speech-to-text provider: Google Cloud Speech-to-Text v2 (Chirp 3).

Needs a *Cloud* API key (`AIzaSy…`, created in the GCP console with the Cloud
Speech-to-Text API enabled) — an AI Studio key (`AQ.…`) answers 401. Free tier:
60 minutes/month. The synchronous Recognize call takes inline audio, good for
the short clips this app handles (its content limit is 1 minute).
"""

import base64

import httpx

from app import config
from app.services.stt_cloud import _common

PREFIX = "google:"
_SOURCE = "google"


def enabled() -> bool:
    return bool(config.GOOGLE_CLOUD_API_KEY and config.GOOGLE_CLOUD_PROJECT)


def models() -> list[str]:
    return [PREFIX + m for m in config.GOOGLE_CLOUD_STT_MODELS] if enabled() else []


def is_cloud(model: str | None) -> bool:
    return bool(model) and model.startswith(PREFIX)


def _word_time(w: dict, key: str) -> float | None:
    """{seconds, nanos} JSON mapping -> seconds; handles camelCase/snakeCase keys."""
    t = w.get(key) or w.get(key.replace("Offset", "Time"))
    if not isinstance(t, dict):
        return None
    return float(t.get("seconds", 0)) + float(t.get("nanos", 0)) / 1e9


def transcribe(wav_path, hint: str = "", model: str | None = None) -> dict:
    if not enabled():
        raise RuntimeError(
            "ยังไม่ได้ใส่ GOOGLE_CLOUD_API_KEY / GOOGLE_CLOUD_PROJECT "
            "(ต้องเป็นคีย์ Cloud แบบ AIzaSy ไม่ใช่คีย์ AI Studio แบบ AQ.)"
        )
    model_id = (model or "").removeprefix(PREFIX) or config.GOOGLE_CLOUD_STT_MODELS[0]
    project, location = config.GOOGLE_CLOUD_PROJECT, config.GOOGLE_CLOUD_LOCATION
    # The implicit recognizer (name "_") lives in the URL path, not the body.
    # "global" uses the plain host; every other location prefixes its own host.
    host = "speech.googleapis.com" if location == "global" else f"{location}-speech.googleapis.com"
    url = (
        f"https://{host}/v2/"
        f"projects/{project}/locations/{location}/recognizers/_:recognize"
    )
    body = {
        "config": {
            "model": model_id,
            "language_codes": ["th"],
            "auto_decoding_config": {},  # WAV/FLAC/MP3/OGG are detected for us
            "features": {"enable_word_time_offsets": True},
        },
        "content": base64.b64encode(wav_path.read_bytes()).decode(),
    }
    r = httpx.post(
        url,
        headers={"x-goog-api-key": config.GOOGLE_CLOUD_API_KEY},
        json=body,
        timeout=config.WHISPER_TIMEOUT,
    )
    if r.status_code >= 400:
        raise RuntimeError(f"Google Cloud STT ตอบ {r.status_code}: {r.text[:300]}")
    # results[] holds one entry per recognized chunk; word offsets are absolute
    # for Recognize (audio starts at 0), so no chunk shifting is needed.
    words: list[dict] = []
    chunks: list[str] = []
    for result in r.json().get("results") or []:
        alt = (result.get("alternatives") or [{}])[0]
        chunks.append(alt.get("transcript", ""))
        for w in alt.get("words") or []:
            start, end = _word_time(w, "startOffset"), _word_time(w, "endOffset")
            text = w.get("word") or w.get("text") or ""
            if text and start is not None and end is not None:
                words.append({"word": text, "start": start, "end": end})
    segments = _common.words_to_segments(words, _SOURCE, gap_split=1.0)
    return {"segments": segments, "text": "".join(chunks), "language": "th"}
