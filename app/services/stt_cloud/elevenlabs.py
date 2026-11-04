"""Cloud speech-to-text provider: ElevenLabs Scribe, converted to whisper's shape."""

import re

import httpx

from app import config
from app.services.stt_cloud import _common
from app.services.thai_text import parse_hint_words

PREFIX = "elevenlabs:"
_SENTENCE_END = re.compile(r"[.!?…。！？]$")


def enabled() -> bool:
    return bool(config.ELEVENLABS_API_KEY)


def models() -> list[str]:
    return [PREFIX + config.ELEVENLABS_STT_MODEL] if enabled() else []


def is_cloud(model: str | None) -> bool:
    return bool(model) and model.startswith(PREFIX)


def _keyterms(hint: str) -> list[str]:
    # Scribe limits: <50 chars and at most 5 words per term; these characters are rejected.
    terms = [t for t in parse_hint_words(hint) if len(t) < 50 and not re.search(r"[<>{}\[\]\\]", t)]
    return terms[:100]  # above 100 terms a 20 s minimum is billed per request


def to_whisper_json(data: dict, pause_split: float = config.PAUSE_SPLIT_SECONDS) -> dict:
    """Scribe words[] -> {"segments": [{text, start, end, words: [{word, start, end}]}]}.

    For Thai, Scribe's "words" are single characters or clusters ("น", "ที", "ห้"),
    so a pause between two of them is often inside a word (a held note in a song).
    Segments therefore break only at sound events, sentence ends and long *spacing*
    (a real word boundary); pauses between words are left to the segmenter, which
    sees whole PyThaiNLP words.
    """
    segments, cur = [], []

    def close():
        nonlocal cur
        seg = _common.make_segment(cur, "elevenlabs")
        if seg:
            segments.append(seg)
        cur = []

    for w in data.get("words") or []:
        kind, text = w.get("type"), w.get("text", "")
        start, end = w.get("start"), w.get("end")
        if kind == "audio_event":
            close()
            continue
        if start is None or end is None:
            continue
        if kind == "spacing":
            cur.append({"word": " ", "start": float(start), "end": float(end)})
            if float(end) - float(start) >= pause_split:
                close()
            continue
        cur.append({"word": text, "start": float(start), "end": float(end)})
        if _SENTENCE_END.search(text):
            close()
    close()
    return {"segments": segments, "text": data.get("text", ""), "language": data.get("language_code")}


def transcribe(wav_path, hint: str = "", model: str | None = None) -> dict:
    if not enabled():
        raise RuntimeError("ยังไม่ได้ใส่ ELEVENLABS_API_KEY")
    model_id = (model or "").removeprefix(PREFIX) or config.ELEVENLABS_STT_MODEL
    data: dict = {
        "model_id": model_id,
        "language_code": "th",
        "timestamps_granularity": "word",
        "tag_audio_events": "true",
        "diarize": "false",
    }
    terms = _keyterms(hint) if config.ELEVENLABS_KEYTERMS else []
    if terms:
        data["keyterms"] = terms  # sent as repeated form fields
    with open(wav_path, "rb") as f:
        r = httpx.post(
            f"{config.ELEVENLABS_BASE_URL}/v1/speech-to-text",
            headers={"xi-api-key": config.ELEVENLABS_API_KEY},
            data=data,
            files={"file": (wav_path.name, f, "audio/wav")},
            timeout=config.WHISPER_TIMEOUT,
        )
    if r.status_code >= 400:
        raise RuntimeError(f"ElevenLabs ตอบ {r.status_code}: {r.text[:300]}")
    return to_whisper_json(r.json())
