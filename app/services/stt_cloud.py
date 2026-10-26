"""Cloud speech-to-text (ElevenLabs Scribe), converted to whisper's verbose_json shape.

Lets the NAS skip running Whisper when an API key is set. The rest of the
pipeline (segmenter, editor, render) doesn't care where the words came from.
"""

import re

import httpx

from app import config
from app.services.thai_text import parse_hint_words

PREFIX = "elevenlabs:"
_SENTENCE_END = re.compile(r"[.!?…。！？]$")


def enabled() -> bool:
    return bool(config.ELEVENLABS_API_KEY)


def models() -> list[str]:
    return [PREFIX + config.ELEVENLABS_STT_MODEL] if enabled() else []


def is_cloud(model: str | None) -> bool:
    return bool(model) and model.startswith(PREFIX)


def default_model() -> str | None:
    """TRANSCRIBE_MODEL, else Scribe when a key is set; None means Whisper's current model."""
    m = config.TRANSCRIBE_MODEL
    if m and not (is_cloud(m) and not enabled()):  # a cloud default without its key falls back to Whisper
        return m
    return models()[0] if enabled() else None


def _keyterms(hint: str) -> list[str]:
    # Scribe limits: <50 chars and at most 5 words per term; these characters are rejected.
    terms = [t for t in parse_hint_words(hint) if len(t) < 50 and not re.search(r"[<>{}\[\]\\]", t)]
    return terms[:100]  # above 100 terms a 20 s minimum is billed per request


def to_whisper_json(data: dict, pause_split: float = config.PAUSE_SPLIT_SECONDS) -> dict:
    """Scribe words[] -> {"segments": [{text, start, end, words: [{word, start, end}]}]}.

    Segments break at sound events (music, laughter), long pauses and sentence ends,
    which is what the segmenter uses as natural cue boundaries.
    """
    segments, cur = [], []

    def close():
        nonlocal cur
        toks = [w for w in cur if w["word"].strip()]
        if toks:
            segments.append({
                "text": "".join(w["word"] for w in cur).strip(),
                "start": toks[0]["start"], "end": toks[-1]["end"], "words": cur,
            })
        cur = []

    last_end = None
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
            continue
        if last_end is not None and float(start) - last_end >= pause_split:
            close()
        cur.append({"word": text, "start": float(start), "end": float(end)})
        last_end = float(end)
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
