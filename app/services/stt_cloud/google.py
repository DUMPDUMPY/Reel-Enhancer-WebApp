"""Cloud speech-to-text provider: Google Cloud Speech-to-Text v2 (Chirp).

Two auth options, in order of setup effort (per the API docs):
- API key (`AIzaSy…` made in the GCP console): works on the global host
  speech.googleapis.com. The project needs billing enabled even for the free
  tier; the regional hosts (us-speech.googleapis.com …) have been seen live to
  answer 401 "API keys are not supported by this API", so prefer global.
- Service account JSON key via GOOGLE_CLOUD_CREDENTIALS — always accepted;
  minted into a Bearer access token (JWT-bearer) and cached ~1 h.

An AI Studio key (`AQ.…`) only works on generativelanguage.googleapis.com — it
answers 401 on every Cloud API. Chirp 3 answers 404 outside the locations that
serve it (us / eu / asia-* / global); free tier is 60 minutes/month and
timestamped files are limited to 20 min. The synchronous Recognize call takes
inline audio (≤1 min), good for this app's short clips.
"""

import base64
import json
import threading
import time
from pathlib import Path

import httpx

from app import config
from app.services.stt_cloud import _common

PREFIX = "google:"
_SOURCE = "google"
# Access tokens last ~1 h; reuse across clips instead of minting one per request.
_token: tuple[str, float] | None = None  # (access_token, exp)
_lock = threading.Lock()


def enabled() -> bool:
    return bool(config.GOOGLE_CLOUD_PROJECT and (config.GOOGLE_CLOUD_CREDENTIALS or config.GOOGLE_CLOUD_API_KEY))


def models() -> list[str]:
    return [PREFIX + m for m in config.GOOGLE_CLOUD_STT_MODELS] if enabled() else []


def is_cloud(model: str | None) -> bool:
    return bool(model) and model.startswith(PREFIX)


def _auth_headers() -> dict[str, str]:
    """Service account first (always accepted); fall back to the API key."""
    if config.GOOGLE_CLOUD_CREDENTIALS:
        return {"Authorization": f"Bearer {_access_token()}"}
    return {"x-goog-api-key": config.GOOGLE_CLOUD_API_KEY}


def _access_token() -> str:
    """Mint an OAuth2 access token from the service-account JSON (JWT-bearer)."""
    global _token
    with _lock:
        if _token and _token[1] > time.time() + 60:
            return _token[0]
        creds = json.loads(Path(config.GOOGLE_CLOUD_CREDENTIALS).read_text())
        try:
            import jwt

            assertion = jwt.encode(
                {
                    "iss": creds["client_email"],
                    "scope": "https://www.googleapis.com/auth/cloud-platform",
                    "aud": "https://oauth2.googleapis.com/token",
                    "exp": int(time.time()) + 3600,
                    "iat": int(time.time()),
                },
                creds["private_key"],
                algorithm="RS256",
                headers={"kid": creds.get("private_key_id")},
            )
        except (ImportError, KeyError, ValueError) as e:
            raise RuntimeError(
                f"อ่าน GOOGLE_CLOUD_CREDENTIALS ไม่ได้ ({e}) — ต้องเป็นไฟล์ JSON ของ "
                "service account (มี client_email และ private_key)"
            ) from e
        r = httpx.post(
            "https://oauth2.googleapis.com/token",
            data={"grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer", "assertion": assertion},
            timeout=config.WHISPER_TIMEOUT,
        )
        if r.status_code >= 400:
            raise RuntimeError(f"แลก access token ไม่สำเร็จ ({r.status_code}): {r.text[:300]}")
        access = r.json().get("access_token")
        if not access:
            raise RuntimeError("Google ไม่ส่ง access_token กลับมา")
        _token = (access, int(r.json().get("expires_in", 3600)) + time.time())
        return _token[0]


def _word_time(w: dict, key: str) -> float | None:
    """{seconds, nanos} JSON mapping -> seconds; handles camelCase/snake_case keys."""
    t = w.get(key) or w.get(key.replace("Offset", "Time"))
    if not isinstance(t, dict):
        return None
    return float(t.get("seconds", 0)) + float(t.get("nanos", 0)) / 1e9


def transcribe(wav_path, hint: str = "", model: str | None = None) -> dict:
    if not enabled():
        raise RuntimeError(
            "ยังไม่ตั้ง GOOGLE_CLOUD_API_KEY (คีย์ AIzaSy + ผูก billing) หรือ "
            "GOOGLE_CLOUD_CREDENTIALS (ไฟล์ JSON ของ service account) พร้อม GOOGLE_CLOUD_PROJECT"
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
            "language_codes": ["th-TH"],
            "auto_decoding_config": {},  # WAV/FLAC/MP3/OGG are detected for us
            "features": {"enable_word_time_offsets": True},
        },
        "content": base64.b64encode(wav_path.read_bytes()).decode(),
    }
    r = httpx.post(url, headers=_auth_headers(), json=body, timeout=config.WHISPER_TIMEOUT)
    if r.status_code == 401:
        raise RuntimeError(
            "Google Cloud STT ตอบ 401 — ถ้าใช้ GOOGLE_CLOUD_API_KEY ลอง: ตั้ง "
            f"GOOGLE_CLOUD_LOCATION=global (host speech.googleapis.com), ผูก billing "
            "กับ project (จำเป็นแม้โควตาฟรี), เปิดใช้ Cloud Speech-to-Text API หรือ "
            "เปลี่ยนไปใช้ service account (GOOGLE_CLOUD_CREDENTIALS) — คีย์ AI Studio "
            f"แบบ AQ. ใช้กับ Cloud API ไม่ได้เสมอ · ข้อความจาก Google: {r.text[:300]}"
        )
    if r.status_code == 404:
        raise RuntimeError(
            f"Google Cloud STT ตอบ 404 — โมเดล {model_id} อาจไม่มีใน location '{location}' "
            f"(chirp_3 อยู่ใน global / us / eu / asia-*) หรือ Cloud Speech-to-Text API ยังไม่ได้เปิดใช้"
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
    return {"segments": segments, "text": "".join(chunks), "language": "th-TH"}
