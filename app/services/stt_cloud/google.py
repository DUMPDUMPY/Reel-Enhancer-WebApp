"""Cloud speech-to-text provider: Google Cloud Speech-to-Text.

Two auth modes, picked automatically per the credentials configured:
- Service account JSON key (GOOGLE_CLOUD_CREDENTIALS) → v2 API, Chirp models
  (chirp_3: best accuracy, diarization, auto language detect). The key is minted
  into a Bearer access token (JWT-bearer) and cached ~1 h.
- Cloud API key (AIzaSy… via GOOGLE_CLOUD_API_KEY) → v1 speech:recognize with
  the latest_long model. Verified live: v1 accepts API keys (200 OK) while v2
  answers 401 "API keys are not supported by this API" on every host/param
  form — Chirp is v2-only, so key users get latest_long instead.

Both modes share the word-offset parsing (v2 uses {seconds, nanos} objects,
v1 "1.400s" strings) and feed the same segmenter. An AI Studio key (`AQ.…`)
only works on generativelanguage.googleapis.com — 401 on every Cloud API.
Free tier: 60 minutes/month (billing must be bound even for the free quota).
The synchronous calls take inline audio (≤1 min), good for this app's clips.
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
    if not enabled():
        return []
    if config.GOOGLE_CLOUD_CREDENTIALS:  # v2: Chirp
        return [PREFIX + m for m in config.GOOGLE_CLOUD_STT_MODELS]
    return [PREFIX + config.GOOGLE_CLOUD_STT_V1_MODEL]  # key-only: v1


def is_cloud(model: str | None) -> bool:
    return bool(model) and model.startswith(PREFIX)


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
    """v2 sends {seconds, nanos} objects, v1 sends "1.400s" strings."""
    t = w.get(key) or w.get(key.replace("Offset", "Time"))
    if isinstance(t, str):
        return _common.parse_offset(t)
    if isinstance(t, dict):
        return float(t.get("seconds", 0)) + float(t.get("nanos", 0)) / 1e9
    return None


def _parse_results(data: dict) -> tuple[list[dict], list[str]]:
    """results[].alternatives[0] words are absolute offsets — no chunk shifting."""
    words: list[dict] = []
    chunks: list[str] = []
    for result in data.get("results") or []:
        alt = (result.get("alternatives") or [{}])[0]
        chunks.append(alt.get("transcript", ""))
        for w in alt.get("words") or []:
            start, end = _word_time(w, "startOffset"), _word_time(w, "endOffset")
            # v1 latest_long leaks sentencepiece "▁" markers into word strings.
            text = (w.get("word") or w.get("text") or "").replace("▁", "")
            if text and start is not None and end is not None:
                words.append({"word": text, "start": start, "end": end})
    return words, chunks


def transcribe(wav_path, hint: str = "", model: str | None = None) -> dict:
    if not enabled():
        raise RuntimeError(
            "ยังไม่ตั้ง GOOGLE_CLOUD_API_KEY (คีย์ AIzaSy) หรือ GOOGLE_CLOUD_CREDENTIALS "
            "(ไฟล์ JSON ของ service account) พร้อม GOOGLE_CLOUD_PROJECT"
        )
    if config.GOOGLE_CLOUD_CREDENTIALS:
        return _transcribe_v2(wav_path, model)
    return _transcribe_v1(wav_path, model)


def _transcribe_v2(wav_path, model: str | None) -> dict:
    """Service account → v2 Chirp. The implicit recognizer (name "_") lives in
    the URL path; "global" uses the plain host, other locations prefix it."""
    model_id = (model or "").removeprefix(PREFIX) or config.GOOGLE_CLOUD_STT_MODELS[0]
    project, location = config.GOOGLE_CLOUD_PROJECT, config.GOOGLE_CLOUD_LOCATION
    host = "speech.googleapis.com" if location == "global" else f"{location}-speech.googleapis.com"
    r = httpx.post(
        f"https://{host}/v2/projects/{project}/locations/{location}/recognizers/_:recognize",
        headers={"Authorization": f"Bearer {_access_token()}"},
        json={
            "config": {
                "model": model_id,
                "language_codes": ["th-TH"],
                "auto_decoding_config": {},
                "features": {"enable_word_time_offsets": True},
            },
            "content": base64.b64encode(wav_path.read_bytes()).decode(),
        },
        timeout=config.WHISPER_TIMEOUT,
    )
    if r.status_code == 401:
        raise RuntimeError(
            "Google Cloud STT ตอบ 401 — ตรวจไฟล์ service account (GOOGLE_CLOUD_CREDENTIALS) "
            f"และ Cloud Speech-to-Text API ใน project · ข้อความจาก Google: {r.text[:300]}"
        )
    if r.status_code == 404:
        raise RuntimeError(
            f"Google Cloud STT ตอบ 404 — โมเดล {model_id} อาจไม่มีใน location '{location}' "
            f"(chirp_3 อยู่ใน global / us / eu / asia-*) หรือ Cloud Speech-to-Text API ยังไม่ได้เปิดใช้"
        )
    if r.status_code >= 400:
        raise RuntimeError(f"Google Cloud STT ตอบ {r.status_code}: {r.text[:300]}")
    words, chunks = _parse_results(r.json())
    segments = _common.words_to_segments(words, _SOURCE, gap_split=1.0)
    return {"segments": segments, "text": "".join(chunks), "language": "th-TH"}


def _transcribe_v1(wav_path, model: str | None) -> dict:
    """API key → v1 speech:recognize (keys are rejected by v2, live-verified)."""
    model_id = (model or "").removeprefix(PREFIX) or config.GOOGLE_CLOUD_STT_V1_MODEL
    r = httpx.post(
        f"https://speech.googleapis.com/v1/speech:recognize?key={config.GOOGLE_CLOUD_API_KEY}",
        json={
            "config": {
                "languageCode": "th-TH",
                "model": model_id,  # default / latest_long / latest_short
                "enableWordTimeOffsets": True,
            },
            # WAV headers carry encoding + sample rate, so no explicit encoding needed.
            "audio": {"content": base64.b64encode(wav_path.read_bytes()).decode()},
        },
        timeout=config.WHISPER_TIMEOUT,
    )
    if r.status_code in (401, 403):
        raise RuntimeError(
            f"Google Cloud STT ตอบ {r.status_code} — เช็คว่าคีย์เป็นชนิด AIzaSy (ไม่ใช่ AI Studio "
            "แบบ AQ.), เปิดใช้ Cloud Speech-to-Text API และผูก billing กับ project แล้ว "
            f"· ข้อความจาก Google: {r.text[:300]}"
        )
    if r.status_code >= 400:
        raise RuntimeError(f"Google Cloud STT ตอบ {r.status_code}: {r.text[:300]}")
    words, chunks = _parse_results(r.json())
    segments = _common.words_to_segments(words, _SOURCE, gap_split=1.0)
    return {"segments": segments, "text": "".join(chunks), "language": "th-TH"}
