"""Client for the whisper.cpp server (Vulkan build running on the Intel iGPU)."""

import threading
from pathlib import Path

import httpx

from app import config

_loaded_model: str | None = None
_lock = threading.Lock()


def list_models() -> list[str]:
    if not config.MODELS_DIR.is_dir():
        return []
    return sorted(p.name for p in config.MODELS_DIR.glob("ggml-*.bin") if "silero" not in p.name and "vad" not in p.name)


def current_model() -> str:
    return _loaded_model or config.WHISPER_DEFAULT_MODEL


def health() -> dict:
    try:
        r = httpx.get(f"{config.WHISPER_URL}/health", timeout=3)
        return {"ok": r.status_code == 200, "status": r.json().get("status", r.text) if r.content else r.status_code}
    except (httpx.HTTPError, ValueError) as e:
        return {"ok": False, "status": str(e)}


def _ensure_model(name: str | None) -> None:
    """Switch the server's model via /load if the user picked a different one."""
    global _loaded_model
    if not name or name == current_model():
        return
    if name not in list_models():
        raise ValueError(f"ไม่พบโมเดล {name} ในโฟลเดอร์ models")
    r = httpx.post(
        f"{config.WHISPER_URL}/load",
        files={"model": (None, str(config.MODELS_DIR / name))},
        timeout=config.WHISPER_TIMEOUT,
    )
    r.raise_for_status()
    _loaded_model = name


def build_prompt(hint: str) -> str:
    """Initial prompt from the user's hint. Whisper only reads ~224 tokens, and Thai
    is token-hungry, so keep it short."""
    hint = " ".join((hint or "").split())
    return hint[:200]


def transcribe(wav_path: Path, hint: str = "", model: str | None = None, vad: bool = True) -> dict:
    """Run whisper on a 16 kHz mono WAV and return the verbose_json result."""
    with _lock:
        _ensure_model(model)
        data = {
            "language": "th",
            "response_format": "verbose_json",
            "temperature": "0.0",
            "temperature_inc": "0.2",
            "suppress_non_speech": "true",
            "vad": "true" if vad else "false",
        }
        prompt = build_prompt(hint)
        if prompt:
            data["prompt"] = prompt
        with wav_path.open("rb") as f:
            r = httpx.post(
                f"{config.WHISPER_URL}/inference",
                data=data,
                files={"file": (wav_path.name, f, "audio/wav")},
                timeout=config.WHISPER_TIMEOUT,
            )
        r.raise_for_status()
        result = r.json()
        if "error" in result:
            raise RuntimeError(result["error"])
        return result
