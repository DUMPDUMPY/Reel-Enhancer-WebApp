"""Cloud speech-to-text, converted to whisper's verbose_json shape.

Lets the NAS skip running Whisper when an API key is set. The rest of the
pipeline (segmenter, editor, render) doesn't care where the words came from.

Providers are picked by the model's prefix, like app/services/llm.py does:
"elevenlabs:…", "gemini:…" or "google:…". A plain model id means the local
Whisper server.
"""

from pathlib import Path

from app import config
from app.services.stt_cloud import elevenlabs, gemini, google

_PROVIDERS = [elevenlabs, gemini, google]


def _pick(model: str | None):
    """The provider module owning this model id, or None for Whisper."""
    for p in _PROVIDERS:
        if p.is_cloud(model):
            return p
    return None


def _models_for_default() -> list[str]:
    """Cloud models in default order: ElevenLabs first, then Gemini, then Google."""
    out: list[str] = []
    for p in _PROVIDERS:
        out.extend(p.models())
    return out


def enabled() -> bool:
    """Any cloud provider usable right now?"""
    return any(p.enabled() for p in _PROVIDERS)


def is_cloud(model: str | None) -> bool:
    return any(p.is_cloud(model) for p in _PROVIDERS)


def default_model() -> str | None:
    """TRANSCRIBE_MODEL, else the first cloud model with a key; None means Whisper's
    current model. A cloud default whose key is gone falls back to Whisper."""
    m = config.TRANSCRIBE_MODEL
    if m:
        if not m.startswith(tuple(p.PREFIX for p in _PROVIDERS)):
            return m  # a Whisper model
        p = _pick(m)
        if p and p.enabled():
            return m
        return None  # cloud default without its key falls back to Whisper
    models = _models_for_default()
    return models[0] if models else None


def models() -> list[str]:
    """Every cloud model selectable in the UI (grouped by provider upstream)."""
    return _models_for_default()


def transcribe(wav_path: Path, hint: str = "", model: str | None = None) -> dict:
    p = _pick(model)
    if p is None:
        raise RuntimeError(f"ไม่รู้จักโมเดลถอดเสียงบนคลาวด์: {model}")
    return p.transcribe(wav_path, hint=hint, model=model)
