"""Runtime settings, read from environment variables (see docker-compose.yml)."""

import os
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


env = _env  # public alias for modules that read their own variables (services/llm.py)


def _env_int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(_env("DATA_DIR", "/data"))
JOBS_DIR = DATA_DIR / "jobs"
# Fonts shipped with the image plus any extra .ttf/.otf the user drops into /data/fonts.
FONT_DIRS = [Path(_env("FONTS_DIR", str(APP_DIR.parent / "fonts"))), DATA_DIR / "fonts"]

# whisper.cpp server (Vulkan build, runs on the Intel iGPU).
WHISPER_URL = _env("WHISPER_URL", "http://whisper:8080")
# Directory holding ggml-*.bin models; mounted at the same path in both containers.
MODELS_DIR = Path(_env("MODELS_DIR", "/models"))
WHISPER_DEFAULT_MODEL = _env("WHISPER_MODEL", "ggml-large-v3-q5_0.bin")
WHISPER_TIMEOUT = _env_float("WHISPER_TIMEOUT", 600)

# Proofreading LLM (optional). Local Ollama, or a cloud API with a key (see services/llm.py:
# ZAI_API_KEY, TYPHOON_API_KEY, OPENROUTER_API_KEY, GEMINI_API_KEY, CUSTOM_LLM_*).
# Empty OLLAMA_HOST disables Ollama.
OLLAMA_HOST = _env("OLLAMA_HOST", "http://host.docker.internal:11434").rstrip("/")
OLLAMA_MODEL = _env("OLLAMA_MODEL", "gemma3:4b")
# Default model for proofreading, e.g. "zai:glm-4.7-flash". Empty: first cloud provider with a key, else Ollama.
PROOFREAD_MODEL = _env("PROOFREAD_MODEL", "")
OLLAMA_TIMEOUT = _env_float("OLLAMA_TIMEOUT", 120)  # also used for cloud APIs

# Rendering. RENDER_MODE: auto | vaapi | cpu
RENDER_MODE = _env("RENDER_MODE", "auto")
VAAPI_DEVICE = _env("VAAPI_DEVICE", "/dev/dri/renderD128")
OUTPUT_WIDTH = _env_int("OUTPUT_WIDTH", 1080)
OUTPUT_HEIGHT = _env_int("OUTPUT_HEIGHT", 1920)
OUTPUT_FPS = _env_int("OUTPUT_FPS", 30)
VIDEO_BITRATE = _env("VIDEO_BITRATE", "12M")
VIDEO_MAXRATE = _env("VIDEO_MAXRATE", "15M")
VIDEO_BUFSIZE = _env("VIDEO_BUFSIZE", "20M")
X264_CRF = _env_int("X264_CRF", 18)
X264_PRESET = _env("X264_PRESET", "medium")
# CPU threads FFmpeg may use when it has to fall back to software encoding.
CPU_THREADS = _env_int("CPU_THREADS", 4)

# Subtitle segmentation (display width counts base glyphs, not Thai marks above/below).
MAX_LINE_WIDTH = _env_int("MAX_LINE_WIDTH", 20)
MAX_LINES = _env_int("MAX_LINES", 2)
MIN_CUE_SECONDS = _env_float("MIN_CUE_SECONDS", 1.0)
MAX_CUE_SECONDS = _env_float("MAX_CUE_SECONDS", 4.0)
PAUSE_SPLIT_SECONDS = _env_float("PAUSE_SPLIT_SECONDS", 0.45)

MAX_UPLOAD_MB = _env_int("MAX_UPLOAD_MB", 1024)
JOB_RETENTION_HOURS = _env_int("JOB_RETENTION_HOURS", 72)
