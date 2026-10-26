import os
import tempfile
from pathlib import Path

_tmp = Path(tempfile.mkdtemp(prefix="reel-test-"))
os.environ.setdefault("DATA_DIR", str(_tmp / "data"))
os.environ.setdefault("MODELS_DIR", str(_tmp / "models"))
os.environ.setdefault("RENDER_MODE", "cpu")
os.environ.setdefault("OLLAMA_HOST", "http://127.0.0.1:9")
os.environ.setdefault("WHISPER_URL", "http://127.0.0.1:9")
# Real keys in the shell must not make tests call paid APIs (cloud is the default when a key is set).
for _k in ("ELEVENLABS_API_KEY", "ZAI_API_KEY", "TYPHOON_API_KEY", "OPENROUTER_API_KEY", "GEMINI_API_KEY",
           "CUSTOM_LLM_API_KEY", "PROOFREAD_MODEL", "STT_MODEL"):
    os.environ[_k] = ""
