import os
import tempfile
from pathlib import Path

_tmp = Path(tempfile.mkdtemp(prefix="reel-test-"))
os.environ.setdefault("DATA_DIR", str(_tmp / "data"))
os.environ.setdefault("MODELS_DIR", str(_tmp / "models"))
os.environ.setdefault("RENDER_MODE", "cpu")
os.environ.setdefault("OLLAMA_HOST", "http://127.0.0.1:9")
os.environ.setdefault("WHISPER_URL", "http://127.0.0.1:9")
