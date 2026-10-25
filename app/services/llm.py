"""LLM providers for proofreading: local Ollama or any OpenAI-compatible API with a key.

A model is referenced as "<provider>:<model>", e.g. "zai:glm-4.7-flash" or
"typhoon:typhoon-v2.5-30b-a3b-instruct". A name without a known provider prefix
is an Ollama model ("gemma3:4b"), so older settings keep working.

Cloud providers are enabled just by setting their API key in the environment,
which lets the NAS proofread without running a model at all.
"""

import json
import re
import threading
import time
from dataclasses import dataclass, field

import httpx

from app import config


@dataclass
class Provider:
    key: str
    name: str
    base_url: str
    api_key: str = ""
    # Shown when the provider's /models can't be listed, or instead of it when set via <KEY>_MODELS.
    default_models: list[str] = field(default_factory=list)
    curated: bool = False  # True: only show default_models (e.g. OpenRouter lists hundreds)
    extra_body: dict = field(default_factory=dict)
    json_mode: bool = True  # supports response_format={"type": "json_object"}
    # Used instead of extra_body's thinking switch for models that refuse to turn thinking off.
    thinking_body: dict = field(default_factory=dict)
    always_think: set[str] = field(default_factory=set)  # learned from the API's refusal

    def body_for(self, model: str) -> dict:
        return {**self.extra_body, **self.thinking_body} if model in self.always_think else dict(self.extra_body)

    @property
    def is_ollama(self) -> bool:
        return self.key == "ollama"

    @property
    def enabled(self) -> bool:
        return bool(self.base_url) and (self.is_ollama or bool(self.api_key))


def _models_env(name: str) -> list[str]:
    return [m.strip() for m in config.env(f"{name}_MODELS", "").split(",") if m.strip()]


def _build() -> dict[str, Provider]:
    e = config.env
    providers = [
        Provider("ollama", "Ollama (ในเครื่อง)", config.OLLAMA_HOST),
        Provider(
            # Coding Plan endpoint: plan subscribers are billed against the plan, not pay-as-you-go credit.
            "zai", "Z.AI GLM", e("ZAI_BASE_URL", "https://api.z.ai/api/coding/paas/v4"), e("ZAI_API_KEY", ""),
            default_models=_models_env("ZAI") or ["glm-4.7-flash", "glm-4.5-flash"],
            curated=bool(_models_env("ZAI")),
            # GLM 4.5+ think before answering by default; spelling fixes don't need it.
            extra_body={"thinking": {"type": "disabled"}},
            # GLM 5.3 always thinks and rejects "disabled" (error 1210); it takes an effort instead.
            thinking_body={"thinking": {"type": "enabled"}, "reasoning_effort": e("ZAI_REASONING_EFFORT", "max")},
        ),
        Provider(
            "typhoon", "Typhoon API", e("TYPHOON_BASE_URL", "https://api.opentyphoon.ai/v1"), e("TYPHOON_API_KEY", ""),
            default_models=_models_env("TYPHOON") or ["typhoon-v2.5-30b-a3b-instruct"],
            curated=bool(_models_env("TYPHOON")),
        ),
        Provider(
            "openrouter", "OpenRouter", e("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"), e("OPENROUTER_API_KEY", ""),
            default_models=_models_env("OPENROUTER"), curated=True,
        ),
        Provider(
            "gemini", "Google Gemini", e("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai"),
            e("GEMINI_API_KEY", ""),
            default_models=_models_env("GEMINI") or ["gemini-2.5-flash"], curated=bool(_models_env("GEMINI")),
        ),
        # Anything else that speaks the OpenAI chat API (Groq, DeepSeek, a LiteLLM proxy, ...).
        Provider(
            "custom", e("CUSTOM_LLM_NAME", "Custom API"), e("CUSTOM_LLM_BASE_URL", ""), e("CUSTOM_LLM_API_KEY", ""),
            default_models=_models_env("CUSTOM_LLM"), curated=bool(_models_env("CUSTOM_LLM")),
        ),
    ]
    return {p.key: p for p in providers}


PROVIDERS = _build()

# Model ids that are clearly not chat models.
_NOT_CHAT = re.compile(r"embed|ocr|asr|whisper|tts|speech|audio|image|vision-only|rerank|moderation|guard", re.I)


def split_ref(ref: str) -> tuple[Provider, str]:
    """'zai:glm-4.7-flash' -> (zai, 'glm-4.7-flash'); 'gemma3:4b' -> (ollama, 'gemma3:4b')."""
    prefix, sep, rest = ref.partition(":")
    if sep and prefix in PROVIDERS and prefix != "ollama":
        return PROVIDERS[prefix], rest
    if sep and prefix == "ollama":
        return PROVIDERS["ollama"], rest
    return PROVIDERS["ollama"], ref


def ref(provider: Provider, model: str) -> str:
    return model if provider.is_ollama else f"{provider.key}:{model}"


def default_model() -> str:
    """PROOFREAD_MODEL if set, else the first cloud provider with a key, else Ollama's default."""
    if config.PROOFREAD_MODEL:
        return config.PROOFREAD_MODEL
    for p in PROVIDERS.values():
        if p.enabled and not p.is_ollama and p.default_models:
            return ref(p, p.default_models[0])
    return config.OLLAMA_MODEL


# ---------- listing models (cached: some providers are slow or rate-limited) ----------

_cache: dict[str, tuple[float, dict]] = {}
_cache_lock = threading.Lock()


def _headers(p: Provider) -> dict:
    return {"Authorization": f"Bearer {p.api_key}"} if p.api_key else {}


def _list_models(p: Provider) -> dict:
    if not p.enabled:
        return {"ok": False, "status": "ยังไม่ได้ใส่ API key" if not p.is_ollama else "ปิดใช้งาน", "models": []}
    try:
        if p.is_ollama:
            r = httpx.get(f"{p.base_url}/api/tags", timeout=4)
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
        elif p.curated:
            return {"ok": True, "status": "พร้อมใช้", "models": p.default_models}
        else:
            r = httpx.get(f"{p.base_url}/models", headers=_headers(p), timeout=6)
            r.raise_for_status()
            data = r.json()
            items = data.get("data", []) if isinstance(data, dict) else data  # Typhoon returns a bare list
            models = [str(m.get("id", "")).removeprefix("models/") for m in items if isinstance(m, dict)]
        models = sorted(m for m in models if m and not _NOT_CHAT.search(m))
        if not p.is_ollama and not models:
            models = p.default_models
        return {"ok": True, "status": "พร้อมใช้", "models": models[:40]}
    except (httpx.HTTPError, ValueError, KeyError) as e:
        if not p.is_ollama and p.default_models:
            # Listing failed but the key may still work for chat; let the user try.
            return {"ok": True, "status": f"ดูรายชื่อโมเดลไม่ได้ ({e.__class__.__name__})", "models": p.default_models}
        return {"ok": False, "status": f"เชื่อมต่อไม่ได้: {e}", "models": []}


def status(force: bool = False) -> dict:
    """All providers with their models, cached for 5 minutes."""
    out = []
    for p in PROVIDERS.values():
        if not p.enabled and p.key == "custom":
            continue
        with _cache_lock:
            hit = _cache.get(p.key)
        if hit and not force and time.time() - hit[0] < 300:
            info = hit[1]
        else:
            info = _list_models(p)
            with _cache_lock:
                _cache[p.key] = (time.time(), info)
        out.append({
            "key": p.key, "name": p.name, "local": p.is_ollama, "enabled": p.enabled,
            **info, "models": [ref(p, m) for m in info["models"]],
        })
    any_ok = any(x["ok"] and x["models"] for x in out)
    return {"ok": any_ok, "providers": out, "default_model": default_model()}


# ---------- chat ----------

def _extract_json(text: str) -> str:
    """Models without a strict JSON mode sometimes wrap the answer in ``` fences or prose."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    if not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            text = text[start:end + 1]
    return text


def _refuses_no_thinking(r: httpx.Response) -> bool:
    """Z.AI error 1210: "This model always engages in thinking and cannot be disabled"."""
    try:
        return str((r.json().get("error") or {}).get("code")) == "1210"
    except ValueError:
        return False


def chat_json(model_ref: str, system: str, user: str, schema: dict, timeout: float) -> tuple[dict, dict]:
    """Ask for a JSON answer; returns (parsed JSON, stats)."""
    p, model = split_ref(model_ref)
    if not p.enabled:
        raise RuntimeError(f"{p.name}: ยังไม่ได้ตั้งค่า (ใส่ API key ใน docker-compose.yml)")
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    t0 = time.time()

    if p.is_ollama:
        r = httpx.post(
            f"{p.base_url}/api/chat",
            json={"model": model, "stream": False, "format": schema, "options": {"temperature": 0}, "messages": messages},
            timeout=timeout,
        )
        if r.status_code >= 400:
            raise RuntimeError(f"Ollama ตอบ {r.status_code}: {r.text[:300]}")
        data = r.json()
        ns = 1e9
        eval_s = (data.get("eval_duration") or 0) / ns
        stats = {
            "total_s": round((data.get("total_duration") or 0) / ns, 2),
            "load_s": round((data.get("load_duration") or 0) / ns, 2),
            "prompt_s": round((data.get("prompt_eval_duration") or 0) / ns, 2),
            "eval_s": round(eval_s, 2),
            "eval_tokens": data.get("eval_count") or 0,
            "tokens_per_s": round((data.get("eval_count") or 0) / eval_s, 1) if eval_s else 0,
            "thinking": bool((data.get("message") or {}).get("thinking")),
        }
        content = (data.get("message") or {}).get("content", "")
    else:
        def post():
            body = {"model": model, "messages": messages, "temperature": 0, **p.body_for(model)}
            if p.json_mode:
                body["response_format"] = {"type": "json_object"}
            return httpx.post(f"{p.base_url}/chat/completions", headers=_headers(p), json=body, timeout=timeout)

        r = post()
        if r.status_code >= 400 and p.thinking_body and model not in p.always_think and _refuses_no_thinking(r):
            p.always_think.add(model)
            r = post()
        if r.status_code >= 400:
            raise RuntimeError(f"{p.name} ตอบ {r.status_code}: {r.text[:300]}")
        data = r.json()
        wall = time.time() - t0
        msg = ((data.get("choices") or [{}])[0].get("message") or {})
        usage = data.get("usage") or {}
        out_tokens = usage.get("completion_tokens") or 0
        stats = {
            "total_s": round(wall, 2), "load_s": 0.0, "prompt_s": 0.0, "eval_s": round(wall, 2),
            "eval_tokens": out_tokens,
            "tokens_per_s": round(out_tokens / wall, 1) if wall and out_tokens else 0,
            "thinking": bool(msg.get("reasoning_content") or msg.get("reasoning")),
        }
        content = msg.get("content") or ""

    try:
        return json.loads(_extract_json(content)), stats
    except ValueError as e:
        raise ValueError(f"โมเดลตอบไม่เป็น JSON: {e}") from e


def unload(model_ref: str) -> None:
    """Free an Ollama model's memory right away (no-op for cloud providers)."""
    p, model = split_ref(model_ref)
    if not p.is_ollama or not p.enabled:
        return
    try:
        httpx.post(f"{p.base_url}/api/generate", json={"model": model, "keep_alive": 0}, timeout=30)
    except httpx.HTTPError:
        pass
