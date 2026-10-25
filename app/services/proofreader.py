"""Optional spelling fixes from a (remote) Ollama server.

The model only sees cue ids and their text and must return the same ids. Any
answer that adds, drops or merges cues is thrown away, and each change is shown
to the user as a suggestion to accept, so the LLM can never shift timing or
silently rewrite what was actually said.
"""

import json

import httpx

from app import config
from app.services.thai_text import display_width

SYSTEM_PROMPT = """คุณคือผู้ตรวจคำผิดซับไตเติลภาษาไทยที่ได้จากระบบถอดเสียงอัตโนมัติ
หน้าที่: แก้เฉพาะคำที่สะกดผิด คำพ้องเสียงที่ระบบฟังผิด และชื่อเฉพาะให้ตรงกับรายชื่อในบริบท
ห้าม: เรียบเรียงประโยคใหม่ แปลภาษา เพิ่มหรือตัดเนื้อหา รวมหรือแยกรายการ เปลี่ยนคำสแลงหรือภาษาพูดให้เป็นทางการ
ถ้ารายการไหนถูกต้องอยู่แล้ว ให้ส่งข้อความเดิมกลับมาทุกตัวอักษร
ตอบเป็น JSON ตาม schema เท่านั้น โดยมี id ครบทุกรายการตามที่ได้รับ"""

SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "integer"}, "text": {"type": "string"}},
                "required": ["id", "text"],
            },
        }
    },
    "required": ["items"],
}


def _flat(text: str) -> str:
    return "".join(line.strip() for line in text.split("\n"))


def status() -> dict:
    if not config.OLLAMA_HOST:
        return {"ok": False, "status": "ปิดใช้งาน (ไม่ได้ตั้ง OLLAMA_HOST)", "models": []}
    try:
        r = httpx.get(f"{config.OLLAMA_HOST}/api/tags", timeout=3)
        r.raise_for_status()
        models = sorted(m["name"] for m in r.json().get("models", []) if "embed" not in m["name"])
        return {"ok": True, "status": "เชื่อมต่อได้", "models": models}
    except (httpx.HTTPError, ValueError, KeyError) as e:
        return {"ok": False, "status": f"เชื่อมต่อไม่ได้: {e}", "models": []}


def ask(items: list[dict], context: str = "", model: str | None = None) -> tuple[list, dict]:
    """Send [{id, text}] to Ollama; return (answer items, timing stats)."""
    user = ""
    if context.strip():
        user += f"บริบท / ชื่อเฉพาะที่ต้องสะกดให้ถูก: {context.strip()}\n\n"
    user += "รายการซับไตเติล:\n" + json.dumps({"items": items}, ensure_ascii=False)

    r = httpx.post(
        f"{config.OLLAMA_HOST}/api/chat",
        json={
            "model": model or config.OLLAMA_MODEL,
            "stream": False,
            "format": SCHEMA,
            "options": {"temperature": 0},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
        },
        timeout=config.OLLAMA_TIMEOUT,
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
    try:
        answer = json.loads(data["message"]["content"])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"Ollama ตอบไม่เป็น JSON: {e}") from e
    return (answer.get("items", []) if isinstance(answer, dict) else []), stats


def unload(model: str) -> None:
    """Free the model's memory now instead of after Ollama's keep-alive (5 min).

    Used between models when comparing, so several models are never resident at
    once (on the NAS the iGPU shares system RAM).
    """
    try:
        httpx.post(f"{config.OLLAMA_HOST}/api/generate", json={"model": model, "keep_alive": 0}, timeout=30)
    except httpx.HTTPError:
        pass


def ask_validated(items: list[dict], context: str = "", model: str | None = None, retries: int = 1) -> dict:
    """ask() + validate(), retrying when the answer is malformed.

    Small models occasionally drop or merge items even with a JSON schema; a
    second try usually comes back complete. Returns suggestions, the accepted
    answer, stats of the last call, attempts and the raw answers that failed.
    """
    failed = []
    for attempt in range(1, retries + 2):
        received, stats = ask(items, context, model)
        try:
            suggestions = validate(items, received)
        except ValueError as e:
            failed.append({"error": str(e), "answer": received})
            continue
        return {"suggestions": suggestions, "received": received, "stats": stats, "attempts": attempt, "failed": failed}
    raise ValueError(f"{failed[-1]['error']} (ลอง {len(failed)} ครั้ง)")


def proofread(cues: list[dict], context: str = "", model: str | None = None) -> list[dict]:
    """Return suggestions [{id, before, after}] for cues whose spelling the model fixed."""
    if not config.OLLAMA_HOST or not cues:
        return []
    items = [{"id": int(c["id"]), "text": _flat(c["text"])} for c in cues]
    return ask_validated(items, context, model)["suggestions"]


def validate(sent: list[dict], received: list) -> list[dict]:
    """Keep only well-formed, conservative edits; reject the whole answer if ids don't match."""
    got = {}
    for it in received:
        if isinstance(it, dict) and isinstance(it.get("id"), int) and isinstance(it.get("text"), str):
            got[it["id"]] = it["text"].strip()
    if set(got) != {it["id"] for it in sent}:
        raise ValueError("Ollama ตอบกลับรายการไม่ครบหรือเกิน จึงไม่ใช้ผลตรวจนี้")

    suggestions = []
    for it in sent:
        before, after = it["text"], _flat(got[it["id"]])
        if not after or after == before:
            continue
        # A spelling fix keeps roughly the same length; anything else is a rewrite.
        wb, wa = display_width(before), display_width(after)
        if abs(wa - wb) > max(4, wb * 0.3):
            continue
        suggestions.append({"id": it["id"], "before": before, "after": after})
    return suggestions
