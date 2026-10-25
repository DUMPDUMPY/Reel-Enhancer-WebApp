"""Optional spelling fixes from an LLM: local Ollama or a cloud API (see services/llm.py).

The model only sees cue ids and their text and must return the same ids. Any
answer that adds, drops or merges cues is thrown away, and each change is shown
to the user as a suggestion to accept, so the LLM can never shift timing or
silently rewrite what was actually said.
"""

import json

from app import config
from app.services import llm
from app.services.thai_text import display_width

SYSTEM_PROMPT = """คุณคือผู้ตรวจคำผิดซับไตเติลภาษาไทยที่ได้จากระบบถอดเสียงอัตโนมัติ
หน้าที่: แก้เฉพาะคำที่สะกดผิด คำพ้องเสียงที่ระบบฟังผิด และชื่อเฉพาะให้ตรงกับรายชื่อในบริบท
ห้าม: เรียบเรียงประโยคใหม่ แปลภาษา เพิ่มหรือตัดเนื้อหา รวมหรือแยกรายการ เปลี่ยนคำสแลงหรือภาษาพูดให้เป็นทางการ
ถ้ารายการไหนถูกต้องอยู่แล้ว ให้ส่งข้อความเดิมกลับมาทุกตัวอักษร
ตอบเป็น JSON เท่านั้น รูปแบบ {"items": [{"id": 1, "text": "..."}]} โดยมี id ครบทุกรายการตามที่ได้รับ"""

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
    return llm.status()


def ask(items: list[dict], context: str = "", model: str | None = None) -> tuple[list, dict]:
    """Send [{id, text}] to the model; return (answer items, timing stats)."""
    user = ""
    if context.strip():
        user += f"บริบท / ชื่อเฉพาะที่ต้องสะกดให้ถูก: {context.strip()}\n\n"
    user += "รายการซับไตเติล:\n" + json.dumps({"items": items}, ensure_ascii=False)
    answer, stats = llm.chat_json(model or llm.default_model(), SYSTEM_PROMPT, user, SCHEMA, config.OLLAMA_TIMEOUT)
    return (answer.get("items", []) if isinstance(answer, dict) else []), stats


def unload(model: str) -> None:
    """Free an Ollama model's memory now instead of after its keep-alive (no-op for cloud APIs)."""
    llm.unload(model)


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
    if not cues:
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
