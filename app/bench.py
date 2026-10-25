"""Compare proofreading models (local Ollama and cloud APIs) on the same text.

A benchmark run is stored like a job (kind="bench") so it reuses the queue and
the node progress UI: one node per model. With the built-in test set every cue
has an expected answer, so each model gets a score; with cues from a real job
there is no answer key and the UI just shows each model's suggestions side by side.
"""

import time

from app import jobs
from app.services import llm, proofreader

BUILTIN_CONTEXT = "กุ๊กไก่, เจ้าเหมียว, หมูจิ๋ว"

# Typical speech-to-text mistakes (names, tone marks, ร/ล, homophones) mixed with
# lines that are already correct, including slang that must NOT be "fixed".
BUILTIN_TESTSET = [
    {"text": "วันนี้กุกไก่แอบหนีเที่ยวอีกแล้ว", "expected": "วันนี้กุ๊กไก่แอบหนีเที่ยวอีกแล้ว", "kind": "ชื่อเฉพาะ"},
    {"text": "แต่ดันเจอเจ้าเมียวดักรออยู่หน้าบ้าน", "expected": "แต่ดันเจอเจ้าเหมียวดักรออยู่หน้าบ้าน", "kind": "ชื่อเฉพาะ"},
    {"text": "หมูจิวบอกว่าไม่เป็นไรหรอก", "expected": "หมูจิ๋วบอกว่าไม่เป็นไรหรอก", "kind": "ชื่อเฉพาะ"},
    {"text": "เดี๋ยวพวกเราไปกินข้าวกันก่อนนะ", "expected": "เดี๋ยวพวกเราไปกินข้าวกันก่อนนะ", "kind": "ถูกอยู่แล้ว"},
    {"text": "เขาไม่ได้ตั้งจัยจะทำร้ายใคร", "expected": "เขาไม่ได้ตั้งใจจะทำร้ายใคร", "kind": "สะกดผิด"},
    {"text": "อย่าลืมกดติดตามกันด้วยน้า", "expected": "อย่าลืมกดติดตามกันด้วยน้า", "kind": "ภาษาพูด (ห้ามแก้)"},
    {"text": "เราต้องลีบกลับบ้านก่อนฝนตก", "expected": "เราต้องรีบกลับบ้านก่อนฝนตก", "kind": "ร/ล"},
    {"text": "วันนี้อากาศร้อนมากกกก", "expected": "วันนี้อากาศร้อนมากกกก", "kind": "ภาษาพูด (ห้ามแก้)"},
    {"text": "คุณแม่ทำกับค่าวอร่อยที่สุดในโลก", "expected": "คุณแม่ทำกับข้าวอร่อยที่สุดในโลก", "kind": "พ้องเสียง"},
    {"text": "นี่มันเรื่องอะไรกันแน่", "expected": "นี่มันเรื่องอะไรกันแน่", "kind": "ถูกอยู่แล้ว"},
    {"text": "ยินดีตอนรับสู่ช่องของเรา", "expected": "ยินดีต้อนรับสู่ช่องของเรา", "kind": "วรรณยุกต์"},
    {"text": "เจ้าเหมียวกะโดดขึ้นไปบนโต๊ะ", "expected": "เจ้าเหมียวกระโดดขึ้นไปบนโต๊ะ", "kind": "สะกดผิด"},
    {"text": "ฉันอยากไปทะเลมากเลยค่ะ", "expected": "ฉันอยากไปทะเลมากเลยค่ะ", "kind": "ถูกอยู่แล้ว"},
    {"text": "โอ้โห แบบนี้ก็ได้เหรอเนี่ย", "expected": "โอ้โห แบบนี้ก็ได้เหรอเนี่ย", "kind": "ภาษาพูด (ห้ามแก้)"},
]


def _norm(s: str) -> str:
    return "".join(s.split())


def builtin_items() -> list[dict]:
    return [{"id": i + 1, **c} for i, c in enumerate(BUILTIN_TESTSET)]


def score(items: list[dict], finals: dict[int, str]) -> dict:
    """Compare each model's final text with the answer key."""
    with_errors = [it for it in items if _norm(it["text"]) != _norm(it["expected"])]
    clean = [it for it in items if _norm(it["text"]) == _norm(it["expected"])]
    fixed = sum(_norm(finals[it["id"]]) == _norm(it["expected"]) for it in with_errors)
    # Changed a wrong line into a different wrong line, e.g. ตั้งจัย -> ตั้งจี้: worse than leaving it.
    wrong = sum(
        _norm(finals[it["id"]]) not in (_norm(it["expected"]), _norm(it["text"])) for it in with_errors
    )
    over = sum(_norm(finals[it["id"]]) != _norm(it["expected"]) for it in clean)
    # Every error fixed is +1, every bad edit (wrong fix or broken correct line) is -1, scaled to 0-100.
    pts = max(fixed - wrong - over, 0)
    return {
        "fixed": fixed, "errors": len(with_errors), "wrong_edits": wrong,
        "over_edits": over, "clean": len(clean),
        "score": round(100 * pts / max(len(with_errors), 1)),
    }


def run_one(items: list[dict], context: str, model: str) -> dict:
    sent = [{"id": it["id"], "text": it["text"]} for it in items]
    t0 = time.time()
    try:
        out = proofreader.ask_validated(sent, context, model)
    except ValueError as e:
        return {"ok": False, "error": str(e), "wall_s": round(time.time() - t0, 2)}
    wall = round(time.time() - t0, 2)
    suggestions, received, stats = out["suggestions"], out["received"], out["stats"]

    # Edits the validator threw away as rewrites (too different from the original).
    got = {it.get("id"): str(it.get("text", "")).strip() for it in received if isinstance(it, dict)}
    accepted = {s["id"] for s in suggestions}
    rejected = sum(1 for it in sent if _norm(got.get(it["id"], "")) != _norm(it["text"]) and it["id"] not in accepted)

    finals = {it["id"]: it["text"] for it in sent}
    finals.update({s["id"]: s["after"] for s in suggestions})
    result = {
        "ok": True, "stats": stats, "wall_s": wall,
        "suggestions": suggestions, "rejected_rewrites": rejected,
        "attempts": out["attempts"], "failed_answers": out["failed"],
        "finals": {str(k): v for k, v in finals.items()},
    }
    if all("expected" in it for it in items):
        result["score"] = score(items, finals)
    return result


def bench_task(job_id: str) -> None:
    job = jobs.load(job_id)
    items, context = job["items"], job.get("context", "")
    for model in job["models"]:
        key = f"m:{model}"
        jobs.step_update(job_id, key, "running", detail="กำลังส่งให้โมเดลตรวจ")
        try:
            res = run_one(items, context, model)
        except Exception as e:
            res = {"ok": False, "error": str(e)[-300:]}
        finally:
            proofreader.unload(model)
        with jobs.edit(job_id) as j:
            j.setdefault("results", {})[model] = res
        if not res["ok"]:
            jobs.step_update(job_id, key, "warn", detail=res["error"][:120])
            continue
        st = res["stats"]
        parts = []
        if "score" in res:
            sc = res["score"]
            parts.append(f"คะแนน {sc['score']} · แก้ถูก {sc['fixed']}/{sc['errors']} · แก้ผิด {sc['wrong_edits']} · แก้เกิน {sc['over_edits']}")
        else:
            parts.append(f"แนะนำ {len(res['suggestions'])} จุด")
        parts.append(f"{res['wall_s']} วิ")
        if res["attempts"] > 1:
            parts.append(f"ตอบผิดรูปแบบ {res['attempts'] - 1} ครั้ง ลองใหม่แล้วผ่าน")
        if st.get("tokens_per_s"):
            parts.append(f"{st['tokens_per_s']} tok/s")
        jobs.step_update(job_id, key, "done", detail=" · ".join(parts))


def nodes_for(models: list[str]) -> list[tuple[str, str, str, float]]:
    return [(f"m:{m}", m, "Ollama" if llm.split_ref(m)[0].is_ollama else "Cloud", 1.0) for m in models]
