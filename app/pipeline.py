"""The heavy steps, run one at a time by the job queue."""

import json
import time

from app import jobs
from app.services import media, proofreader, renderer, subtitles, transcriber
from app.services.segmenter import build_cues
from app.services.thai_text import parse_hint_words


def transcribe_task(job_id: str, model: str | None, vad: bool, proofread: bool, ollama_model: str | None) -> None:
    job = jobs.load(job_id)
    d = jobs.job_dir(job_id)
    hint = job.get("hint", "")
    t0 = time.time()
    if not job["info"].get("has_audio"):
        raise ValueError("คลิปนี้ไม่มีเสียง ถอดเสียงไม่ได้ (ยังเพิ่มแคปชันเองได้)")

    jobs.set_task(job_id, "transcribe", "running", 0.05, "แยกเสียงจากคลิป")
    wav = media.extract_audio(d / job["source"], d / "audio.wav")

    jobs.set_task(job_id, "transcribe", "running", 0.15, "AI กำลังฟังเสียงพูด (iGPU)")
    result = transcriber.transcribe(wav, hint=hint, model=model, vad=vad)
    (d / "whisper.json").write_text(json.dumps(result, ensure_ascii=False))

    jobs.set_task(job_id, "transcribe", "running", 0.85, "จัดซับเป็นท่อน 1–2 บรรทัด")
    cues = build_cues(result, parse_hint_words(hint))
    jobs.update(
        job_id, cues=cues, suggestions=[], proofread_error="",
        transcribe_meta={"model": transcriber.current_model(), "seconds": round(time.time() - t0, 1), "vad": vad},
    )

    if proofread and cues:
        jobs.set_task(job_id, "transcribe", "running", 0.9, "Ollama กำลังตรวจคำผิด")
        _proofread(job_id, cues, hint, ollama_model)


def _proofread(job_id: str, cues: list[dict], hint: str, ollama_model: str | None) -> None:
    # Proofreading is a bonus: if Ollama is down or answers badly, keep the transcript.
    try:
        suggestions = proofreader.proofread(cues, hint, ollama_model)
        jobs.update(job_id, suggestions=suggestions, proofread_error="")
    except Exception as e:
        jobs.update(job_id, suggestions=[], proofread_error=f"ตรวจคำผิดไม่สำเร็จ: {e}")


def proofread_task(job_id: str, ollama_model: str | None) -> None:
    job = jobs.load(job_id)
    jobs.set_task(job_id, "proofread", "running", 0.3, "Ollama กำลังตรวจคำผิด")
    _proofread(job_id, job.get("cues") or [], job.get("hint", ""), ollama_model)


def render_task(job_id: str) -> None:
    job = jobs.load(job_id)
    d = jobs.job_dir(job_id)
    style = subtitles.SubtitleStyle.from_dict(job.get("style"))
    opts = renderer.RenderOptions.from_dict(job.get("render_options"))
    t0 = time.time()

    jobs.set_task(job_id, "render", "running", 0.02, "สร้างไฟล์ซับ .ass")
    ass = subtitles.build_ass(job.get("cues") or [], style, d / "subs.ass", parse_hint_words(job.get("hint", "")))

    out = d / "output.mp4"
    tmp = d / "output.part.mp4"
    tmp.unlink(missing_ok=True)

    def progress(p: float) -> None:
        jobs.set_task(job_id, "render", "running", 0.05 + p * 0.93, f"เรนเดอร์ {int(p * 100)}%")

    jobs.set_task(job_id, "render", "running", 0.05, "เริ่มเรนเดอร์")
    variant = renderer.render(d / job["source"], tmp, ass, d, job["info"], opts, progress)
    tmp.replace(out)
    jobs.update(
        job_id, output="output.mp4",
        render_meta={"variant": variant, "seconds": round(time.time() - t0, 1), "size": out.stat().st_size},
    )
