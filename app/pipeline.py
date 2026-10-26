"""The heavy steps, run one at a time by the job queue.

Every task is a chain of nodes (see jobs.step); the phone draws them live.
"""

import json
import time

from app import jobs
from app.services import llm, media, proofreader, renderer, stt_cloud, subtitles, transcriber
from app.services.segmenter import build_cues
from app.services.thai_text import parse_hint_words

# (key, label, device, weight) - weight ~ share of the total time, for the overall bar
TRANSCRIBE_NODES = [
    ("extract", "แยกเสียงจากคลิป", "CPU", 0.05),
    ("whisper", "AI ฟังเสียงพูด", "iGPU", 0.75),
    ("segment", "จัดท่อน 1–2 บรรทัด", "CPU", 0.05),
    ("proofread", "AI ตรวจคำผิด", "LLM", 0.15),
]
PROOFREAD_NODES = [("proofread", "AI ตรวจคำผิด", "LLM", 1.0)]
RENDER_NODES = [
    ("ass", "สร้างไฟล์ซับ .ass", "CPU", 0.02),
    ("hw", "ตรวจ iGPU", "iGPU", 0.03),
    ("render", "Enhance + ฝังซับ + เข้ารหัส", "iGPU", 0.93),
    ("finish", "บันทึกไฟล์ Master MP4", "CPU", 0.02),
]

VARIANT_LABEL = {"gpu": "iGPU (VAAPI)", "gpu-swdec": "iGPU (ถอดรหัสด้วย CPU)", "cpu": "CPU (libx264)"}


def transcribe_task(job_id: str, model: str | None, vad: bool, proofread: bool, llm_model: str | None) -> None:
    model = model or stt_cloud.default_model()
    job = jobs.load(job_id)
    d = jobs.job_dir(job_id)
    hint = job.get("hint", "")
    t0 = time.time()
    if not job["info"].get("has_audio"):
        with jobs.step(job_id, "extract"):
            raise ValueError("คลิปนี้ไม่มีเสียง ถอดเสียงไม่ได้ (ยังเพิ่มแคปชันเองได้)")

    with jobs.step(job_id, "extract"):
        wav = media.extract_audio(d / job["source"], d / "audio.wav")

    if stt_cloud.is_cloud(model):
        # Cloud STT: nothing runs on the NAS for this step.
        used_model = model
        jobs.step_update(job_id, "whisper", device="Cloud")
        with jobs.step(job_id, "whisper", model):
            result = stt_cloud.transcribe(wav, hint=hint, model=model)
            (d / "whisper.json").write_text(json.dumps(result, ensure_ascii=False))
    else:
        with jobs.step(job_id, "whisper", (model or transcriber.current_model()).removeprefix("ggml-").removesuffix(".bin")):
            result = transcriber.transcribe(wav, hint=hint, model=model, vad=vad)
            if vad and not build_cues(result, parse_hint_words(hint)):
                # Silero VAD is trained on speech and can drop singing over music entirely.
                jobs.step_update(job_id, "whisper", detail="VAD ไม่เจอเสียงพูด (อาจเป็นเพลง) กำลังถอดใหม่แบบไม่ใช้ VAD")
                result = transcriber.transcribe(wav, hint=hint, model=model, vad=False)
                vad = False
                jobs.step_update(job_id, "whisper", detail="VAD ไม่เจอเสียงพูด ถอดใหม่แบบไม่ใช้ VAD แล้ว")
            (d / "whisper.json").write_text(json.dumps(result, ensure_ascii=False))
        used_model = transcriber.current_model()

    with jobs.step(job_id, "segment"):
        cues = build_cues(result, parse_hint_words(hint))
        jobs.step_update(job_id, "segment", detail=f"{len(cues)} ท่อน")
    jobs.update(
        job_id, cues=cues, suggestions=[], proofread_error="",
        transcribe_meta={"model": used_model, "seconds": round(time.time() - t0, 1), "vad": vad},
    )

    if not proofread or not cues:
        jobs.step_update(job_id, "proofread", "skipped", detail="ปิดไว้" if not proofread else "ไม่มีซับ")
        return
    _proofread(job_id, cues, hint, llm_model)


def device_label(model_ref: str) -> str:
    """Node badge: "Ollama" for the local model, "Cloud" for an API."""
    return "Ollama" if llm.split_ref(model_ref)[0].is_ollama else "Cloud"


def _proofread(job_id: str, cues: list[dict], hint: str, llm_model: str | None) -> None:
    # Proofreading is a bonus: if the LLM is down or answers badly, keep the transcript.
    model = llm_model or llm.default_model()
    jobs.step_update(job_id, "proofread", "running", 0.0, model, device=device_label(model))

    def progress(done: int, total: int) -> None:
        jobs.step_update(job_id, "proofread", progress=done / total, detail=f"{model} · ชุด {done}/{total}")

    try:
        suggestions, errors = proofreader.proofread(cues, hint, model, on_progress=progress)
    except Exception as e:
        jobs.update(job_id, suggestions=[], proofread_error=f"ตรวจคำผิดไม่สำเร็จ: {e}")
        jobs.step_update(job_id, "proofread", "warn", detail="ไม่สำเร็จ ข้ามไป (ซับยังอยู่ครบ)")
        return
    found = f"แนะนำแก้ {len(suggestions)} จุด" if suggestions else "ไม่พบคำผิด"
    if errors:
        jobs.update(job_id, suggestions=suggestions, proofread_error="ตรวจคำผิดไม่ครบ: " + " · ".join(errors))
        jobs.step_update(job_id, "proofread", "warn", detail=f"{found} · ตรวจไม่สำเร็จ {len(errors)} ชุด")
        return
    jobs.update(job_id, suggestions=suggestions, proofread_error="")
    jobs.step_update(job_id, "proofread", "done", detail=found)


def proofread_task(job_id: str, llm_model: str | None) -> None:
    job = jobs.load(job_id)
    _proofread(job_id, job.get("cues") or [], job.get("hint", ""), llm_model)


def render_task(job_id: str) -> None:
    job = jobs.load(job_id)
    d = jobs.job_dir(job_id)
    style = subtitles.SubtitleStyle.from_dict(job.get("style"))
    opts = renderer.RenderOptions.from_dict(job.get("render_options"))
    t0 = time.time()

    with jobs.step(job_id, "ass"):
        burn = job.get("burn_subs", True)
        cues = (job.get("cues") or []) if burn else []
        ass = subtitles.build_ass(cues, style, d / "subs.ass", parse_hint_words(job.get("hint", "")))
        if burn:
            jobs.step_update(job_id, "ass", detail=f"{len(cues)} ท่อน · {style.font} {style.size}px")
        else:
            jobs.step_update(job_id, "ass", "skipped", detail="ไม่ฝังซับ (Enhance อย่างเดียว)")

    with jobs.step(job_id, "hw"):
        caps = renderer.detect_caps()
        if caps.mode == "vaapi":
            jobs.step_update(job_id, "hw", detail="พร้อมใช้ VAAPI" + (" (low-power)" if caps.low_power else ""))
        else:
            jobs.step_update(job_id, "hw", "warn", detail="ใช้ไม่ได้ จะเรนเดอร์ด้วย CPU")

    out = d / "output.mp4"
    tmp = d / "output.part.mp4"
    tmp.unlink(missing_ok=True)
    last = [0.0]

    def progress(p: float) -> None:
        if p - last[0] >= 0.02 or p >= 1.0:  # don't rewrite job.json on every ffmpeg tick
            last[0] = p
            jobs.step_update(job_id, "render", progress=p)

    def status(variant: str, failed: str | None) -> None:
        last[0] = 0.0
        note = f"{VARIANT_LABEL[failed]} ไม่สำเร็จ → ลอง {VARIANT_LABEL[variant]}" if failed else VARIANT_LABEL[variant]
        jobs.step_update(job_id, "render", progress=0.0, detail=note, device="CPU" if variant == "cpu" else "iGPU")

    with jobs.step(job_id, "render"):
        variant = renderer.render(d / job["source"], tmp, ass, d, job["info"], opts, progress, status)
        jobs.step_update(job_id, "render", detail=VARIANT_LABEL[variant])

    with jobs.step(job_id, "finish"):
        tmp.replace(out)
        size = out.stat().st_size
        jobs.step_update(job_id, "finish", detail=f"{size / 1048576:.1f} MB")
    jobs.update(
        job_id, output="output.mp4",
        render_meta={"variant": variant, "seconds": round(time.time() - t0, 1), "size": size},
    )
