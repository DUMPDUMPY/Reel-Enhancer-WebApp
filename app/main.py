"""FastAPI app: mobile web UI + JSON API."""

import logging
import shutil
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import bench, config, jobs, pipeline
from app.services import llm, media, proofreader, renderer, stt_cloud, subtitles, transcriber

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".3gp"}


def _housekeeping() -> None:
    while True:
        try:
            removed = jobs.cleanup_old()
            if removed:
                log.info("removed %d old jobs", removed)
        except Exception:
            log.exception("cleanup failed")
        time.sleep(3600)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    jobs.recover_interrupted()
    threading.Thread(target=_housekeeping, daemon=True).start()
    threading.Thread(target=renderer.detect_caps, daemon=True).start()
    yield


app = FastAPI(title="Reel Enhancer", lifespan=lifespan)


@app.middleware("http")
async def revalidate_ui(request, call_next):
    """Make browsers recheck the page/JS/CSS (cheap 304 via ETag), so a rebuild shows up without
    clearing the phone's cache. API responses and fonts set their own caching."""
    response = await call_next(request)
    if not request.url.path.startswith(("/api/", "/fonts/")):
        response.headers.setdefault("Cache-Control", "no-cache")
    return response


def _job_or_404(job_id: str) -> dict:
    try:
        return jobs.load(job_id)
    except KeyError:
        raise HTTPException(404, "ไม่พบงานนี้ (อาจถูกลบไปแล้ว)") from None


def _ensure_idle(job: dict) -> None:
    if (job.get("task") or {}).get("state") in ("queued", "running"):
        raise HTTPException(409, "งานนี้กำลังทำงานอยู่ รอให้เสร็จก่อน")


# ---------- system ----------

@app.get("/api/system")
def system() -> dict:
    caps = renderer.detect_caps()
    return {
        "whisper": {**transcriber.health(), "models": transcriber.list_models(), "current": transcriber.current_model()},
        "stt_cloud": {"ok": stt_cloud.enabled(), "models": stt_cloud.models()},
        "stt_default": stt_cloud.default_model() or transcriber.current_model(),
        "llm": proofreader.status(),
        "render": {"mode": caps.mode, "low_power": caps.low_power, "detail": caps.detail, "checked": caps.checked},
        "fonts": subtitles.list_fonts(),
        "defaults": {
            "style": subtitles.SubtitleStyle().to_dict(),
            "render_options": renderer.RenderOptions().__dict__,
            "output": [config.OUTPUT_WIDTH, config.OUTPUT_HEIGHT],
        },
        "busy": jobs.busy(),
    }


@app.post("/api/hwcheck")
def hwcheck() -> dict:
    caps = renderer.detect_caps(force=True)
    return {"mode": caps.mode, "low_power": caps.low_power, "detail": caps.detail}


# ---------- Ollama memory ----------

@app.get("/api/ollama/loaded")
def ollama_loaded() -> dict:
    return llm.ollama_loaded()


class UnloadRequest(BaseModel):
    model: str | None = None  # None: every loaded model


@app.post("/api/ollama/unload")
def ollama_unload(req: UnloadRequest) -> dict:
    info = llm.ollama_loaded()
    if not info["ok"]:
        raise HTTPException(503, info["error"])
    names = [req.model] if req.model else [m["name"] for m in info["models"]]
    for name in names:
        try:
            llm.unload_now(name)
        except RuntimeError as e:
            raise HTTPException(502, str(e))
    return llm.ollama_loaded()


@app.get("/fonts/{name}")
def font(name: str) -> FileResponse:
    p = subtitles.font_file(name)
    if not p:
        raise HTTPException(404)
    return FileResponse(p, headers={"Cache-Control": "max-age=86400"})


# ---------- jobs ----------

@app.get("/api/jobs")
def list_jobs() -> list[dict]:
    return [
        {"id": j["id"], "filename": j.get("filename"), "created": j.get("created"),
         "has_output": bool(j.get("output")), "cues": len(j.get("cues") or []),
         "size": jobs.dir_size(j["id"]), "busy": (j.get("task") or {}).get("state") in ("queued", "running")}
        for j in jobs.list_recent(200)  # all of them, so old jobs can be deleted from the list
        if j.get("kind") != "bench"
    ]


@app.post("/api/jobs")
def create_job(file: UploadFile = File(...), hint: str = Form("")) -> dict:
    ext = Path(file.filename or "clip.mp4").suffix.lower()
    if ext not in VIDEO_EXTS:
        raise HTTPException(400, f"ไม่รองรับไฟล์ {ext} (ใช้ได้: {', '.join(sorted(VIDEO_EXTS))})")
    job_id = jobs.new_id()
    d = jobs.job_dir(job_id)
    d.mkdir(parents=True)
    src = d / f"source{ext}"
    limit = config.MAX_UPLOAD_MB * 1024 * 1024
    size = 0
    try:
        with src.open("wb") as f:
            while chunk := file.file.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(413, f"ไฟล์ใหญ่เกิน {config.MAX_UPLOAD_MB} MB")
                f.write(chunk)
        info = media.probe(src)
    except HTTPException:
        shutil.rmtree(d, ignore_errors=True)
        raise
    except Exception as e:
        shutil.rmtree(d, ignore_errors=True)
        raise HTTPException(400, f"อ่านไฟล์วิดีโอไม่ได้: {e}") from e
    try:
        media.thumbnail(src, d / "thumb.jpg", at=min(0.5, info["duration"] / 2))
    except Exception:
        log.warning("thumbnail failed for %s", job_id)

    job = {
        "id": job_id,
        "created": time.time(),
        "filename": file.filename,
        "source": src.name,
        "size": size,
        "info": info,
        "hint": hint.strip(),
        "cues": [],
        "suggestions": [],
        "style": subtitles.SubtitleStyle().to_dict(),
        "render_options": renderer.RenderOptions().__dict__,
        "task": {"name": None, "state": "idle", "progress": 0, "message": "", "error": ""},
    }
    return jobs.save(job)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    # "now" lets the phone show live elapsed times without trusting its own clock.
    return {**_job_or_404(job_id), "now": time.time()}


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str) -> dict:
    job = _job_or_404(job_id)
    _ensure_idle(job)
    shutil.rmtree(jobs.job_dir(job_id), ignore_errors=True)
    return {"ok": True}


class TranscribeRequest(BaseModel):
    hint: str | None = None
    model: str | None = None
    vad: bool = True
    proofread: bool = False
    llm_model: str | None = None


@app.post("/api/jobs/{job_id}/transcribe")
def transcribe(job_id: str, req: TranscribeRequest) -> dict:
    job = _job_or_404(job_id)
    _ensure_idle(job)
    if req.hint is not None:
        jobs.update(job_id, hint=req.hint.strip())
    model = req.model or stt_cloud.default_model()
    jobs.submit(job_id, "transcribe", pipeline.TRANSCRIBE_NODES, pipeline.transcribe_task, model, req.vad, req.proofread, req.llm_model)
    return jobs.load(job_id)


class Cue(BaseModel):
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    text: str = ""


class CuesRequest(BaseModel):
    cues: list[Cue]
    hint: str | None = None
    style: dict | None = None
    render_options: dict | None = None


def _normalize_cues(cues: list[Cue], duration: float) -> list[dict]:
    out = []
    for c in sorted(cues, key=lambda c: c.start):
        text = "\n".join(line.strip() for line in c.text.strip().splitlines() if line.strip())
        if not text:
            continue
        start = round(min(c.start, max(duration - 0.1, 0)), 3) if duration else round(c.start, 3)
        end = round(max(c.end, start + 0.1), 3)
        if duration:
            end = min(end, duration)
        out.append({"id": len(out) + 1, "start": start, "end": end, "text": text})
    return out


def _save_edits(job_id: str, req: CuesRequest) -> dict:
    job = _job_or_404(job_id)
    _ensure_idle(job)
    fields: dict = {"cues": _normalize_cues(req.cues, job["info"].get("duration") or 0)}
    if req.hint is not None:
        fields["hint"] = req.hint.strip()
    if req.style is not None:
        fields["style"] = subtitles.SubtitleStyle.from_dict(req.style).to_dict()
    if req.render_options is not None:
        fields["render_options"] = renderer.RenderOptions.from_dict(req.render_options).__dict__
    return jobs.update(job_id, **fields)


@app.put("/api/jobs/{job_id}/cues")
def save_cues(job_id: str, req: CuesRequest) -> dict:
    return _save_edits(job_id, req)


class ProofreadRequest(BaseModel):
    llm_model: str | None = None


@app.post("/api/jobs/{job_id}/proofread")
def proofread(job_id: str, req: ProofreadRequest) -> dict:
    job = _job_or_404(job_id)
    _ensure_idle(job)
    if not job.get("cues"):
        raise HTTPException(400, "ยังไม่มีซับให้ตรวจ")
    jobs.submit(job_id, "proofread", pipeline.PROOFREAD_NODES, pipeline.proofread_task, req.llm_model)
    return jobs.load(job_id)


@app.post("/api/jobs/{job_id}/render")
def render(job_id: str, req: CuesRequest) -> dict:
    _save_edits(job_id, req)
    jobs.update(job_id, output=None)
    jobs.submit(job_id, "render", pipeline.RENDER_NODES, pipeline.render_task)
    return jobs.load(job_id)


@app.get("/api/jobs/{job_id}/source")
def source(job_id: str) -> FileResponse:
    job = _job_or_404(job_id)
    return FileResponse(jobs.job_dir(job_id) / job["source"])


@app.get("/api/jobs/{job_id}/thumb.jpg")
def thumb(job_id: str) -> FileResponse:
    _job_or_404(job_id)
    p = jobs.job_dir(job_id) / "thumb.jpg"
    if not p.is_file():
        raise HTTPException(404)
    return FileResponse(p)


@app.get("/api/jobs/{job_id}/subs.vtt")
def subs_vtt(job_id: str) -> PlainTextResponse:
    job = _job_or_404(job_id)
    return PlainTextResponse(subtitles.build_vtt(job.get("cues") or []), media_type="text/vtt")


@app.get("/api/jobs/{job_id}/output")
def output(job_id: str, download: bool = False) -> FileResponse:
    job = _job_or_404(job_id)
    if not job.get("output"):
        raise HTTPException(404, "ยังไม่มีไฟล์ที่เรนเดอร์เสร็จ")
    p = jobs.job_dir(job_id) / job["output"]
    if not p.is_file():
        raise HTTPException(404, "ไม่พบไฟล์")
    stem = Path(job.get("filename") or "reel").stem[:60] or "reel"
    name = f"{stem}_master.mp4"
    if download:
        return FileResponse(p, media_type="video/mp4", filename=name)
    return FileResponse(p, media_type="video/mp4")


@app.get("/api/jobs/{job_id}/log")
def render_log(job_id: str) -> PlainTextResponse:
    _job_or_404(job_id)
    p = jobs.job_dir(job_id) / "render.log"
    return PlainTextResponse(p.read_text(errors="replace")[-20000:] if p.is_file() else "")


# ---------- model comparison ----------

@app.get("/api/bench/testset")
def bench_testset() -> dict:
    return {"context": bench.BUILTIN_CONTEXT, "items": bench.builtin_items()}


@app.get("/api/bench")
def bench_runs() -> list[dict]:
    out = []
    for j in jobs.list_recent(50):
        if j.get("kind") != "bench":
            continue
        out.append({
            "id": j["id"], "created": j.get("created"), "models": j.get("models"),
            "source_name": j.get("source_name"), "state": (j.get("task") or {}).get("state"),
            "busy": (j.get("task") or {}).get("state") in ("queued", "running"),
        })
    return out[:20]


class BenchRequest(BaseModel):
    models: list[str] = Field(min_length=1, max_length=8)
    source: str = "builtin"  # "builtin" or a job id whose cues should be used
    context: str | None = None


@app.post("/api/bench")
def start_bench(req: BenchRequest) -> dict:
    models = list(dict.fromkeys(m.strip() for m in req.models if m.strip()))
    if not models:
        raise HTTPException(400, "เลือกโมเดลอย่างน้อย 1 ตัว")
    if req.source == "builtin":
        items, name = bench.builtin_items(), "ชุดทดสอบมาตรฐาน"
        context = bench.BUILTIN_CONTEXT if req.context is None else req.context
    else:
        src = _job_or_404(req.source)
        cues = [c for c in src.get("cues") or [] if c.get("text", "").strip()]
        if not cues:
            raise HTTPException(400, "งานนี้ยังไม่มีซับ")
        items = [{"id": i + 1, "text": "".join(x.strip() for x in c["text"].split("\n"))} for i, c in enumerate(cues)]
        name = f"ซับจาก {src.get('filename') or src['id']}"
        context = src.get("hint", "") if req.context is None else req.context
    job_id = jobs.new_id()
    jobs.save({
        "id": job_id, "kind": "bench", "created": time.time(), "models": models,
        "source": req.source, "source_name": name, "context": context.strip(),
        "items": items, "results": {},
        "task": {"name": None, "state": "idle", "progress": 0, "message": "", "error": ""},
    })
    jobs.submit(job_id, "bench", bench.nodes_for(models), bench.bench_task)
    return jobs.load(job_id)


app.mount("/", StaticFiles(directory=config.APP_DIR / "static", html=True), name="static")
