"""Job storage (one folder + job.json per upload) and a single-worker task queue.

Only one heavy task (transcribe / render) runs at a time so the iGPU and the
NAS are never overloaded; the phone polls GET /api/jobs/{id} for progress.
"""

import json
import logging
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from app import config

log = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="worker")
_lock = threading.RLock()


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def job_dir(job_id: str) -> Path:
    if not job_id.isalnum() or len(job_id) > 32:
        raise KeyError(job_id)
    return config.JOBS_DIR / job_id


def load(job_id: str) -> dict:
    p = job_dir(job_id) / "job.json"
    if not p.is_file():
        raise KeyError(job_id)
    with _lock:
        return json.loads(p.read_text())


def save(job: dict) -> dict:
    d = job_dir(job["id"])
    d.mkdir(parents=True, exist_ok=True)
    job["updated"] = time.time()
    tmp = d / "job.json.tmp"
    with _lock:
        tmp.write_text(json.dumps(job, ensure_ascii=False, indent=1))
        tmp.replace(d / "job.json")
    return job


def update(job_id: str, **fields: Any) -> dict:
    with _lock:
        job = load(job_id)
        job.update(fields)
        return save(job)


def set_task(job_id: str, name: str | None, state: str, progress: float = 0.0, message: str = "", error: str = "") -> None:
    with _lock:
        job = load(job_id)
        task = job.get("task") or {}
        task.update(name=name, state=state, progress=round(progress, 3), message=message, error=error)
        if state in ("done", "error"):
            task["ended"] = time.time()
        job["task"] = task
        save(job)


# ---------- pipeline nodes ----------
# Each task is a chain of nodes (key, label, device, weight). The phone draws them
# as a pipeline and shows which one is running, on which hardware, and for how long.

def _new_task(name: str, nodes: list[tuple[str, str, str, float]]) -> dict:
    return {
        "name": name, "state": "queued", "progress": 0.0, "message": "รอคิว", "error": "",
        "queued": time.time(), "started": None, "ended": None,
        "steps": [
            {"key": k, "label": label, "device": device, "weight": w, "state": "pending",
             "progress": 0.0, "detail": "", "started": None, "ended": None}
            for k, label, device, w in nodes
        ],
    }


def _recompute(task: dict) -> None:
    steps = task.get("steps") or []
    total = sum(s["weight"] for s in steps) or 1
    done = 0.0
    for s in steps:
        if s["state"] in ("done", "skipped", "warn"):
            done += s["weight"]
        elif s["state"] == "running":
            done += s["weight"] * s["progress"]
    task["progress"] = round(done / total, 3)
    running = next((s for s in steps if s["state"] == "running"), None)
    if running:
        task["message"] = running["label"] + (f" · {running['detail']}" if running["detail"] else "")


def step_update(job_id: str, key: str, state: str | None = None, progress: float | None = None,
                detail: str | None = None, device: str | None = None) -> None:
    """Update one node: state pending|running|done|skipped|warn|error."""
    with _lock:
        job = load(job_id)
        task = job.get("task") or {}
        for s in task.get("steps") or []:
            if s["key"] != key:
                continue
            now = time.time()
            if state:
                if state == "running" and not s["started"]:
                    s["started"] = now
                if state in ("done", "skipped", "warn", "error"):
                    s["ended"] = now
                    if state == "done":
                        s["progress"] = 1.0
                s["state"] = state
            if progress is not None:
                s["progress"] = round(min(max(progress, 0.0), 1.0), 3)
            if detail is not None:
                s["detail"] = detail
            if device is not None:
                s["device"] = device
            break
        _recompute(task)
        job["task"] = task
        save(job)


@contextmanager
def step(job_id: str, key: str, detail: str = ""):
    """with step(job_id, "whisper"): ... -> running, then done (or error if it raises)."""
    step_update(job_id, key, "running", 0.0, detail)
    try:
        yield
    except Exception as e:
        step_update(job_id, key, "error", detail=str(e)[-300:])
        raise
    with _lock:
        state = next((s["state"] for s in load(job_id)["task"].get("steps", []) if s["key"] == key), None)
    if state == "running":  # the body may have already set skipped/warn
        step_update(job_id, key, "done")


def submit(job_id: str, name: str, nodes: list[tuple[str, str, str, float]], fn, *args) -> None:
    """Queue fn(job_id, *args); its state and nodes are tracked in job['task']."""
    update(job_id, task=_new_task(name, nodes))

    def run():
        with _lock:
            job = load(job_id)
            job["task"].update(state="running", started=time.time(), message="เริ่มทำงาน")
            save(job)
        try:
            fn(job_id, *args)
            with _lock:
                job = load(job_id)
                for s in job["task"]["steps"]:
                    if s["state"] == "pending":
                        s["state"] = "skipped"
                job["task"].update(state="done", progress=1.0, message="เสร็จแล้ว", ended=time.time())
                save(job)
        except Exception as e:  # report any failure to the phone instead of dying silently
            log.exception("task %s failed for job %s", name, job_id)
            set_task(job_id, name, "error", message="เกิดข้อผิดพลาด", error=str(e)[-1500:])

    _executor.submit(run)


def busy() -> bool:
    for p in config.JOBS_DIR.glob("*/job.json"):
        try:
            if json.loads(p.read_text()).get("task", {}).get("state") in ("queued", "running"):
                return True
        except (OSError, ValueError):
            continue
    return False


def recover_interrupted() -> None:
    """Mark tasks that were running when the container stopped as failed."""
    config.JOBS_DIR.mkdir(parents=True, exist_ok=True)
    for p in config.JOBS_DIR.glob("*/job.json"):
        try:
            job = json.loads(p.read_text())
            if job.get("task", {}).get("state") in ("queued", "running"):
                job["task"].update(state="error", error="งานถูกหยุดเพราะโปรแกรมรีสตาร์ต กรุณากดใหม่อีกครั้ง")
                for s in job["task"].get("steps") or []:
                    if s["state"] == "running":
                        s["state"] = "error"
                save(job)
        except (OSError, ValueError, KeyError):
            continue


def cleanup_old() -> int:
    """Delete job folders not touched for JOB_RETENTION_HOURS."""
    cutoff = time.time() - config.JOB_RETENTION_HOURS * 3600
    removed = 0
    if not config.JOBS_DIR.is_dir():
        return 0
    for d in config.JOBS_DIR.iterdir():
        meta = d / "job.json"
        try:
            ts = meta.stat().st_mtime if meta.exists() else d.stat().st_mtime
        except OSError:
            continue
        if ts < cutoff:
            shutil.rmtree(d, ignore_errors=True)
            removed += 1
    return removed


def list_recent(limit: int = 20) -> list[dict]:
    jobs = []
    if config.JOBS_DIR.is_dir():
        for p in config.JOBS_DIR.glob("*/job.json"):
            try:
                jobs.append(json.loads(p.read_text()))
            except (OSError, ValueError):
                continue
    jobs.sort(key=lambda j: j.get("created", 0), reverse=True)
    return jobs[:limit]
