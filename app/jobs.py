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
    update(job_id, task={"name": name, "state": state, "progress": round(progress, 3), "message": message, "error": error})


def submit(job_id: str, name: str, fn, *args) -> None:
    """Queue fn(job_id, *args); its state is tracked in job['task']."""
    set_task(job_id, name, "queued", message="รอคิว")

    def run():
        set_task(job_id, name, "running", message="กำลังทำงาน")
        try:
            fn(job_id, *args)
            set_task(job_id, name, "done", 1.0, message="เสร็จแล้ว")
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
