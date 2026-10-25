import time

import pytest
from fastapi.testclient import TestClient

from app import main
from app.services import transcriber
from tests.test_render import make_clip

FAKE = {"segments": [{"text": " วันนี้กุ๊กไก่แอบหนีเที่ยว", "start": 0.1, "end": 1.8, "words": [
    {"word": " วันนี้", "start": 0.1, "end": 0.5}, {"word": "กุ๊ก", "start": 0.5, "end": 0.8},
    {"word": "ไก่", "start": 0.8, "end": 1.0}, {"word": "แอบหนีเที่ยว", "start": 1.0, "end": 1.8}]}]}


def wait(client, job_id, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["task"]["state"] in ("done", "error"):
            return job
        time.sleep(0.2)
    raise AssertionError("task timed out")


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(transcriber, "transcribe", lambda wav, hint="", model=None, vad=True: FAKE)
    with TestClient(main.app) as c:
        yield c


def test_full_flow(client, tmp_path):
    src = make_clip(tmp_path / "clip.mp4", size="720x1280")
    with src.open("rb") as f:
        r = client.post("/api/jobs", files={"file": ("clip.mp4", f, "video/mp4")}, data={"hint": "กุ๊กไก่"})
    assert r.status_code == 200, r.text
    job = r.json()
    jid = job["id"]
    assert job["info"]["width"] == 720

    r = client.post(f"/api/jobs/{jid}/transcribe", json={"proofread": False})
    assert r.status_code == 200
    job = wait(client, jid)
    assert job["task"]["state"] == "done", job["task"]
    assert job["cues"] and "กุ๊กไก่" in job["cues"][0]["text"]
    nodes = {n["key"]: n for n in job["task"]["steps"]}
    assert [n["key"] for n in job["task"]["steps"]] == ["extract", "whisper", "segment", "proofread"]
    assert all(nodes[k]["state"] == "done" and nodes[k]["ended"] >= nodes[k]["started"] for k in ("extract", "whisper", "segment"))
    assert nodes["proofread"]["state"] == "skipped"
    assert job["task"]["progress"] == 1.0 and "now" in job

    cues = job["cues"] + [{"start": 1.85, "end": 2.0, "text": "เพิ่มเอง 😺"}]
    r = client.put(f"/api/jobs/{jid}/cues", json={"cues": cues, "style": {"size": 90}})
    assert r.status_code == 200
    assert len(r.json()["cues"]) == 2 and r.json()["style"]["size"] == 90

    assert client.get(f"/api/jobs/{jid}/subs.vtt").text.startswith("WEBVTT")

    r = client.post(f"/api/jobs/{jid}/render", json={"cues": cues, "render_options": {"fit": "crop"}})
    assert r.status_code == 200
    job = wait(client, jid, 120)
    assert job["task"]["state"] == "done", job["task"]
    assert job["render_meta"]["variant"] == "cpu"
    nodes = {n["key"]: n for n in job["task"]["steps"]}
    assert nodes["hw"]["state"] == "warn"  # no iGPU in tests
    assert nodes["render"]["state"] == "done" and nodes["render"]["detail"] == "CPU (libx264)"
    assert nodes["render"]["device"] == "CPU"
    assert nodes["finish"]["state"] == "done" and "MB" in nodes["finish"]["detail"]
    r = client.get(f"/api/jobs/{jid}/output?download=1")
    assert r.status_code == 200 and r.headers["content-type"] == "video/mp4"
    assert "attachment" in r.headers["content-disposition"]

    r = client.get(f"/api/jobs/{jid}/source", headers={"Range": "bytes=0-99"})
    assert r.status_code == 206  # seeking works on the phone

    assert client.delete(f"/api/jobs/{jid}").status_code == 200
    assert client.get(f"/api/jobs/{jid}").status_code == 404


def test_rejects_non_video(client):
    r = client.post("/api/jobs", files={"file": ("a.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_system_reports_offline_services(client):
    s = client.get("/api/system").json()
    assert s["whisper"]["ok"] is False and s["ollama"]["ok"] is False
    assert s["render"]["mode"] == "cpu"
    assert any(f["family"] == "Kanit" for f in s["fonts"])


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Reel Enhancer" in r.text


def test_ollama_failure_is_a_warning_not_an_error(client, tmp_path):
    src = make_clip(tmp_path / "clip.mp4", size="720x1280")
    with src.open("rb") as f:
        jid = client.post("/api/jobs", files={"file": ("clip.mp4", f, "video/mp4")}).json()["id"]
    client.post(f"/api/jobs/{jid}/transcribe", json={"proofread": True})
    job = wait(client, jid)
    assert job["task"]["state"] == "done"
    node = job["task"]["steps"][-1]
    assert node["key"] == "proofread" and node["state"] == "warn"
    assert job["cues"] and job["proofread_error"]


def test_failed_step_is_marked_error(client, tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("whisper down")
    monkeypatch.setattr(transcriber, "transcribe", boom)
    src = make_clip(tmp_path / "clip.mp4", size="720x1280")
    with src.open("rb") as f:
        jid = client.post("/api/jobs", files={"file": ("clip.mp4", f, "video/mp4")}).json()["id"]
    client.post(f"/api/jobs/{jid}/transcribe", json={})
    job = wait(client, jid)
    states = {n["key"]: n["state"] for n in job["task"]["steps"]}
    assert job["task"]["state"] == "error" and "whisper down" in job["task"]["error"]
    assert states == {"extract": "done", "whisper": "error", "segment": "pending", "proofread": "pending"}
