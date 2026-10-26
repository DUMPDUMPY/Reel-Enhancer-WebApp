import time

import pytest
from fastapi.testclient import TestClient

from app import jobs, main
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
    assert s["whisper"]["ok"] is False and s["llm"]["ok"] is False
    by_key = {p["key"]: p for p in s["llm"]["providers"]}
    assert by_key["zai"]["enabled"] is False and "API key" in by_key["zai"]["status"]
    assert s["llm"]["providers"][-1]["key"] == "ollama"  # cloud providers listed first
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


def test_retry_without_vad_when_vad_finds_nothing(monkeypatch, tmp_path):
    calls = []

    def fake(wav, hint="", model=None, vad=True):
        calls.append(vad)
        return {"segments": []} if vad else FAKE  # VAD dropped the singing

    monkeypatch.setattr(transcriber, "transcribe", fake)
    with TestClient(main.app) as c:
        src = make_clip(tmp_path / "song.mp4", size="720x1280")
        with src.open("rb") as f:
            jid = c.post("/api/jobs", files={"file": ("song.mp4", f, "video/mp4")}).json()["id"]
        c.post(f"/api/jobs/{jid}/transcribe", json={"vad": True, "model": "ggml-large-v3-q5_0.bin"})
        job = wait(c, jid)
    assert calls == [True, False]
    assert job["cues"] and job["transcribe_meta"]["vad"] is False
    node = next(s for s in job["task"]["steps"] if s["key"] == "whisper")
    assert node["state"] == "done" and "ไม่ใช้ VAD" in node["detail"]


def test_list_shows_size_and_delete_frees_it(client, tmp_path):
    src = make_clip(tmp_path / "clip.mp4", size="720x1280")
    with src.open("rb") as f:
        jid = client.post("/api/jobs", files={"file": ("clip.mp4", f, "video/mp4")}).json()["id"]
    row = next(j for j in client.get("/api/jobs").json() if j["id"] == jid)
    assert row["size"] > 0 and row["busy"] is False
    assert client.delete(f"/api/jobs/{jid}").json() == {"ok": True}
    assert jid not in [j["id"] for j in client.get("/api/jobs").json()]
    assert not jobs.job_dir(jid).exists()


def test_transcribe_defaults_to_cloud_stt_when_key_set(client, tmp_path, monkeypatch):
    from app import config
    from app.services import stt_cloud
    monkeypatch.setattr(config, "ELEVENLABS_API_KEY", "k")
    seen = {}
    monkeypatch.setattr(stt_cloud, "transcribe", lambda wav, hint="", model=None: seen.setdefault("model", model) and FAKE)
    s = client.get("/api/system").json()
    assert s["stt_default"] == "elevenlabs:scribe_v2"
    src = make_clip(tmp_path / "clip.mp4", size="720x1280")
    with src.open("rb") as f:
        jid = client.post("/api/jobs", files={"file": ("clip.mp4", f, "video/mp4")}).json()["id"]
    client.post(f"/api/jobs/{jid}/transcribe", json={})
    assert wait(client, jid)["task"]["state"] == "done"
    assert seen["model"] == "elevenlabs:scribe_v2"


def test_ollama_loaded_and_unload(client, monkeypatch):
    import httpx
    from app.services import llm
    loaded = [{"name": "gemma3:4b", "size": 4_000_000_000, "size_vram": 0}]
    unloaded = []

    def fake_get(url, timeout=None):
        assert url.endswith("/api/ps")
        return httpx.Response(200, json={"models": loaded}, request=httpx.Request("GET", url))

    def fake_post(url, json=None, timeout=None):
        assert json["keep_alive"] == 0
        unloaded.append(json["model"])
        loaded[:] = [m for m in loaded if m["name"] != json["model"]]
        return httpx.Response(200, json={}, request=httpx.Request("POST", url))

    monkeypatch.setattr(llm.httpx, "get", fake_get)
    monkeypatch.setattr(llm.httpx, "post", fake_post)
    info = client.get("/api/ollama/loaded").json()
    assert info["ok"] and info["models"][0]["name"] == "gemma3:4b"
    info = client.post("/api/ollama/unload", json={}).json()  # no model: stop everything
    assert unloaded == ["gemma3:4b"] and info["models"] == []


def test_ollama_offline_unload_is_503(client):
    assert client.get("/api/ollama/loaded").json()["ok"] is False
    assert client.post("/api/ollama/unload", json={"model": "x"}).status_code == 503


def test_ui_files_are_revalidated(client):
    for path in ("/", "/app.js", "/style.css"):
        assert client.get(path).headers["cache-control"] == "no-cache"
    assert "no-cache" not in client.get("/api/jobs").headers.get("cache-control", "")
