import time

import pytest
from fastapi.testclient import TestClient

from app import bench, main
from app.services import proofreader

STATS = {"total_s": 1.0, "load_s": 0.0, "prompt_s": 0.1, "eval_s": 0.9, "eval_tokens": 90, "tokens_per_s": 100.0, "thinking": False}
EXPECTED = {it["id"]: it["expected"] for it in bench.builtin_items()}


CALLS = {}


def fake_ask(items, context="", model=None):
    CALLS[model] = CALLS.get(model, 0) + 1
    if model == "flaky" and CALLS[model] == 1:  # drops items once, then answers properly
        return [{"id": items[0]["id"], "text": "x"}], STATS
    if model in ("good", "flaky"):
        return [{"id": it["id"], "text": EXPECTED[it["id"]]} for it in items], STATS
    if model == "lazy":
        return [{"id": it["id"], "text": it["text"]} for it in items], STATS
    if model == "broken":
        return [{"id": items[0]["id"], "text": "x"}], STATS
    if model == "rewriter":  # "fixes" slang and rewrites a line completely
        out = [{"id": it["id"], "text": EXPECTED[it["id"]]} for it in items]
        out[5]["text"] = "อย่าลืมกดติดตามกันด้วยนะ"  # slang น้า -> นะ: an over-edit
        out[0]["text"] = "ในวันนี้น้องกุ๊กไก่ได้แอบหนีออกไปเที่ยวเล่นข้างนอกอีกครั้งหนึ่งแล้ว"
        return out, STATS
    raise RuntimeError("model not found")


def test_score_counts_fixes_and_over_edits():
    items = bench.builtin_items()
    perfect = bench.score(items, EXPECTED)
    assert perfect["score"] == 100 and perfect["fixed"] == perfect["errors"] and perfect["over_edits"] == 0
    untouched = bench.score(items, {it["id"]: it["text"] for it in items})
    assert untouched["score"] == 0 and untouched["fixed"] == 0 and untouched["over_edits"] == 0


def test_wrong_fixes_cost_points():
    items = bench.builtin_items()
    finals = dict(EXPECTED)
    finals[5] = "เขาไม่ได้ตั้งจี้จะทำร้ายใคร"  # ตั้งจัย -> ตั้งจี้: changed, still wrong
    sc = bench.score(items, finals)
    assert sc["wrong_edits"] == 1 and sc["fixed"] == sc["errors"] - 1
    assert sc["score"] == round(100 * (sc["errors"] - 2) / sc["errors"])


def test_testset_has_errors_and_clean_lines():
    items = bench.builtin_items()
    assert sum(it["text"] != it["expected"] for it in items) >= 6
    assert sum(it["text"] == it["expected"] for it in items) >= 4


UNLOADED = []


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(proofreader, "ask", fake_ask)
    monkeypatch.setattr(proofreader, "unload", UNLOADED.append)
    with TestClient(main.app) as c:
        yield c


def wait(client, jid, timeout=30):
    end = time.time() + timeout
    while time.time() < end:
        run = client.get(f"/api/jobs/{jid}").json()
        if run["task"]["state"] in ("done", "error"):
            return run
        time.sleep(0.1)
    raise AssertionError("timeout")


def test_bench_run_compares_models(client):
    CALLS.clear()
    UNLOADED.clear()
    r = client.post("/api/bench", json={"models": ["good", "lazy", "broken", "rewriter", "missing", "flaky"]})
    assert r.status_code == 200, r.text
    run = wait(client, r.json()["id"])
    assert run["task"]["state"] == "done"
    res = run["results"]
    assert res["good"]["score"]["score"] == 100
    assert res["lazy"]["score"]["score"] == 0
    assert res["broken"]["ok"] is False and "ไม่ครบ" in res["broken"]["error"]
    assert UNLOADED == ["good", "lazy", "broken", "rewriter", "missing", "flaky"]  # freed after each model
    assert CALLS["broken"] == 2                    # retried once before giving up
    assert res["flaky"]["ok"] and res["flaky"]["attempts"] == 2 and res["flaky"]["score"]["score"] == 100
    assert res["flaky"]["failed_answers"][0]["answer"] == [{"id": 1, "text": "x"}]
    assert res["missing"]["ok"] is False
    rw = res["rewriter"]
    assert rw["rejected_rewrites"] == 1           # the full rewrite was thrown away
    assert rw["score"]["over_edits"] == 1         # น้า -> นะ counted against it
    assert rw["score"]["score"] < 100
    states = {n["label"]: n["state"] for n in run["task"]["steps"]}
    assert states == {"good": "done", "lazy": "done", "broken": "warn", "rewriter": "done", "missing": "warn", "flaky": "done"}

    # bench runs are listed separately from editing jobs
    assert any(b["id"] == run["id"] for b in client.get("/api/bench").json())
    assert all(j["id"] != run["id"] for j in client.get("/api/jobs").json())


def test_bench_validates_input(client):
    assert client.post("/api/bench", json={"models": []}).status_code == 422
    assert client.post("/api/bench", json={"models": ["good"], "source": "nope"}).status_code == 404
    t = client.get("/api/bench/testset").json()
    assert t["items"] and "กุ๊กไก่" in t["context"]
