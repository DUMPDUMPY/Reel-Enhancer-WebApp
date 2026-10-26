import pytest

from app.services import proofreader
from app.services.proofreader import validate

SENT = [{"id": 1, "text": "กุกไก่แอบหนีเที่ยว"}, {"id": 2, "text": "เจอเจ้าเหมียว"}]


def test_keeps_small_spelling_fixes():
    out = validate(SENT, [{"id": 1, "text": "กุ๊กไก่แอบหนีเที่ยว"}, {"id": 2, "text": "เจอเจ้าเหมียว"}])
    assert out == [{"id": 1, "before": "กุกไก่แอบหนีเที่ยว", "after": "กุ๊กไก่แอบหนีเที่ยว"}]


def test_rejects_answer_with_missing_or_extra_items():
    with pytest.raises(ValueError):
        validate(SENT, [{"id": 1, "text": "x"}])
    with pytest.raises(ValueError):
        validate(SENT, [{"id": 1, "text": "a"}, {"id": 2, "text": "b"}, {"id": 3, "text": "c"}])


def test_ignores_rewrites():
    out = validate(SENT, [{"id": 1, "text": "วันนี้น้องกุ๊กไก่แอบออกไปเที่ยวเล่นนอกบ้านอีกแล้วนะ"}, {"id": 2, "text": "เจอเจ้าเหมียว"}])
    assert out == []


def test_proofread_in_batches_skips_a_failed_batch(monkeypatch):
    calls = []

    def fake_ask(items, context="", model=None, retries=1):
        calls.append([it["id"] for it in items])
        if len(calls) == 2:
            raise RuntimeError("timed out")
        return {"suggestions": [{"id": items[0]["id"], "before": "a", "after": "b"}]}

    monkeypatch.setattr(proofreader, "ask_validated", fake_ask)
    cues = [{"id": i, "text": f"ท่อน{i}"} for i in range(1, 26)]
    seen = []
    suggestions, errors = proofreader.proofread(cues, batch=10, on_progress=lambda d, t: seen.append((d, t)))
    assert calls == [list(range(1, 11)), list(range(11, 21)), list(range(21, 26))]
    assert [s["id"] for s in suggestions] == [1, 21] and len(errors) == 1 and "ชุด 2" in errors[0]
    assert seen == [(1, 3), (2, 3), (3, 3)]


def test_proofread_raises_when_every_batch_fails(monkeypatch):
    monkeypatch.setattr(proofreader, "ask_validated", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("timed out")))
    with pytest.raises(RuntimeError):
        proofreader.proofread([{"id": 1, "text": "ก"}], batch=10)
