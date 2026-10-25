import pytest

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
