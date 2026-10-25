from app.services.thai_text import display_width, parse_hint_words, tokenize, wrap_lines


def test_display_width_ignores_marks_above_and_below():
    assert display_width("กุ๊กไก่") == 4  # ก ก ไ ก
    assert display_width("ที่") == 1


def test_tokenize_keeps_text_and_hint_words():
    text = "กุ๊กไก่ชอบเจ้าเหมียว"
    words = tokenize(text, ("กุ๊กไก่", "เจ้าเหมียว"))
    assert "".join(words) == text
    assert "กุ๊กไก่" in words and "เจ้าเหมียว" in words


def test_parse_hint_words():
    assert parse_hint_words("กุ๊กไก่, เจ้าเหมียว\nหมูจิ๋ว") == ("กุ๊กไก่", "หมูจิ๋ว", "เจ้าเหมียว")


def test_wrap_never_splits_inside_a_word():
    text = "วันนี้กุ๊กไก่แอบหนีเที่ยวอีกแล้ว"
    lines = wrap_lines(text, 12, 2, ("กุ๊กไก่",))
    assert "".join(lines) == text
    assert len(lines) == 2
    assert all(display_width(line) <= 12 for line in lines)
    assert not any(line.endswith("กุ๊ก") for line in lines)


def test_wrap_respects_user_line_breaks():
    assert wrap_lines("สวัสดี\nครับ", 20) == ["สวัสดี", "ครับ"]
