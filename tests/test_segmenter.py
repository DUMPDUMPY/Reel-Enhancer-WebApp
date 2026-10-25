from app.services.segmenter import build_cues, fix_timing
from app.services.thai_text import display_width


def _seg(tokens, start, end, p0=None, p1=None):
    """Fake whisper segment; token times on [p0, p1] like whisper.cpp with VAD."""
    p0 = start if p0 is None else p0
    p1 = end if p1 is None else p1
    step = (p1 - p0) / len(tokens)
    return {
        "text": "".join(tokens), "start": start, "end": end,
        "words": [{"word": t, "start": p0 + i * step, "end": p0 + (i + 1) * step} for i, t in enumerate(tokens)],
    }


TOKENS = [" วันนี้", "กุ๊ก", "ไก่", "แอบ", "หนี", "เที่ยว", "อีก", "แล้ว", " แต่", "ดัน", "เจอ",
          "เจ้า", "เหมียว", "ดัก", "รอ", "อยู่", "หน้า", "บ้าน", "!"]


def test_cues_are_short_and_keep_all_words():
    cues = build_cues({"segments": [_seg(TOKENS, 0.5, 5.4)]}, ("กุ๊กไก่", "เจ้าเหมียว"))
    assert len(cues) >= 2
    joined = "".join(c["text"].replace("\n", "") for c in cues)
    assert joined.replace(" ", "") == "".join(TOKENS).replace(" ", "")
    for c in cues:
        lines = c["text"].split("\n")
        assert len(lines) <= 2
        assert all(display_width(line) <= 20 for line in lines)
        assert c["end"] > c["start"]


def test_token_times_are_mapped_onto_vad_segment_times():
    # whisper.cpp leaves token times on the VAD-trimmed timeline (0..2 s),
    # but the segment really sits at 10..12 s.
    cues = build_cues({"segments": [_seg([" สวัสดี", "ครับ"], 10.0, 12.0, p0=0.0, p1=2.0)]})
    assert cues[0]["start"] == 10.0


def test_hallucinations_and_music_are_dropped():
    result = {"segments": [
        {"text": " ขอบคุณที่รับชม", "start": 0, "end": 2, "words": []},
        {"text": " [เพลง]", "start": 2, "end": 4, "words": []},
        _seg([" สวัสดี"], 5, 6),
    ]}
    cues = build_cues(result)
    assert [c["text"] for c in cues] == ["สวัสดี"]


def test_long_pause_starts_a_new_cue():
    seg = {"text": " หนึ่ง สอง", "start": 0, "end": 5, "words": [
        {"word": " หนึ่ง", "start": 0.0, "end": 0.8},
        {"word": " สอง", "start": 4.0, "end": 5.0},
    ]}
    cues = build_cues({"segments": [seg]})
    assert [c["text"] for c in cues] == ["หนึ่ง", "สอง"]


def test_fix_timing_removes_overlap_and_sets_minimum():
    cues = fix_timing([{"start": 0, "end": 0.3, "text": "a"}, {"start": 0.5, "end": 3, "text": "b"}])
    assert cues[0]["end"] <= cues[1]["start"]
    assert cues[1]["end"] - cues[1]["start"] >= 1.0
