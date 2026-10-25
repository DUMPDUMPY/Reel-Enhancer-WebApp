"""Turn whisper.cpp verbose_json output into short, readable 1-2 line Thai cues.

Segmentation is deterministic: whisper's token timestamps give the timing,
PyThaiNLP gives the word boundaries, and we pack whole words into cues. The LLM
proofreader never moves words between cues, so timing can't drift.
"""

import re
from dataclasses import dataclass

from app import config
from app.services.thai_text import display_width, tokenize, wrap_lines

# Phrases Whisper tends to invent over music or silence in Thai/English clips.
HALLUCINATIONS = [
    "ขอบคุณที่รับชม",
    "ขอบคุณสําหรับการรับชม",
    "ขอบคุณสำหรับการรับชม",
    "ฝากกดไลค์",
    "กดติดตาม",
    "กดซับ",
    "ซับไตเติ้ลโดย",
    "คำบรรยายโดย",
    "โปรดติดตามตอนต่อไป",
    "subtitles by",
    "thanks for watching",
    "thank you for watching",
    "please subscribe",
]
_NON_SPEECH = re.compile(r"^[\s\[\(（【♪♫*#…\.\-–—]*(music|ดนตรี|เพลง|เสียงดนตรี|blank_audio|silence)?[\s\]\)）】♪♫*#…\.\-–—]*$", re.I)


@dataclass
class Word:
    text: str
    start: float
    end: float
    seg_index: int


def _is_hallucination(text: str, phrases: bool = True) -> bool:
    t = re.sub(r"\s+", " ", text).strip().lower()
    if not t or _NON_SPEECH.match(t):
        return True
    return phrases and any(h in t for h in HALLUCINATIONS) and display_width(t) <= 40


def _segment_tokens(seg: dict) -> list[tuple[str, float, float]]:
    """(text, start, end) per token, rescaled onto the segment's real timeline.

    With VAD enabled whisper.cpp maps segment times back to the original audio
    but leaves token times on the silence-stripped timeline, so we stretch the
    token times linearly to fit the segment. Without VAD this is a no-op.
    """
    s0, s1 = float(seg.get("start", 0.0)), float(seg.get("end", 0.0))
    raw = []
    for w in seg.get("words") or []:
        text = w.get("word", "")
        if not text or re.fullmatch(r"\[_?[A-Z_0-9]+_?\]|<\|.*?\|>", text.strip()):
            continue
        raw.append((text, w.get("start"), w.get("end")))
    if not raw:
        text = seg.get("text", "")
        return [(text, s0, s1)] if text.strip() else []

    timed = [(t, a, b) for t, a, b in raw if a is not None and b is not None]
    if len(timed) == len(raw):
        p0 = min(float(a) for _, a, _ in timed)
        p1 = max(float(b) for _, _, b in timed)
        if p1 - p0 > 0.01 and s1 > s0:
            k = (s1 - s0) / (p1 - p0)
            return [(t, s0 + (float(a) - p0) * k, s0 + (float(b) - p0) * k) for t, a, b in raw]

    # No usable token timing: spread the segment duration by text width.
    total = sum(max(display_width(t), 1) for t, _, _ in raw)
    out, cursor = [], s0
    for t, _, _ in raw:
        dur = (s1 - s0) * max(display_width(t), 1) / total
        out.append((t, cursor, cursor + dur))
        cursor += dur
    return out


def words_from_whisper(result: dict, extra_words: tuple[str, ...] = ()) -> list[Word]:
    """Re-tokenize each whisper segment into real Thai words and time each word."""
    words: list[Word] = []
    for seg_index, seg in enumerate(result.get("segments") or []):
        if _is_hallucination(seg.get("text", ""), phrases=seg.get("source") != "elevenlabs"):
            continue
        tokens = _segment_tokens(seg)
        text = "".join(t for t, _, _ in tokens)
        if not text.strip():
            continue
        # char offset -> (start, end) of the whisper token covering it
        char_times: list[tuple[float, float]] = []
        for t, a, b in tokens:
            char_times.extend([(a, b)] * len(t))

        offset = 0
        for w in tokenize(text, extra_words):
            first, last = offset, offset + len(w) - 1
            offset += len(w)
            if w.isspace():
                words.append(Word(" ", char_times[first][0], char_times[last][1], seg_index))
                continue
            words.append(Word(w, char_times[first][0], char_times[last][1], seg_index))
        # A space between whisper segments so the packer can break there.
        words.append(Word(" ", words[-1].end, words[-1].end, seg_index))
    return words


def _cue_text(words: list[Word]) -> str:
    return re.sub(r"\s+", " ", "".join(w.text for w in words)).strip()


def pack_cues(
    words: list[Word],
    max_width: int = config.MAX_LINE_WIDTH,
    max_lines: int = config.MAX_LINES,
    max_seconds: float = config.MAX_CUE_SECONDS,
    pause_split: float = config.PAUSE_SPLIT_SECONDS,
    extra_words: tuple[str, ...] = (),
) -> list[dict]:
    capacity = max_width * max_lines
    groups: list[list[Word]] = []
    cur: list[Word] = []

    def close():
        nonlocal cur
        while cur and cur[-1].text.isspace():
            cur.pop()
        if cur:
            groups.append(cur)
        cur = []

    for w in words:
        if w.text.isspace():
            if cur:
                cur.append(w)
                # Thai writes a space between phrases: a natural place to end a cue
                # once it is reasonably full.
                if display_width(_cue_text(cur)) >= capacity * 0.5:
                    close()
            continue
        if cur:
            prev = next(x for x in reversed(cur) if not x.text.isspace())
            gap = w.start - prev.end
            candidate = _cue_text(cur + [w])
            too_wide = display_width(candidate) > capacity or len(wrap_lines(candidate, max_width, max_lines, extra_words)) > max_lines
            too_long = w.end - cur[0].start > max_seconds
            new_sentence = w.seg_index != prev.seg_index and display_width(_cue_text(cur)) >= max_width * 0.5
            if too_wide or too_long or gap >= pause_split or new_sentence:
                close()
        cur.append(w)
    close()

    cues = []
    for g in groups:
        text = "\n".join(wrap_lines(_cue_text(g), max_width, max_lines, extra_words))
        cues.append({"start": g[0].start, "end": g[-1].end, "text": text})
    return fix_timing(cues)


def fix_timing(cues: list[dict], min_seconds: float = config.MIN_CUE_SECONDS, tail: float = 0.2) -> list[dict]:
    """Sort, give every cue a minimum on-screen time and remove overlaps."""
    cues = sorted((dict(c) for c in cues), key=lambda c: c["start"])
    for i, c in enumerate(cues):
        c["start"] = max(0.0, round(float(c["start"]), 3))
        nxt = cues[i + 1]["start"] if i + 1 < len(cues) else None
        end = max(float(c["end"]) + tail, c["start"] + min_seconds)
        if nxt is not None:
            end = min(end, nxt - 0.04)
        c["end"] = round(max(end, c["start"] + 0.2), 3)
        c["id"] = i + 1
    return cues


def build_cues(result: dict, extra_words: tuple[str, ...] = ()) -> list[dict]:
    return pack_cues(words_from_whisper(result, extra_words), extra_words=extra_words)
