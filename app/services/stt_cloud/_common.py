"""Helpers shared by the cloud speech-to-text providers."""

import re

_SENTENCE_END = re.compile(r"[.!?…。！？]$")


def parse_offset(value) -> float | None:
    """Time offsets from Google/Gemini are strings like "0.100s"."""
    if value is None:
        return None
    text = str(value).strip().removesuffix("s")
    try:
        return float(text)
    except ValueError:
        return None


def make_segment(cur: list[dict], source: str) -> dict | None:
    """Word dicts -> a whisper-style segment, or None when nothing is left.

    Edge spacing is trimmed so the segment start/end match the token span (the
    segmenter rescales token times onto the segment, which would otherwise shift
    them)."""
    while cur and not cur[0]["word"].strip():
        cur.pop(0)
    while cur and not cur[-1]["word"].strip():
        cur.pop()
    if not cur:
        return None
    return {
        "text": "".join(w["word"] for w in cur),
        "start": cur[0]["start"], "end": cur[-1]["end"], "words": cur,
        # Cloud providers mark real speech with a source, so the segmenter must
        # not drop lines like "อย่าลืมกดติดตาม" as Whisper outro hallucinations.
        "source": source,
    }


def words_to_segments(words: list[dict], source: str, gap_split: float) -> list[dict]:
    """Plain word dicts (no spacing/audio events) -> whisper-style segments.

    Breaks at sentence-final punctuation and at gaps >= gap_split. For Thai the
    gap bar is deliberately high (>= 1 s): Gemini and Google may return single
    characters or clusters, and a held note in a song leaves a smaller gap
    *inside* a word — those pauses are left to the segmenter, which splits cues
    per PyThaiNLP word."""
    segments, cur = [], []
    for i, w in enumerate(words):
        cur.append(w)
        gap_ahead = i + 1 < len(words) and words[i + 1]["start"] - w["end"] >= gap_split
        if _SENTENCE_END.search(w["word"]) or gap_ahead:
            seg = make_segment(cur, source)
            if seg:
                segments.append(seg)
            cur = []
    seg = make_segment(cur, source)
    if seg:
        segments.append(seg)
    return segments
