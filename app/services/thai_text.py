"""Thai-aware text helpers: display width, word tokenizing and line wrapping.

Thai has no spaces between words, so neither libass nor the browser can find a
safe place to break a line. We tokenize with PyThaiNLP and only break between
words, never inside one.
"""

import re
import unicodedata
from functools import lru_cache

from pythainlp.corpus import thai_words
from pythainlp.tokenize import word_tokenize
from pythainlp.util import Trie, dict_trie


def display_width(text: str) -> int:
    """Number of horizontal glyph slots: Thai vowels/tone marks above or below don't count."""
    return sum(1 for ch in text if unicodedata.category(ch) not in ("Mn", "Me", "Cf"))


@lru_cache(maxsize=1)
def _base_words() -> frozenset[str]:
    return frozenset(thai_words())


@lru_cache(maxsize=32)
def _trie(extra_words: tuple[str, ...]) -> Trie:
    return dict_trie(set(_base_words()) | set(extra_words))


def parse_hint_words(hint: str) -> tuple[str, ...]:
    """Proper nouns from the hint box ("กุ๊กไก่, เจ้าเหมียว") so they are never split."""
    words = (w.strip() for w in re.split(r"[,\n،、;|/]+", hint or ""))
    return tuple(sorted({w for w in words if w and " " not in w and len(w) <= 40}))


def tokenize(text: str, extra_words: tuple[str, ...] = ()) -> list[str]:
    """Split into words, keeping whitespace tokens so "".join(result) == text."""
    if not text:
        return []
    return word_tokenize(text, custom_dict=_trie(extra_words), engine="newmm", keep_whitespace=True)


def _balanced_split(words: list[str]) -> tuple[str, str]:
    """Split one line into two at the word boundary that best balances their widths."""
    best = None
    for i in range(1, len(words)):
        left, right = "".join(words[:i]).strip(), "".join(words[i:]).strip()
        if not left or not right:
            continue
        score = abs(display_width(left) - display_width(right))
        # Prefer breaking at a space (Thai uses spaces between phrases).
        if words[i - 1].isspace() or words[i].isspace():
            score -= 3
        if best is None or score < best[0]:
            best = (score, left, right)
    if best is None:
        joined = "".join(words).strip()
        return joined, ""
    return best[1], best[2]


def wrap_lines(text: str, max_width: int, max_lines: int = 2, extra_words: tuple[str, ...] = ()) -> list[str]:
    """Wrap a cue into at most `max_lines` lines, breaking only between words.

    Existing line breaks typed by the user are respected; a line that is still
    too wide gets wrapped again. If the text needs more than `max_lines`, the
    extra lines are kept (the editor warns the user instead of dropping words).
    """
    out: list[str] = []
    for raw in text.replace("\\N", "\n").split("\n"):
        raw = raw.strip()
        if not raw:
            continue
        if display_width(raw) <= max_width:
            out.append(raw)
            continue
        words = tokenize(raw, extra_words)
        if display_width(raw) <= max_width * 2 and len(out) + 2 <= max_lines:
            left, right = _balanced_split(words)
            if display_width(left) <= max_width and display_width(right) <= max_width:
                out.extend(x for x in (left, right) if x)
                continue
        line = ""
        for w in words:
            candidate = line + w
            if line and display_width(candidate.strip()) > max_width:
                out.append(line.strip())
                line = w.lstrip()
            else:
                line = candidate
        if line.strip():
            out.append(line.strip())
    return out
