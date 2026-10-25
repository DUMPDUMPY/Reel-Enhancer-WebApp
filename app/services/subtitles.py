"""Build the burn-in .ass file (and a WebVTT preview) from the edited cues."""

from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import pysubs2
from fontTools.ttLib import TTFont

from app import config
from app.services.thai_text import wrap_lines


@dataclass
class SubtitleStyle:
    font: str = "Kanit"
    size: int = 78
    color: str = "#FFFFFF"
    outline_color: str = "#000000"
    outline: float = 4
    shadow: float = 2
    margin_v: int = 400  # px above the bottom edge: clears TikTok / FB Reels captions
    margin_h: int = 90
    box: bool = False  # semi-transparent box instead of outline

    @classmethod
    def from_dict(cls, d: dict | None) -> "SubtitleStyle":
        d = d or {}
        s = cls()
        for k, v in d.items():
            if hasattr(s, k) and v is not None:
                setattr(s, k, type(getattr(s, k))(v))
        s.size = min(max(s.size, 30), 160)
        s.margin_v = min(max(s.margin_v, 0), config.OUTPUT_HEIGHT - 200)
        s.margin_h = min(max(s.margin_h, 0), config.OUTPUT_WIDTH // 3)
        return s

    def max_line_width(self) -> int:
        """Rough Thai glyphs per line for this font size (bold Thai ~0.58 em per glyph)."""
        usable = config.OUTPUT_WIDTH - 2 * self.margin_h
        return max(6, int(usable / (self.size * 0.58)))

    def to_dict(self) -> dict:
        return asdict(self)


@lru_cache(maxsize=64)
def _family(path: Path, mtime: float) -> str | None:
    try:
        font = TTFont(path, lazy=True)
        name = font["name"]
        # Prefer the typographic family (16) over the legacy family (1).
        rec = name.getName(16, 3, 1, 0x409) or name.getName(1, 3, 1, 0x409) or name.getName(1, 1, 0, 0)
        return str(rec) if rec else None
    except Exception:
        return None


def list_fonts() -> list[dict]:
    """[{family, file}] for every font the renderer (libass) will see."""
    fonts, seen = [], set()
    for d in config.FONT_DIRS:
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() not in (".ttf", ".otf"):
                continue
            fam = _family(p, p.stat().st_mtime)
            if fam and fam not in seen:
                seen.add(fam)
                fonts.append({"family": fam, "file": p.name, "dir": str(d)})
    return fonts


def font_file(name: str) -> Path | None:
    for d in config.FONT_DIRS:
        p = d / name
        if p.is_file() and p.parent == d:
            return p
    return None


def _ass_color(hex_color: str, alpha: int = 0) -> pysubs2.Color:
    h = hex_color.lstrip("#")
    if len(h) != 6:
        h = "FFFFFF"
    return pysubs2.Color(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), alpha)


def build_ass(cues: list[dict], style: SubtitleStyle, out_path: Path, extra_words: tuple[str, ...] = ()) -> Path:
    subs = pysubs2.SSAFile()
    subs.info["PlayResX"] = str(config.OUTPUT_WIDTH)
    subs.info["PlayResY"] = str(config.OUTPUT_HEIGHT)
    subs.info["WrapStyle"] = "2"  # we insert \N ourselves; libass can't wrap Thai
    subs.info["ScaledBorderAndShadow"] = "yes"

    st = pysubs2.SSAStyle()
    st.fontname = style.font
    st.fontsize = style.size
    st.bold = True
    st.primarycolor = _ass_color(style.color)
    st.alignment = pysubs2.Alignment.BOTTOM_CENTER
    st.marginv = style.margin_v
    st.marginl = st.marginr = style.margin_h
    if style.box:
        st.borderstyle = 3
        st.outline = 18  # box padding
        st.shadow = 0
        st.outlinecolor = _ass_color(style.outline_color, alpha=90)
        st.backcolor = _ass_color(style.outline_color, alpha=90)
    else:
        st.borderstyle = 1
        st.outline = style.outline
        st.shadow = style.shadow
        st.outlinecolor = _ass_color(style.outline_color)
        st.backcolor = pysubs2.Color(0, 0, 0, 110)
    subs.styles["Default"] = st

    width = style.max_line_width()
    for c in cues:
        text = (c.get("text") or "").strip()
        if not text:
            continue
        lines = wrap_lines(text, width, max_lines=config.MAX_LINES, extra_words=extra_words)
        safe = "\\N".join(line.replace("{", "(").replace("}", ")") for line in lines)
        subs.append(pysubs2.SSAEvent(start=int(float(c["start"]) * 1000), end=int(float(c["end"]) * 1000), text=safe))
    subs.sort()
    subs.save(str(out_path), format_="ass")
    return out_path


def _vtt_time(t: float) -> str:
    ms = int(round(max(t, 0) * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def build_vtt(cues: list[dict]) -> str:
    out = ["WEBVTT", ""]
    for c in sorted(cues, key=lambda c: float(c["start"])):
        text = (c.get("text") or "").strip()
        if text:
            out += [f"{_vtt_time(float(c['start']))} --> {_vtt_time(float(c['end']))}", text, ""]
    return "\n".join(out)
