import shutil
import subprocess
from pathlib import Path

import pytest

from app import config
from app.services import media, renderer, subtitles

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def make_clip(path: Path, size="1280x720", seconds=2, audio=True) -> Path:
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc2=size={size}:rate=25:duration={seconds}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}", "-c:a", "aac"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-shortest", str(path)]
    subprocess.run(cmd, check=True)
    return path


CUES = [
    {"start": 0.2, "end": 1.2, "text": "วันนี้กุ๊กไก่แอบหนีเที่ยว\nอีกแล้ว"},
    {"start": 1.3, "end": 1.9, "text": "แต่ดันเจอเจ้าเหมียวดักรออยู่หน้าบ้าน!"},
]


def test_build_ass(tmp_path):
    style = subtitles.SubtitleStyle.from_dict({"size": 80, "color": "#FFE600", "margin_v": 400})
    p = subtitles.build_ass(CUES, style, tmp_path / "s.ass")
    text = p.read_text()
    assert "PlayResX: 1080" in text and "PlayResY: 1920" in text
    assert "Kanit" in text
    assert "\\N" in text  # explicit Thai line breaks
    assert text.count("Dialogue:") == 2


def test_fonts_are_listed():
    families = {f["family"] for f in subtitles.list_fonts()}
    assert {"Kanit", "Prompt", "Sarabun"} <= families


def test_geometry():
    assert renderer._fit_geometry(1080, 1920, "pad") == (1080, 1920)
    assert renderer._fit_geometry(1280, 720, "pad") == (1080, 608)
    assert renderer._fit_geometry(1280, 720, "crop") == (3412, 1920)
    assert renderer._fit_geometry(720, 1280, "pad") == (1080, 1920)


def test_vaapi_command_shape(tmp_path):
    info = {"width": 1080, "height": 1920, "rotation": 270, "hdr": True, "has_audio": True, "duration": 2}
    caps = renderer.Caps(checked=True, vaapi=True, gpu_denoise=True, gpu_sharpen=False, gpu_tonemap=True)
    cmd = renderer.build_command(Path("in.mp4"), Path("out.mp4"), Path("/x/s.ass"), Path("/x/fonts"), info,
                                 renderer.RenderOptions(), "gpu", caps)
    vf = cmd[cmd.index("-vf") + 1]
    assert "-hwaccel" in cmd and "h264_vaapi" in cmd
    assert vf.index("transpose_vaapi=dir=clock") < vf.index("scale_vaapi") < vf.index("hwdownload") < vf.index("ass=") < vf.index("hwupload")
    assert "tonemap_vaapi" in vf and "denoise_vaapi" in vf
    assert "unsharp" in vf  # GPU sharpen unsupported -> CPU fallback filter


def test_cpu_render_end_to_end(tmp_path):
    src = make_clip(tmp_path / "in.mp4")
    info = media.probe(src)
    assert info["width"] == 1280 and info["has_audio"]
    ass = subtitles.build_ass(CUES, subtitles.SubtitleStyle(), tmp_path / "s.ass")
    out = tmp_path / "out.mp4"
    seen = []
    variant = renderer.render(src, out, ass, tmp_path, info, renderer.RenderOptions(), seen.append)
    assert variant == "cpu"
    o = media.probe(out)
    assert (o["width"], o["height"]) == (config.OUTPUT_WIDTH, config.OUTPUT_HEIGHT)
    assert o["vcodec"] == "h264" and o["has_audio"]
    assert abs(o["fps"] - 30) < 0.1
    assert seen and seen[-1] > 0.5
