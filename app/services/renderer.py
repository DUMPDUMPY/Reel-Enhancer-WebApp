"""FFmpeg render: fit to 1080x1920, clean up + sharpen, burn subtitles, encode for Reels.

GPU path (Intel iGPU through VAAPI):
    decode (VAAPI) -> rotate / tonemap / scale / denoise / sharpen (VAAPI)
    -> download -> pad or crop, fps, burn .ass (light CPU work)
    -> upload -> h264_vaapi encode
CPU path (fallback): the same steps with software filters and libx264.

What the iGPU supports is detected once with tiny test encodes (see detect_caps),
and every render falls back to the next variant if FFmpeg fails.
"""

import logging
import os
import shutil
import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from app import config

log = logging.getLogger(__name__)

SHARPEN_LEVELS = {  # name: (VAAPI sharpness 0-64, CPU unsharp luma amount)
    "off": (0, 0.0),
    "light": (32, 0.5),
    "medium": (44, 0.8),
    "strong": (56, 1.1),
}
DENOISE_LEVELS = {  # name: (VAAPI denoise 0-64, CPU hqdn3d)
    "off": (0, ""),
    "light": (8, "hqdn3d=1.5:1.5:3:3"),
    "medium": (16, "hqdn3d=3:3:6:6"),
}


@dataclass
class RenderOptions:
    fit: str = "pad"  # pad (keep everything, add bars) | crop (fill the frame)
    sharpen: str = "medium"
    denoise: str = "light"

    @classmethod
    def from_dict(cls, d: dict | None) -> "RenderOptions":
        d = d or {}
        o = cls()
        if d.get("fit") in ("pad", "crop"):
            o.fit = d["fit"]
        if d.get("sharpen") in SHARPEN_LEVELS:
            o.sharpen = d["sharpen"]
        if d.get("denoise") in DENOISE_LEVELS:
            o.denoise = d["denoise"]
        return o


@dataclass
class Caps:
    checked: bool = False
    vaapi: bool = False
    low_power: bool = False
    gpu_denoise: bool = False
    gpu_sharpen: bool = False
    gpu_tonemap: bool = False
    detail: list[str] = field(default_factory=list)

    @property
    def mode(self) -> str:
        if config.RENDER_MODE == "cpu" or not self.vaapi:
            return "cpu"
        return "vaapi"


_caps = Caps()
_caps_lock = threading.Lock()


def _run_quiet(cmd: list[str], timeout: float = 30) -> tuple[bool, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode == 0, p.stderr[-600:]
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, str(e)


def _vaapi_test(vf: str, low_power: bool) -> tuple[bool, str]:
    cmd = [
        "ffmpeg", "-v", "error", "-init_hw_device", f"vaapi=va:{config.VAAPI_DEVICE}", "-filter_hw_device", "va",
        "-f", "lavfi", "-i", "testsrc2=size=320x568:rate=30:duration=0.5",
        "-vf", f"format=nv12,hwupload,{vf}", "-c:v", "h264_vaapi", "-low_power", "1" if low_power else "0",
        "-f", "null", "-",
    ]
    return _run_quiet(cmd)


def detect_caps(force: bool = False) -> Caps:
    """Probe what the iGPU can do. Cheap (a few 0.5 s test clips), cached."""
    global _caps
    with _caps_lock:
        if _caps.checked and not force:
            return _caps
        caps = Caps(checked=True)
        if config.RENDER_MODE == "cpu":
            caps.detail.append("RENDER_MODE=cpu: ไม่ใช้ iGPU")
        elif not os.path.exists(config.VAAPI_DEVICE):
            caps.detail.append(f"ไม่พบ {config.VAAPI_DEVICE} (ยังไม่ได้ map /dev/dri เข้า container)")
        else:
            base = "scale_vaapi=w=270:h=480"
            for lp in (False, True):
                ok, err = _vaapi_test(base, lp)
                if ok:
                    caps.vaapi, caps.low_power = True, lp
                    caps.detail.append(f"h264_vaapi ใช้ได้ (low_power={int(lp)})")
                    break
                caps.detail.append(f"h264_vaapi low_power={int(lp)} ไม่ได้: {err.strip()[-200:]}")
            if caps.vaapi:
                caps.gpu_denoise = _vaapi_test(f"{base},denoise_vaapi=denoise=8", caps.low_power)[0]
                caps.gpu_sharpen = _vaapi_test(f"{base},sharpness_vaapi=sharpness=44", caps.low_power)[0]
                caps.gpu_tonemap = _vaapi_test(
                    f"{base},tonemap_vaapi=format=nv12:t=bt709:m=bt709:p=bt709", caps.low_power
                )[0]
                caps.detail.append(
                    f"denoise_vaapi={caps.gpu_denoise} sharpness_vaapi={caps.gpu_sharpen} tonemap_vaapi={caps.gpu_tonemap}"
                )
        _caps = caps
        log.info("render caps: %s", caps)
        return caps


def _fit_geometry(w: int, h: int, fit: str) -> tuple[int, int]:
    """Size to scale the source to before padding/cropping to the output frame."""
    W, H = config.OUTPUT_WIDTH, config.OUTPUT_HEIGHT
    if w <= 0 or h <= 0:
        return W, H
    src_wider = w / h > W / H
    if (fit == "pad") == src_wider:
        sw, sh = W, round(W * h / w)
    else:
        sw, sh = round(H * w / h), H
    # Within 1% of 9:16: just stretch the last few pixels instead of thin bars.
    if abs(sw - W) <= W * 0.01:
        sw = W
    if abs(sh - H) <= H * 0.01:
        sh = H
    return max(2, sw // 2 * 2), max(2, sh // 2 * 2)


def _frame_filter(sw: int, sh: int, fit: str) -> str:
    W, H = config.OUTPUT_WIDTH, config.OUTPUT_HEIGHT
    if (sw, sh) == (W, H):
        return ""
    if fit == "crop":
        return f"crop={W}:{H}"
    return f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:black"


def _sub_filter(ass_path: Path, fonts_dir: Path) -> str:
    def esc(p: Path) -> str:
        return str(p).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")

    return f"ass=filename='{esc(ass_path)}':fontsdir='{esc(fonts_dir)}'"


_TRANSPOSE = {270: "clock", 90: "cclock", 180: "reversal"}


def build_command(
    src: Path, out: Path, ass_path: Path, fonts_dir: Path, info: dict, opts: RenderOptions,
    variant: str, caps: Caps,
) -> list[str]:
    """variant: 'gpu' (VAAPI decode), 'gpu-swdec' (software decode, VAAPI filters+encode), 'cpu'."""
    sw, sh = _fit_geometry(info["width"], info["height"], opts.fit)
    frame = _frame_filter(sw, sh, opts.fit)
    sharp_gpu, sharp_cpu = SHARPEN_LEVELS[opts.sharpen]
    dn_gpu, dn_cpu = DENOISE_LEVELS[opts.denoise]
    subs = _sub_filter(ass_path, fonts_dir)
    fps = f"fps={config.OUTPUT_FPS}"

    cmd = ["ffmpeg", "-y", "-hide_banner", "-nostats", "-progress", "pipe:1"]
    if variant == "cpu":
        chain = []
        if info.get("hdr"):
            chain.append("zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv")
        chain.append(f"scale={sw}:{sh}:flags=lanczos")
        if dn_cpu:
            chain.append(dn_cpu)
        if sharp_cpu:
            chain.append(f"unsharp=5:5:{sharp_cpu}:3:3:0.4")
        chain += [frame, fps, subs, "format=yuv420p"]
        cmd += ["-threads", str(config.CPU_THREADS), "-i", str(src)]
        cmd += ["-vf", ",".join(f for f in chain if f), "-filter_threads", str(config.CPU_THREADS)]
        cmd += [
            "-c:v", "libx264", "-preset", config.X264_PRESET, "-crf", str(config.X264_CRF),
            "-maxrate", config.VIDEO_MAXRATE, "-bufsize", config.VIDEO_BUFSIZE,
            "-profile:v", "high", "-level", "4.2", "-threads", str(config.CPU_THREADS),
        ]
    else:
        cmd += ["-init_hw_device", f"vaapi=va:{config.VAAPI_DEVICE}", "-filter_hw_device", "va"]
        if variant == "gpu":
            cmd += ["-hwaccel", "vaapi", "-hwaccel_device", "va", "-hwaccel_output_format", "vaapi"]
        cmd += ["-noautorotate", "-i", str(src)]
        chain = [] if variant == "gpu" else ["format=nv12", "hwupload"]
        if info.get("rotation") in _TRANSPOSE:
            chain.append(f"transpose_vaapi=dir={_TRANSPOSE[info['rotation']]}")
        if info.get("hdr") and caps.gpu_tonemap:
            chain.append("tonemap_vaapi=format=nv12:t=bt709:m=bt709:p=bt709")
        chain.append(f"scale_vaapi=w={sw}:h={sh}:mode=hq:format=nv12")
        cpu_extra = []
        if dn_gpu and caps.gpu_denoise:
            chain.append(f"denoise_vaapi=denoise={dn_gpu}")
        elif dn_cpu:
            cpu_extra.append(dn_cpu)
        if sharp_gpu and caps.gpu_sharpen:
            chain.append(f"sharpness_vaapi=sharpness={sharp_gpu}")
        elif sharp_cpu:
            cpu_extra.append(f"unsharp=5:5:{sharp_cpu}:3:3:0.4")
        chain += ["hwdownload", "format=nv12", *cpu_extra, frame, fps, subs, "format=nv12", "hwupload"]
        cmd += ["-vf", ",".join(f for f in chain if f), "-filter_threads", "2"]
        cmd += [
            "-c:v", "h264_vaapi", "-profile:v", "high", "-rc_mode", "VBR",
            "-b:v", config.VIDEO_BITRATE, "-maxrate", config.VIDEO_MAXRATE, "-bufsize", config.VIDEO_BUFSIZE,
            "-g", str(config.OUTPUT_FPS * 2), "-low_power", "1" if caps.low_power else "0",
        ]

    cmd += ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv"]
    cmd += ["-map", "0:v:0"]
    if info.get("has_audio"):
        cmd += ["-map", "0:a:0", "-c:a", "aac", "-b:a", "192k", "-ar", "48000"]
    cmd += ["-movflags", "+faststart", str(out)]
    return cmd


def prepare_fonts_dir(job_dir: Path, font_dirs: list[Path]) -> Path:
    """libass takes one fontsdir: link every font we know about into the job folder."""
    d = job_dir / "fonts"
    d.mkdir(exist_ok=True)
    for src_dir in font_dirs:
        if not src_dir.is_dir():
            continue
        for p in src_dir.iterdir():
            if p.suffix.lower() in (".ttf", ".otf") and not (d / p.name).exists():
                try:
                    (d / p.name).symlink_to(p.resolve())
                except OSError:
                    shutil.copy2(p, d / p.name)
    return d


def _run_ffmpeg(cmd: list[str], duration: float, log_path: Path, on_progress: Callable[[float], None] | None) -> bool:
    with log_path.open("a") as logf:
        logf.write("\n$ " + " ".join(cmd) + "\n")
        logf.flush()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=logf, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            key, _, val = line.strip().partition("=")
            if key == "out_time_us" and on_progress and duration > 0 and val.isdigit():
                on_progress(min(int(val) / 1e6 / duration, 1.0))
        return proc.wait() == 0


def render(
    src: Path, out: Path, ass_path: Path, job_dir: Path, info: dict, opts: RenderOptions,
    on_progress: Callable[[float], None] | None = None,
    on_status: Callable[[str, str | None], None] | None = None,
) -> str:
    """Render with the best available variant; returns the variant that worked.

    on_status(variant, failed_variant) is called before each attempt.
    """
    caps = detect_caps()
    variants = ["gpu", "gpu-swdec", "cpu"] if caps.mode == "vaapi" else ["cpu"]
    fonts_dir = prepare_fonts_dir(job_dir, config.FONT_DIRS)
    log_path = job_dir / "render.log"
    failed = None
    for v in variants:
        if on_status:
            on_status(v, failed)
        cmd = build_command(src, out, ass_path, fonts_dir, info, opts, v, caps)
        if _run_ffmpeg(cmd, info.get("duration") or 0, log_path, on_progress):
            return v
        log.warning("render variant %s failed, see %s", v, log_path)
        failed = v
    tail = log_path.read_text(errors="replace")[-800:]
    raise RuntimeError(f"FFmpeg เรนเดอร์ไม่สำเร็จ:\n{tail}")
