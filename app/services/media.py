"""ffprobe / ffmpeg helpers for reading the uploaded clip."""

import json
import subprocess
from pathlib import Path

HDR_TRANSFERS = {"smpte2084", "arib-std-b67"}


def probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    data = json.loads(out)
    video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    audio = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)
    if video is None:
        raise ValueError("ไฟล์นี้ไม่มีภาพวิดีโอ")

    rotation = 0
    for sd in video.get("side_data_list") or []:
        if "rotation" in sd:
            rotation = int(round(float(sd["rotation"])))
    if not rotation and "rotate" in (video.get("tags") or {}):
        rotation = -int(video["tags"]["rotate"])
    rotation %= 360  # ffprobe reports counter-clockwise degrees, e.g. -90 -> 270

    w, h = int(video.get("width") or 0), int(video.get("height") or 0)
    if rotation in (90, 270):
        w, h = h, w
    num, _, den = (video.get("avg_frame_rate") or "0/1").partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 0.0
    duration = float(data.get("format", {}).get("duration") or video.get("duration") or 0)
    return {
        "width": w,
        "height": h,
        "rotation": rotation,
        "fps": round(fps, 3),
        "duration": round(duration, 3),
        "vcodec": video.get("codec_name"),
        "pix_fmt": video.get("pix_fmt"),
        "hdr": video.get("color_transfer") in HDR_TRANSFERS,
        "has_audio": audio is not None,
    }


def extract_audio(src: Path, wav: Path) -> Path:
    """16 kHz mono PCM, what whisper expects."""
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)],
        check=True, capture_output=True,
    )
    return wav


def thumbnail(src: Path, out: Path, at: float = 0.5) -> Path:
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-ss", str(at), "-i", str(src), "-frames:v", "1", "-vf", "scale=360:-2", str(out)],
        check=True, capture_output=True,
    )
    return out
