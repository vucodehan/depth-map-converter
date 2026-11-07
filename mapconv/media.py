"""Read and write media with the local ffmpeg build."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from mapconv.grade import even_crop
from mapconv.paths import WORK

_CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


@dataclass
class MediaInfo:
    path: str
    width: int
    height: int
    fps: float
    duration: float
    nb_frames: int
    has_audio: bool
    vfr: bool

    def summary(self) -> str:
        audio = "with audio" if self.has_audio else "no audio"
        rate = "variable frame rate" if self.vfr else "constant frame rate"
        seconds = f"{self.duration:.2f}s" if self.duration else "unknown length"
        return (
            f"{self.width}x{self.height}, {self.fps:.3f} fps, {seconds}, "
            f"{self.nb_frames or '?'} frames, {audio}, {rate}"
        )


def ffmpeg_exe() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def ffprobe_exe() -> str:
    ffmpeg = ffmpeg_exe()
    name = "ffprobe.exe" if os.name == "nt" else "ffprobe"
    sibling = Path(ffmpeg).with_name(name)
    if sibling.exists():
        return str(sibling)
    found = shutil.which("ffprobe")
    if found:
        return found
    raise RuntimeError("ffprobe was not found next to ffmpeg.")


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        capture_output=True,
        creationflags=_CREATE_NO_WINDOW,
        check=False,
    )


def _parse_rate(value: str | None) -> float:
    if not value or value in {"0/0", "N/A"}:
        return 0.0
    if "/" in value:
        num, den = value.split("/", 1)
        den_f = float(den)
        if den_f == 0:
            return 0.0
        return float(num) / den_f
    return float(value)


def probe(path: str | Path) -> MediaInfo:
    src = str(path)
    proc = _run(
        [
            ffprobe_exe(),
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            src,
        ]
    )
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", errors="replace")[-1500:]
        raise RuntimeError(f"Could not read {src}. {detail}")
    data = json.loads(proc.stdout.decode("utf-8", errors="replace"))
    video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    if video is None:
        raise RuntimeError(f"No video stream in {src}.")
    has_audio = any(s.get("codec_type") == "audio" for s in data.get("streams", []))
    avg = _parse_rate(video.get("avg_frame_rate"))
    raw = _parse_rate(video.get("r_frame_rate"))
    fps = avg or raw or 24.0
    duration = float(video.get("duration") or data.get("format", {}).get("duration") or 0)
    nb = int(video.get("nb_frames") or 0)
    if nb <= 0 and duration and fps:
        nb = int(round(duration * fps))
    vfr = bool(avg and raw and abs(raw - avg) / max(avg, 1e-3) > 0.05)
    return MediaInfo(
        path=src,
        width=int(video["width"]),
        height=int(video["height"]),
        fps=fps,
        duration=duration,
        nb_frames=nb,
        has_audio=has_audio,
        vfr=vfr,
    )


def _scale_filter(max_side: int) -> str:
    if max_side and max_side > 0:
        return (
            f"scale='if(gt(iw,ih),min({int(max_side)},iw),-2)'"
            f":'if(gt(ih,iw),min({int(max_side)},ih),-2)'"
            f":flags=lanczos:force_divisible_by=2"
        )
    return "scale=iw:ih:force_divisible_by=2"


def read_frames(
    path: str | Path,
    max_side: int = 1280,
    start_sec: float = 0,
    max_frames: int | None = None,
    max_seconds: float | None = None,
    target_fps: float = -1,
) -> tuple[np.ndarray, float, str]:
    """Return RGB uint8 frames [N,H,W,3], output fps, and a short note."""
    info = probe(path)
    notes = []
    if info.vfr:
        notes.append(f"Variable frame rate. Frames are read at {info.fps:.3f} fps.")
    fps = info.fps
    vf_parts = [_scale_filter(max_side)]
    if target_fps and target_fps > 0 and abs(target_fps - info.fps) > 0.05:
        fps = float(target_fps)
        vf_parts.append(f"fps={fps:.6f}")
        notes.append(f"Frame rate changed from {info.fps:.3f} to {fps:.3f}.")
    elif info.vfr:
        vf_parts.append(f"fps={info.fps:.6f}")

    folder = WORK / f"in_{uuid.uuid4().hex}"
    folder.mkdir(parents=True, exist_ok=True)
    pattern = str(folder / "frame_%06d.png")
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y"]
    if start_sec and start_sec > 0:
        cmd += ["-ss", f"{float(start_sec):.3f}"]
    cmd += ["-i", str(path), "-an", "-vf", ",".join(vf_parts)]
    if max_seconds and max_seconds > 0 and not max_frames:
        cmd += ["-t", f"{float(max_seconds):.3f}"]
    if max_frames:
        cmd += ["-frames:v", str(int(max_frames))]
    if not any(part.startswith("fps=") for part in vf_parts):
        cmd += ["-fps_mode", "passthrough"]
    cmd.append(pattern)
    try:
        proc = _run(cmd)
        if proc.returncode != 0:
            detail = proc.stderr.decode("utf-8", errors="replace")[-2000:]
            raise RuntimeError(f"ffmpeg could not read frames. {detail}")
        files = sorted(folder.glob("frame_*.png"))
        if not files:
            raise RuntimeError("ffmpeg wrote no frames. The file may be empty or use an unsupported codec.")
        frames = [np.asarray(Image.open(file).convert("RGB")) for file in files]
        stacked = np.stack([even_crop(frame) for frame in frames], axis=0)
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    return stacked, fps, " ".join(notes)


def extract_audio(src: str | Path, dest: Path) -> Path | None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    copied = dest.with_suffix(".m4a")
    proc = _run(
        [
            ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(src),
            "-vn",
            "-acodec",
            "copy",
            str(copied),
        ]
    )
    if proc.returncode == 0 and copied.exists() and copied.stat().st_size > 500:
        return copied
    copied.unlink(missing_ok=True)
    encoded = dest.with_suffix(".aac.m4a")
    proc = _run(
        [
            ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(src),
            "-vn",
            "-acodec",
            "aac",
            "-b:a",
            "192k",
            str(encoded),
        ]
    )
    if proc.returncode == 0 and encoded.exists() and encoded.stat().st_size > 500:
        return encoded
    encoded.unlink(missing_ok=True)
    return None


def write_video(
    frames: np.ndarray,
    dest: Path,
    fps: float,
    crf: int = 14,
    audio_path: Path | None = None,
) -> None:
    if frames.ndim != 4 or frames.shape[0] == 0:
        raise RuntimeError("No frames to write.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames.shape[1], frames.shape[2]
    if height % 2 or width % 2:
        raise RuntimeError(f"Frame size {width}x{height} is not even, so H.264 yuv420p cannot store it.")
    rate = fps if fps and fps > 0 else 24.0

    def encode(audio_mode: str | None) -> None:
        cmd = [
            ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            f"{rate:.6f}",
            "-i",
            "-",
        ]
        if audio_path is not None and audio_mode is not None:
            cmd += ["-i", str(audio_path), "-map", "0:v:0", "-map", "1:a:0"]
        else:
            cmd += ["-an"]
        cmd += [
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            str(int(crf)),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ]
        if audio_mode == "copy":
            cmd += ["-c:a", "copy", "-shortest"]
        elif audio_mode == "aac":
            cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest"]
        cmd.append(str(dest))
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            creationflags=_CREATE_NO_WINDOW,
        )
        assert proc.stdin is not None
        try:
            for frame in frames:
                proc.stdin.write(np.ascontiguousarray(frame).tobytes())
            proc.stdin.close()
        except BrokenPipeError:
            pass
        err = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        code = proc.wait()
        if code != 0 or not dest.exists() or dest.stat().st_size < 1000:
            raise RuntimeError(err[-2000:] or f"ffmpeg exited with code {code}")

    if audio_path is None:
        encode(None)
        return
    try:
        encode("copy")
    except RuntimeError:
        if dest.exists():
            dest.unlink()
        encode("aac")


def save_png_sequence(frames: np.ndarray, folder: Path) -> None:
    import cv2

    folder.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(folder / f"frame_{index:05d}.png"), bgr)


def load_image(path: str | Path, max_side: int) -> np.ndarray:
    from PIL import ImageOps

    image = ImageOps.exif_transpose(Image.open(path).convert("RGB"))
    rgb = np.asarray(image)
    height, width = rgb.shape[:2]
    long_side = max(height, width)
    if max_side and max_side > 0 and long_side > max_side:
        scale = max_side / long_side
        resized = (
            max(2, int(round(width * scale))),
            max(2, int(round(height * scale))),
        )
        rgb = np.asarray(image.resize(resized, Image.Resampling.LANCZOS))
    return even_crop(rgb)
