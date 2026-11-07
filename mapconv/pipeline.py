"""Preview and full export. Depth videos stay on one gray scale; other maps are optional layers."""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from mapconv import depth_still, depth_video, edges
from mapconv.device import recommend_encoder, snap_input_size
from mapconv.grade import (
    clay_shade,
    composite_stack,
    dither_u8,
    grade_gray,
    gray_to_rgb,
    ink_mask,
)
from mapconv.media import (
    extract_audio,
    load_image,
    probe,
    read_frames,
    save_png_sequence,
    write_video,
)
from mapconv.paths import OUTPUTS
from mapconv.recipe import MAP_KIND, MAP_LABELS, Recipe

PREVIEW_FRAMES = 32


@dataclass
class RenderResult:
    frames: np.ndarray
    notes: list[str]
    layers: list[tuple[np.ndarray, str]]
    raw_depth: np.ndarray | None


def resolve_encoder(recipe: Recipe) -> str:
    if recipe.encoder == "auto":
        encoder, _size = recommend_encoder()
        return encoder
    return recipe.encoder


def _tick(progress, amount: float, desc: str) -> None:
    if progress is not None:
        progress(amount, desc=desc)


def _u8(values01: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(values01 * 255.0), 0, 255).astype(np.uint8)


def _render_map(name: str, frame: np.ndarray, depth_plane, recipe: Recipe) -> tuple[np.ndarray, bool]:
    """Return the layer as uint8 RGB, and whether MLSD found nothing."""
    if name in {"depth", "clay"}:
        if depth_plane is None:
            raise RuntimeError(f"{MAP_LABELS[name]} needs a depth estimate, and none was produced.")
        field = depth_plane.astype(np.float32)
        if name == "clay":
            if recipe.invert:
                field = 1.0 - field
            return _u8(clay_shade(field, recipe.clay_relief)), False
        return _u8(gray_to_rgb(grade_gray(field, recipe))), False
    native = edges.run_map(name, frame, recipe)
    blank = name == "mlsd" and int(native.max()) == 0
    if MAP_KIND[name] == "edge":
        mask = ink_mask(native).astype(np.float32) / 255.0
        return _u8(gray_to_rgb(mask)), blank
    return native, blank


def render_frames(
    frames: np.ndarray,
    recipe: Recipe,
    depth_mode: str,
    encoder: str,
    preview: bool,
    progress=None,
) -> RenderResult:
    if frames.ndim != 4 or frames.shape[0] == 0:
        raise RuntimeError("Nothing to convert.")
    notes: list[str] = []
    count = int(frames.shape[0])
    input_size = snap_input_size(recipe.input_size)
    label = {"vits": "Small", "vitb": "Base", "vitl": "Large"}[encoder]
    if recipe.needs_depth():
        if depth_mode == "video":
            notes.append(
                f"Video Depth Anything {label}, input size {input_size}. "
                "One depth scale for every frame in this run."
            )
        else:
            notes.append(f"Depth Anything V2 {label} (still image).")
        if recipe.clip_extremes:
            notes.append("Gray levels use the 1st to 99th percentile so a few extreme pixels do not flatten the map.")
    if "clay" in recipe.selected_maps():
        notes.append(
            "Clay is a gray shaded form lit from the depth, like a textureless playblast. "
            "It is not a modeled 3D scene."
        )
    if preview and depth_mode == "video" and recipe.needs_depth():
        notes.append(
            "Preview contrast is fit to this window only. "
            "The full export uses one scale for the whole clip."
        )

    raw_depth = None
    depth_norm = None
    if recipe.needs_depth():
        _tick(progress, 0.15, "Estimating depth")
        if depth_mode == "video":
            raw_depth, oom_note = depth_video.infer(frames, 24.0, encoder, input_size)
            if oom_note:
                notes.append(oom_note)
        else:
            planes = [depth_still.infer(frames[index], encoder) for index in range(count)]
            raw_depth = np.stack(planes, axis=0)
        from mapconv.grade import normalize_depth

        depth_norm = normalize_depth(raw_depth, recipe.clip_extremes)

    chosen = []
    seen = set()
    for slot, (name, opacity, mode) in enumerate(recipe.stack(), start=1):
        if not name or name == "none" or float(opacity) <= 0:
            continue
        if name in seen:
            notes.append(f"{MAP_LABELS[name]} is already in the stack, so layer {slot} was skipped.")
            continue
        seen.add(name)
        chosen.append((slot, name, float(opacity), mode if mode in {"blend", "transparent"} else "blend"))
    if not chosen:
        raise RuntimeError("Choose at least one map. The bottom layer is the start of the stack.")
    if any(mode == "transparent" for _slot, _name, _opacity, mode in chosen):
        notes.append(
            "Transparent layers drop pixels that match the frame border, then cover what is underneath by their opacity."
        )

    needs_aux = any(edges.uses_gpu_model(name, recipe) for _slot, name, _opacity, _mode in chosen)
    if needs_aux and recipe.needs_depth():
        depth_video.release()
        depth_still.release()

    middle = count // 2
    output = np.empty_like(frames)
    layers: list[tuple[np.ndarray, str]] = []
    mlsd_blank = False

    for index in range(count):
        _tick(progress, 0.2 + 0.6 * (index / max(count, 1)), f"Frame {index + 1} of {count}")
        frame = frames[index]
        stack = []
        previews = []
        for slot, name, opacity, mode in chosen:
            rgb, blank = _render_map(name, frame, None if depth_norm is None else depth_norm[index], recipe)
            if blank:
                mlsd_blank = True
            stack.append((rgb, opacity, mode))
            mode_label = "alpha" if mode == "blend" else "transparent"
            previews.append((rgb, f"{slot}. {MAP_LABELS[name]} · {mode_label} {opacity:.2f}"))

        merged = composite_stack(stack)
        output[index] = dither_u8(merged, index, recipe.dither)
        if index == middle:
            layers = [(frame, "Original"), *previews, (output[index], "Composite")]

    if mlsd_blank:
        notes.append("MLSD drew no lines on at least one frame. Lower the score threshold if the shot should have edges.")
    if recipe.invert:
        notes.append("Invert is on, so near is black. Leave it off for the usual near-white depth.")
    return RenderResult(frames=output, notes=notes, layers=layers, raw_depth=raw_depth)


def _new_dir(prefix: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = OUTPUTS / f"{prefix}_{stamp}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _safe_stem(path: Path) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in path.stem)
    return (cleaned[:80] or "map").strip("_") or "map"


def _write_outputs(result: RenderResult, folder: Path, stem: str, recipe: Recipe, fps: float, audio: Path | None) -> list[Path]:
    written = []
    video_path = folder / f"{stem}_{recipe.tag()}.mp4"
    write_video(result.frames, video_path, fps=fps, crf=int(recipe.crf), audio_path=audio)
    written.append(video_path)
    if recipe.save_frames:
        frame_dir = folder / f"{stem}_{recipe.tag()}_frames"
        save_png_sequence(result.frames, frame_dir)
        written.append(frame_dir)
    if recipe.save_npz and result.raw_depth is not None:
        npz_path = folder / f"{stem}_{recipe.tag()}_depth.npz"
        np.savez_compressed(npz_path, depths=result.raw_depth.astype(np.float32))
        written.append(npz_path)
        result.notes.append("The .npz is the model disparity before contrast, blur, and invert.")
    return written


def preview_video(path: str, recipe: Recipe, start_sec: float, progress=None) -> tuple[np.ndarray, str, list, str]:
    _tick(progress, 0.02, "Reading preview frames")
    info = probe(path)
    start = max(0.0, float(start_sec or 0))
    if info.duration and start >= info.duration:
        start = max(0.0, info.duration - 0.25)
    frames, fps, read_note = read_frames(
        path,
        max_side=int(recipe.max_side),
        start_sec=start,
        max_frames=PREVIEW_FRAMES,
        target_fps=float(recipe.target_fps),
    )
    encoder = resolve_encoder(recipe)
    result = render_frames(frames, recipe, "video", encoder, preview=True, progress=progress)
    _tick(progress, 0.9, "Writing preview")
    folder = _new_dir("preview")
    video_path = folder / f"{_safe_stem(Path(path))}_{recipe.tag()}.mp4"
    write_video(result.frames, video_path, fps=fps, crf=int(recipe.crf), audio_path=None)
    height, width = result.frames.shape[1], result.frames.shape[2]
    notes = [info.summary()]
    if read_note:
        notes.append(read_note)
    notes.append(
        f"Preview: {result.frames.shape[0]} frames, {width}x{height}, {fps:.3f} fps, starting at {start:.2f}s. No audio in the preview."
    )
    notes.extend(result.notes)
    middle = result.frames[result.frames.shape[0] // 2]
    _tick(progress, 1.0, "Preview ready")
    return middle, str(video_path), result.layers, "\n".join(notes)


def _estimate_bytes(info, recipe: Recipe) -> int:
    seconds = info.duration or 0
    if recipe.max_seconds and recipe.max_seconds > 0:
        seconds = min(seconds or recipe.max_seconds, float(recipe.max_seconds))
    fps = info.fps or 24.0
    if recipe.target_fps and recipe.target_fps > 0:
        fps = float(recipe.target_fps)
    frames = max(1, int(seconds * fps))
    long_side = max(info.width, info.height)
    limit = int(recipe.max_side) if recipe.max_side else long_side
    scale = min(1.0, limit / max(long_side, 1))
    width = max(2, int(info.width * scale))
    height = max(2, int(info.height * scale))
    # RGB frames plus a float32 depth plane, with a little headroom.
    return int(frames * width * height * (3 + 4) * 1.3)


def generate_video(path: str, recipe: Recipe, progress=None) -> tuple[str, str]:
    _tick(progress, 0.02, "Reading video")
    info = probe(path)
    estimate = _estimate_bytes(info, recipe)
    if estimate > 10 * 1024 ** 3:
        gb = estimate / (1024 ** 3)
        raise RuntimeError(
            f"This clip would use about {gb:.1f} GB of RAM at the current size and length. "
            "Lower Max resolution or Max seconds, then preview again."
        )
    frames, fps, read_note = read_frames(
        path,
        max_side=int(recipe.max_side),
        max_seconds=float(recipe.max_seconds) if recipe.max_seconds and recipe.max_seconds > 0 else None,
        target_fps=float(recipe.target_fps),
    )
    encoder = resolve_encoder(recipe)
    result = render_frames(frames, recipe, "video", encoder, preview=False, progress=progress)
    _tick(progress, 0.88, "Writing video")
    audio = None
    if recipe.keep_audio and info.has_audio:
        audio = extract_audio(path, _new_dir("audio") / "track.m4a")
    folder = _new_dir("video")
    written = _write_outputs(result, folder, _safe_stem(Path(path)), recipe, fps, audio)
    height, width = result.frames.shape[1], result.frames.shape[2]
    notes = [info.summary()]
    if read_note:
        notes.append(read_note)
    if recipe.max_seconds and recipe.max_seconds > 0 and info.duration > float(recipe.max_seconds) + 0.05:
        notes.append(f"Stopped at {float(recipe.max_seconds):.1f}s. Raise Max seconds to convert more of the file.")
    notes.append(
        f"Export: {result.frames.shape[0]} frames, {width}x{height}, {fps:.3f} fps, "
        f"H.264 CRF {int(recipe.crf)}, yuv420p."
    )
    if audio is None and recipe.keep_audio:
        notes.append("No audio was copied. The source may not have an audio track.")
    elif audio is not None:
        notes.append("Original audio is in the export.")
    notes.extend(result.notes)
    notes.append("Saved:")
    notes.extend(str(item) for item in written)
    _tick(progress, 1.0, "Done")
    return str(written[0]), "\n".join(notes)


def preview_image(path: str, recipe: Recipe, progress=None) -> tuple[np.ndarray, list, str]:
    _tick(progress, 0.05, "Reading image")
    frame = load_image(path, int(recipe.max_side))
    frames = frame[None, ...]
    encoder = resolve_encoder(recipe)
    result = render_frames(frames, recipe, "still", encoder, preview=False, progress=progress)
    notes = [f"{frame.shape[1]}x{frame.shape[0]}"]
    notes.extend(result.notes)
    _tick(progress, 1.0, "Preview ready")
    return result.frames[0], result.layers, "\n".join(notes)


def generate_images(paths: list[str], recipe: Recipe, progress=None) -> tuple[list[tuple[str, str]], str]:
    if not paths:
        raise RuntimeError("Add at least one image.")
    folder = _new_dir("images")
    encoder = resolve_encoder(recipe)
    gallery = []
    notes = []
    total = len(paths)
    for index, path in enumerate(paths):
        _tick(progress, index / total, f"Image {index + 1} of {total}")
        frame = load_image(path, int(recipe.max_side))
        result = render_frames(frame[None, ...], recipe, "still", encoder, preview=False, progress=None)
        stem = _safe_stem(Path(path))
        image_path = folder / f"{index:03d}_{stem}_{recipe.tag()}.png"
        Image.fromarray(result.frames[0]).save(image_path)
        if recipe.save_npz and result.raw_depth is not None:
            np.savez_compressed(
                folder / f"{index:03d}_{stem}_{recipe.tag()}_depth.npz",
                depths=result.raw_depth.astype(np.float32),
            )
        gallery.append((str(image_path), Path(path).name))
        notes.append(f"{Path(path).name}: {image_path}")
    notes.extend(result.notes)
    _tick(progress, 1.0, "Done")
    return gallery, "Saved:\n" + "\n".join(notes)


def format_error(exc: BaseException) -> str:
    traceback.print_exc()
    return f"{type(exc).__name__}: {exc}"
