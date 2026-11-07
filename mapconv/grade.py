"""Turn model output into the gray (or overlaid) frames that get saved."""

from __future__ import annotations

import cv2
import numpy as np

from mapconv.recipe import Recipe


def normalize_depth(depths: np.ndarray, clip_extremes: bool) -> np.ndarray:
    """Map a disparity stack to 0-1. One scale for every frame in the stack."""
    flat = depths.reshape(-1).astype(np.float32)
    if flat.size == 0:
        return depths.astype(np.float32)
    if clip_extremes and flat.size > 32:
        sample = flat
        if sample.size > 2_000_000:
            rng = np.random.default_rng(0)
            sample = sample[rng.choice(sample.size, 2_000_000, replace=False)]
        lo, hi = np.percentile(sample, [1.0, 99.0])
    else:
        lo, hi = float(flat.min()), float(flat.max())
    span = float(hi - lo)
    if span < 1e-6:
        return np.zeros(depths.shape, dtype=np.float32)
    out = (depths.astype(np.float32) - float(lo)) / span
    return np.clip(out, 0.0, 1.0)


def grade_gray(values01: np.ndarray, recipe: Recipe) -> np.ndarray:
    x = values01.astype(np.float32)
    if recipe.invert:
        x = 1.0 - x
    contrast = float(recipe.contrast)
    if abs(contrast - 1.0) > 1e-4:
        x = (x - 0.5) * contrast + 0.5
    x = np.clip(x, 0.0, 1.0)
    blur = float(recipe.blur)
    if blur > 0:
        k = max(3, int(round(blur * 4)) | 1)
        x = cv2.GaussianBlur(x, (k, k), sigmaX=blur)
        x = np.clip(x, 0.0, 1.0)
    return x


def grade_color(rgb: np.ndarray, recipe: Recipe) -> np.ndarray:
    x = rgb.astype(np.float32) / 255.0
    if recipe.invert:
        x = 1.0 - x
    blur = float(recipe.blur)
    if blur > 0:
        k = max(3, int(round(blur * 4)) | 1)
        x = cv2.GaussianBlur(x, (k, k), sigmaX=blur)
    return np.clip(x, 0.0, 1.0)


def ink_mask(rgb: np.ndarray) -> np.ndarray:
    """0-255 mask. Works for white-on-black edges and black-on-white drawings."""
    if rgb.ndim == 2:
        gray = rgb
    else:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if float(gray.mean()) > 127.0:
        gray = 255 - gray
    return gray


def to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.repeat(image[:, :, None], 3, axis=2)
    if image.shape[2] == 4:
        return image[:, :, :3]
    return image


def gray_to_rgb(values01: np.ndarray) -> np.ndarray:
    return np.repeat(values01[:, :, None], 3, axis=2)


def clay_shade(depth01: np.ndarray, relief: float = 40.0) -> np.ndarray:
    """Gray clay from a disparity map. High values are closer. Returns HxWx3 in 0-1.

    Screen-space normals are lit from the upper left, with a little fill so the
    shadow side stays readable. This is a textureless form, not a 3D scene.
    """
    depth = cv2.GaussianBlur(depth01.astype(np.float32), (5, 5), 0)
    gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    scale = max(float(relief), 0.0)
    nx = -gx * scale
    ny = -gy * scale
    nz = np.ones_like(nx)
    length = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-8
    nx, ny, nz = nx / length, ny / length, nz / length
    # Toward the light: left, up (negative image y), and out of the screen.
    key = np.array([-0.45, -0.55, 0.70], dtype=np.float32)
    key /= np.linalg.norm(key)
    fill = np.array([0.35, 0.15, 0.45], dtype=np.float32)
    fill /= np.linalg.norm(fill)
    ndot_key = np.clip(nx * key[0] + ny * key[1] + nz * key[2], 0.0, 1.0)
    ndot_fill = np.clip(nx * fill[0] + ny * fill[1] + nz * fill[2], 0.0, 1.0)
    shade = np.clip(0.16 + 0.66 * ndot_key + 0.18 * ndot_fill, 0.0, 1.0)
    return gray_to_rgb(shade)


def _keyed_alpha(rgb: np.ndarray) -> np.ndarray:
    """1 where the pixel differs from the border, 0 where it matches the background."""
    if rgb.ndim == 2:
        gray = rgb.astype(np.float32)
    else:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    background = float(np.median(border))
    diff = np.abs(gray - background)
    # A floor keeps sparse lines opaque even when most of the frame is empty.
    scale = max(float(np.percentile(diff, 92)), 12.0)
    alpha = np.clip(diff / scale, 0.0, 1.0)
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
    return np.clip(alpha, 0.0, 1.0)


def composite_stack(layers: list[tuple[np.ndarray, float, str]]) -> np.ndarray:
    """Stack uint8 RGB layers from bottom to top.

    mode "blend" mixes the whole frame by opacity.
    mode "transparent" drops pixels that match the border, then applies opacity.
    Returns float HxWx3 in 0-255.
    """
    if not layers:
        raise RuntimeError("Choose at least one map.")
    height, width = layers[0][0].shape[:2]
    out = np.zeros((height, width, 3), dtype=np.float32)
    for rgb, opacity, mode in layers:
        amount = float(opacity)
        if amount <= 0:
            continue
        src = rgb.astype(np.float32)
        if mode == "transparent":
            alpha = _keyed_alpha(rgb) * amount
        else:
            alpha = np.full((height, width), np.float32(amount), dtype=np.float32)
        cover = alpha[:, :, None]
        out = out * (1.0 - cover) + src * cover
    return np.clip(out, 0.0, 255.0)


def dither_u8(values: np.ndarray, frame_index: int, enabled: bool) -> np.ndarray:
    x = values.astype(np.float32)
    if enabled:
        rng = np.random.default_rng(10_000 + int(frame_index))
        x = x + rng.uniform(-0.5, 0.5, size=x.shape).astype(np.float32)
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def even_crop(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    return frame[: height - (height % 2), : width - (width % 2)]
