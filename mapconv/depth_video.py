"""Video Depth Anything, called the way the upstream repo infers a clip.

The 32-frame window and overlap stay inside their code. We only choose the
device, the encoder, and the input size, then fix attention memory use.
"""

from __future__ import annotations

import gc
import sys

import numpy as np
import torch
import torch.nn.functional as F

from mapconv.device import precision_is_fp32, snap_input_size, torch_device
from mapconv.paths import VDA_ROOT, VIDEO_WEIGHTS

_model = None
_model_encoder = None
_patched = False

MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}


def _ensure_source() -> None:
    entry = VDA_ROOT / "video_depth_anything" / "video_depth.py"
    if not entry.exists():
        raise FileNotFoundError(
            "Video Depth Anything source is missing. Run install.py first."
        )
    root = str(VDA_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _use_sdpa() -> None:
    """Use PyTorch's memory-efficient attention instead of a full QK matrix.

    Upstream uses xFormers when it is installed, and a dense attention matrix
    when it is not. Dense attention at a 32-frame window does not fit a 16 GB
    GPU. Scaled-dot-product attention is the same operation and selects the
    flash or memory-efficient kernel that current NVIDIA cards provide.
    """
    global _patched
    if _patched:
        return
    from video_depth_anything.dinov2_layers.attention import MemEffAttention

    def forward(self, x, attn_bias=None):
        if attn_bias is not None:
            raise RuntimeError("Nested attention bias is not used by this depth model.")
        batch, tokens, channels = x.shape
        qkv = (
            self.qkv(x)
            .reshape(batch, tokens, 3, self.num_heads, channels // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        query, key, value = qkv.unbind(0)
        out = F.scaled_dot_product_attention(query, key, value, scale=self.scale)
        out = out.transpose(1, 2).reshape(batch, tokens, channels)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

    MemEffAttention.forward = forward
    _patched = True


def release() -> None:
    global _model, _model_encoder
    _model = None
    _model_encoder = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def checkpoint_path(encoder: str):
    from mapconv.paths import CHECKPOINTS

    name, _url, _min_size = VIDEO_WEIGHTS[encoder]
    for path in (CHECKPOINTS / name, VDA_ROOT / "checkpoints" / name):
        if path.exists():
            return path
    return CHECKPOINTS / name


def load(encoder: str):
    global _model, _model_encoder
    if encoder not in MODEL_CONFIGS:
        raise ValueError(f"Unknown depth encoder {encoder}.")
    if _model is not None and _model_encoder == encoder:
        return _model
    release()
    _ensure_source()
    import contextlib
    import io
    import logging

    logging.getLogger("dinov2").setLevel(logging.ERROR)
    _use_sdpa()
    weight = checkpoint_path(encoder)
    if not weight.exists() or weight.stat().st_size < 1_000_000:
        raise FileNotFoundError(
            f"Missing checkpoint {weight.name}. Run install.py, or pick another depth model."
        )
    with contextlib.redirect_stdout(io.StringIO()):
        from video_depth_anything.video_depth import VideoDepthAnything

    device = torch_device()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = VideoDepthAnything(**MODEL_CONFIGS[encoder])
    state = torch.load(weight, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    _model = model
    _model_encoder = encoder
    return model


def infer(frames: np.ndarray, fps: float, encoder: str, input_size: int) -> tuple[np.ndarray, str]:
    """Return float disparity [N,H,W] and a note if the input size was reduced after an out-of-memory error."""
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise RuntimeError("Depth expects RGB frames shaped [frame, height, width, 3].")
    device = torch_device()
    fp32 = precision_is_fp32() or device == "cpu"
    size = snap_input_size(input_size)
    attempts = [size]
    if size > 364:
        attempts.append(364)
    if size > 280:
        attempts.append(280)
    # Keep order, drop duplicates.
    ordered = []
    for item in attempts:
        if item not in ordered:
            ordered.append(item)

    last_error = None
    used = size
    for attempt in ordered:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            model = load(encoder)
            depths, _returned_fps = model.infer_video_depth(
                frames,
                float(fps if fps else 24.0),
                input_size=int(attempt),
                device=device,
                fp32=fp32,
            )
            used = attempt
            note = ""
            if attempt != size:
                note = (
                    f"Input size {size} ran out of GPU memory. "
                    f"Finished at input size {attempt} with the same model."
                )
            return np.asarray(depths), note
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            last_error = exc
            release()
    raise RuntimeError(
        f"Not enough GPU memory for the {encoder} depth model. "
        "Choose Small or Base, or lower Input size. "
        f"Last error: {last_error}"
    )
