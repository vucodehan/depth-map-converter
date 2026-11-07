"""GPU selection helpers. Torch is imported inside functions so start.py can set the device first."""

from __future__ import annotations

import os


def precision_is_fp32() -> bool:
    return os.environ.get("MAPCONV_PRECISION", "fp16").strip().lower() == "fp32"


def torch_device() -> str:
    import torch

    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def gpu_total_gb() -> float | None:
    import torch

    if torch_device() != "cuda":
        return None
    props = torch.cuda.get_device_properties(0)
    return props.total_memory / (1024 ** 3)


def recommend_encoder() -> tuple[str, int]:
    """Model and input size that should fit the selected GPU on the first run."""
    mem = gpu_total_gb()
    if mem is None:
        return "vits", 364
    # Large at 518 measured about 10.5 GB on an RTX 5080 laptop once attention
    # uses PyTorch SDPA. Leave headroom below that for the display.
    if mem >= 15:
        return "vitl", 518
    if mem >= 11:
        return "vitl", 392
    if mem >= 8:
        return "vitb", 392
    return "vits", 364


def device_summary() -> str:
    import torch

    prec = "fp32" if precision_is_fp32() else "fp16"
    if torch_device() != "cuda":
        return f"CPU, {prec}. Video depth will be slow."
    name = torch.cuda.get_device_name(0)
    mem = gpu_total_gb() or 0
    enc, size = recommend_encoder()
    label = {"vits": "Small", "vitb": "Base", "vitl": "Large"}[enc]
    return f"{name}, {mem:.1f} GB, {prec}. Auto depth model: {label} at input size {size}."


def snap_input_size(value: int) -> int:
    size = int(value)
    size = max(280, min(700, size))
    snapped = int(round(size / 14.0) * 14)
    return max(280, min(700, snapped))
