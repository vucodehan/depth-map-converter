"""Depth Anything V2 for a still image. Videos do not use this path."""

from __future__ import annotations

import gc

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from mapconv.device import precision_is_fp32, torch_device
from mapconv.paths import STILL_MODELS

_model = None
_processor = None
_encoder = None


def release() -> None:
    global _model, _processor, _encoder
    _model = None
    _processor = None
    _encoder = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load(encoder: str):
    global _model, _processor, _encoder
    if _model is not None and _encoder == encoder:
        return _model, _processor
    release()
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    repo = STILL_MODELS[encoder]
    device = torch_device()
    processor = AutoImageProcessor.from_pretrained(repo, use_fast=True)
    model = AutoModelForDepthEstimation.from_pretrained(repo)
    model = model.to(device).eval()
    _model = model
    _processor = processor
    _encoder = encoder
    return model, processor


def infer(rgb: np.ndarray, encoder: str) -> np.ndarray:
    """Return float disparity [H,W]. Larger values are closer."""
    model, processor = load(encoder)
    device = torch_device()
    fp32 = precision_is_fp32() or device == "cpu"
    height, width = rgb.shape[:2]
    inputs = processor(images=Image.fromarray(rgb), return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        with torch.autocast(device_type=device, enabled=not fp32):
            predicted = model(**inputs).predicted_depth
    predicted = F.interpolate(
        predicted.unsqueeze(1).float(),
        size=(height, width),
        mode="bicubic",
        align_corners=False,
    )
    return predicted[0, 0].detach().float().cpu().numpy()
