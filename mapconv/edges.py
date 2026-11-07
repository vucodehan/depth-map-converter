"""Edge, pose, and normal maps.

Canny is local OpenCV. The learned maps come from controlnet_aux and are
loaded the first time a preview asks for them.
"""

from __future__ import annotations

import importlib
import inspect
import warnings

import cv2
import numpy as np

from mapconv.device import torch_device
from mapconv.grade import to_rgb
from mapconv.recipe import MAP_KIND, Recipe

ANNOTATORS = "lllyasviel/Annotators"
_detectors: dict[str, object] = {}


def uses_gpu_model(map_name: str, recipe: Recipe) -> bool:
    if map_name in {"none", "depth", "clay", "canny"}:
        return False
    return True


def _detector_key(map_name: str, recipe: Recipe) -> str:
    if map_name in {"softedge", "scribble"} and recipe.softedge == "pidinet":
        return "pidinet"
    if map_name in {"softedge", "scribble"}:
        return "hed"
    return map_name


_DETECTORS = {
    "lineart": ("controlnet_aux.lineart", "LineartDetector"),
    "lineart_anime": ("controlnet_aux.lineart_anime", "LineartAnimeDetector"),
    "mlsd": ("controlnet_aux.mlsd", "MLSDdetector"),
    "hed": ("controlnet_aux.hed", "HEDdetector"),
    "pidinet": ("controlnet_aux.pidi", "PidiNetDetector"),
    "normal": ("controlnet_aux.normalbae", "NormalBaeDetector"),
    "dwpose": ("controlnet_aux.dwpose", "DWposeDetector"),
    "openpose": ("controlnet_aux.open_pose", "OpenposeDetector"),
}


def _trust_local_checkpoints() -> None:
    """Let the annotator checkpoints load without the weights_only warning.

    Those files are plain weight dictionaries downloaded from the annotator
    repo. Passing weights_only=False explicitly avoids the environment-variable
    warning PyTorch prints when that switch is forced globally.
    """
    import torch

    if getattr(torch.load, "_mapconv_wrapped", False):
        return
    original = torch.load

    def load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)

    load._mapconv_wrapped = True
    torch.load = load


def _detector_class(key: str):
    module_name, class_name = _DETECTORS[key]
    _trust_local_checkpoints()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", DeprecationWarning)
        module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _load(key: str, device: str):
    if key in _detectors:
        return _detectors[key]
    if key == "pose":
        detector = _load_pose()
    elif key in _DETECTORS:
        detector = _detector_class(key).from_pretrained(ANNOTATORS)
    else:
        raise ValueError(f"No detector for {key}.")
    if hasattr(detector, "to"):
        detector.to(device)
    _detectors[key] = detector
    return detector


def _load_pose():
    errors = []
    for key in ("dwpose", "openpose"):
        try:
            return _detector_class(key).from_pretrained(ANNOTATORS)
        except Exception as exc:
            errors.append(str(exc))
    raise RuntimeError(
        "Could not load a pose model (DWPose, then OpenPose). " + " | ".join(errors)
    )


def _call(detector, rgb: np.ndarray, **kwargs) -> np.ndarray:
    signature = inspect.signature(detector.__call__)
    accepted = {}
    for key, value in kwargs.items():
        if key in signature.parameters:
            accepted[key] = value
    if "output_type" in signature.parameters:
        accepted["output_type"] = "np"
    result = detector(rgb, **accepted)
    if not isinstance(result, np.ndarray):
        result = np.asarray(result)
    return to_rgb(result)


def _resize(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    if rgb.shape[0] == height and rgb.shape[1] == width:
        return rgb
    return cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)


def canny(rgb: np.ndarray, low: int, high: int) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, int(low), int(high))
    return np.repeat(edges[:, :, None], 3, axis=2)


def run_map(map_name: str, rgb: np.ndarray, recipe: Recipe) -> np.ndarray:
    """Return an HxWx3 uint8 map at the frame size."""
    if map_name in {"none", "depth"}:
        raise ValueError(map_name)
    height, width = rgb.shape[:2]
    if map_name == "canny":
        low = min(int(recipe.canny_low), int(recipe.canny_high))
        high = max(int(recipe.canny_low), int(recipe.canny_high))
        return canny(rgb, low, high)

    key = _detector_key(map_name, recipe)
    detector = _load(key, torch_device())
    long_side = max(height, width)
    kwargs = {
        "detect_resolution": int(recipe.detect_resolution),
        "image_resolution": int(long_side),
    }
    if map_name == "lineart":
        kwargs["coarse"] = bool(recipe.lineart_coarse)
    elif map_name == "mlsd":
        kwargs["thr_v"] = float(recipe.mlsd_score)
        kwargs["thr_d"] = float(recipe.mlsd_distance)
    elif map_name == "scribble":
        kwargs["scribble"] = True
    elif map_name == "pose":
        kwargs["include_body"] = True
        kwargs["include_hand"] = bool(recipe.pose_hands)
        kwargs["include_face"] = bool(recipe.pose_face)
    rendered = _call(detector, rgb, **kwargs)
    return _resize(rendered, width, height)


def kind_of(map_name: str) -> str:
    return MAP_KIND[map_name]
