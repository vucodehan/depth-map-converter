"""Project folders. Weights and the upstream depth repo stay next to the app."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
THIRD_PARTY = ROOT / "third_party"
VDA_ROOT = THIRD_PARTY / "Video-Depth-Anything"
CHECKPOINTS = ROOT / "checkpoints"
HF_HOME = CHECKPOINTS / "hf"
OUTPUTS = ROOT / "outputs"
WORK = ROOT / "work"
PREFERENCES = ROOT / "preferences.json"

VDA_PIN = "4f5ae23172ba60fd7bc11ef671cca678842c7072"
VDA_REPO = "https://github.com/DepthAnything/Video-Depth-Anything.git"

VIDEO_WEIGHTS = {
    "vits": (
        "video_depth_anything_vits.pth",
        "https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth",
        20_000_000,
    ),
    "vitb": (
        "video_depth_anything_vitb.pth",
        "https://huggingface.co/depth-anything/Video-Depth-Anything-Base/resolve/main/video_depth_anything_vitb.pth",
        80_000_000,
    ),
    "vitl": (
        "video_depth_anything_vitl.pth",
        "https://huggingface.co/depth-anything/Video-Depth-Anything-Large/resolve/main/video_depth_anything_vitl.pth",
        200_000_000,
    ),
}

STILL_MODELS = {
    "vits": "depth-anything/Depth-Anything-V2-Small-hf",
    "vitb": "depth-anything/Depth-Anything-V2-Base-hf",
    "vitl": "depth-anything/Depth-Anything-V2-Large-hf",
}
