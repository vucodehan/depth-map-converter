"""Map choices and the settings that change the pixels."""

from __future__ import annotations

from dataclasses import dataclass


MAP_LABELS = {
    "depth": "Depth",
    "clay": "Clay",
    "canny": "Canny",
    "lineart": "Line art",
    "lineart_anime": "Line art (anime)",
    "mlsd": "MLSD (straight lines)",
    "softedge": "Soft edge",
    "scribble": "Scribble",
    "pose": "Pose (OpenPose)",
    "normal": "Normal",
}

BASE_CHOICES = [(label, key) for key, label in MAP_LABELS.items()]
OVERLAY_CHOICES = [("None", "none")] + BASE_CHOICES

# depth: grayscale disparity. clay: shaded gray form from that depth.
# edge: marks. pose/normal: color images.
MAP_KIND = {
    "depth": "depth",
    "clay": "clay",
    "canny": "edge",
    "lineart": "edge",
    "lineart_anime": "edge",
    "mlsd": "edge",
    "softedge": "edge",
    "scribble": "edge",
    "pose": "pose",
    "normal": "color",
}

BLEND_CHOICES = [
    ("Alpha blend", "blend"),
    ("Transparent overlay", "transparent"),
]

ENCODER_CHOICES = [
    ("Auto", "auto"),
    ("Small (fast, Apache-2.0)", "vits"),
    ("Base", "vitb"),
    ("Large (best, needs more VRAM)", "vitl"),
]

# Bottom to top. Form maps sit under lines and the pose.
STACK_ORDER = [
    "clay",
    "depth",
    "normal",
    "canny",
    "lineart",
    "lineart_anime",
    "mlsd",
    "softedge",
    "scribble",
    "pose",
]

PRIMARY_MAPS = ["clay", "depth", "canny", "pose"]


@dataclass
class Recipe:
    layer1: str = "depth"
    layer1_opacity: float = 1.0
    layer1_mode: str = "blend"
    layer2: str = "none"
    layer2_opacity: float = 0.4
    layer2_mode: str = "transparent"
    layer3: str = "none"
    layer3_opacity: float = 0.4
    layer3_mode: str = "transparent"
    layer4: str = "none"
    layer4_opacity: float = 0.4
    layer4_mode: str = "transparent"
    clay_relief: float = 40.0
    encoder: str = "auto"
    input_size: int = 392
    max_side: int = 1280
    contrast: float = 1.0
    blur: float = 0.0
    invert: bool = False
    clip_extremes: bool = True
    dither: bool = True
    keep_audio: bool = True
    crf: int = 14
    save_frames: bool = False
    save_npz: bool = False
    max_seconds: float = 60
    target_fps: float = -1
    canny_low: int = 100
    canny_high: int = 200
    lineart_coarse: bool = True
    mlsd_score: float = 0.1
    mlsd_distance: float = 0.1
    softedge: str = "hed"
    pose_hands: bool = False
    pose_face: bool = False
    detect_resolution: int = 512

    def stack(self) -> list[tuple[str, float, str]]:
        """Bottom to top: (map name, opacity, blend mode)."""
        return [
            (self.layer1, float(self.layer1_opacity), self.layer1_mode or "blend"),
            (self.layer2, float(self.layer2_opacity), self.layer2_mode or "transparent"),
            (self.layer3, float(self.layer3_opacity), self.layer3_mode or "transparent"),
            (self.layer4, float(self.layer4_opacity), self.layer4_mode or "transparent"),
        ]

    def selected_maps(self) -> list[str]:
        names: list[str] = []
        for name, opacity, _mode in self.stack():
            if not name or name == "none" or float(opacity) <= 0:
                continue
            if name not in names:
                names.append(name)
        return names

    def needs_depth(self) -> bool:
        names = set(self.selected_maps())
        return "depth" in names or "clay" in names

    def tag(self) -> str:
        parts = [name for name, opacity, _mode in self.stack() if name and name != "none" and float(opacity) > 0]
        return "+".join(parts) if parts else "depth"


def maps_to_layers(selected: list[str] | None) -> tuple[list[tuple[str, float, str]], list[str]]:
    """Turn the checked maps into up to four layers, bottom to top.

    Clay is under depth. Depth over clay is a transparent overlay so the far
    background stays clay. Canny and pose sit on top with their backgrounds removed.
    """
    chosen_set = set(selected or [])
    chosen = [name for name in STACK_ORDER if name in chosen_set]
    overflow = chosen[4:]
    chosen = chosen[:4]
    layers: list[tuple[str, float, str]] = []
    for index, name in enumerate(chosen):
        if index == 0:
            layers.append((name, 1.0, "blend"))
        elif name == "depth" and "clay" in chosen:
            layers.append((name, 0.55, "transparent"))
        elif name == "pose":
            layers.append((name, 0.85, "transparent"))
        elif name == "canny":
            layers.append((name, 0.5, "transparent"))
        else:
            layers.append((name, 0.45, "transparent"))
    while len(layers) < 4:
        layers.append(("none", 0.4, "transparent"))
    return layers, overflow
