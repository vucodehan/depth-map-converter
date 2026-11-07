"""Create the local environment, fetch Video Depth Anything, and download weights.

Run this once:

    python install.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parent
VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
VDA = ROOT / "third_party" / "Video-Depth-Anything"
VDA_PIN = "4f5ae23172ba60fd7bc11ef671cca678842c7072"
VDA_REPO = "https://github.com/DepthAnything/Video-Depth-Anything.git"
CHECKPOINTS = ROOT / "checkpoints"

WEIGHTS = [
    (
        "video_depth_anything_vits.pth",
        "https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth",
        20_000_000,
    ),
    (
        "video_depth_anything_vitb.pth",
        "https://huggingface.co/depth-anything/Video-Depth-Anything-Base/resolve/main/video_depth_anything_vitb.pth",
        80_000_000,
    ),
    (
        "video_depth_anything_vitl.pth",
        "https://huggingface.co/depth-anything/Video-Depth-Anything-Large/resolve/main/video_depth_anything_vitl.pth",
        200_000_000,
    ),
]

STILL_REPOS = [
    "depth-anything/Depth-Anything-V2-Small-hf",
    "depth-anything/Depth-Anything-V2-Base-hf",
    "depth-anything/Depth-Anything-V2-Large-hf",
]

TORCH_INDEXES = [
    "https://download.pytorch.org/whl/cu128",
    "https://download.pytorch.org/whl/cu130",
]


def _python_ok() -> None:
    if sys.version_info < (3, 10):
        raise SystemExit("Python 3.10 or newer is required.")


def _in_venv() -> bool:
    return VENV_PY.exists() and Path(sys.executable).resolve() == VENV_PY.resolve()


def _pip(*args: str) -> None:
    # Skip machine-wide pip indexes. This PC has an extra index that does not
    # resolve, and pip waits on it for every file.
    script = r"""
import pip._internal.configuration as cfg

def _iter(self):
    for kind, files in cfg.Configuration._orig_iter_config_files(self):
        if kind in ("global", "user"):
            continue
        yield kind, files

cfg.Configuration._orig_iter_config_files = cfg.Configuration.iter_config_files
cfg.Configuration.iter_config_files = _iter
import sys
from pip._internal.cli.main import main
sys.argv = ["pip", *sys.argv[1:]]
raise SystemExit(main())
"""
    subprocess.check_call([sys.executable, "-c", script, *args])


def _cuda_works() -> bool:
    code = (
        "import torch\n"
        "print(torch.__version__)\n"
        "ok = torch.cuda.is_available()\n"
        "print('CUDA', ok)\n"
        "print(torch.cuda.get_device_name(0) if ok else 'cpu')\n"
        "raise SystemExit(0 if ok else 1)\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    print(proc.stdout)
    if proc.stderr:
        print(proc.stderr)
    return proc.returncode == 0 and "CUDA True" in proc.stdout


def _install_torch() -> None:
    if shutil.which("nvidia-smi") is None:
        print("No nvidia-smi. Installing the default PyTorch build for CPU.")
        _pip("install", "torch", "torchvision")
        return
    for index in TORCH_INDEXES:
        print(f"Installing PyTorch from {index}")
        try:
            _pip(
                "install",
                "--upgrade",
                "torch",
                "torchvision",
                "--index-url",
                index,
                "--extra-index-url",
                "https://pypi.org/simple",
            )
        except subprocess.CalledProcessError:
            print(f"Could not install from {index}")
            continue
        if _cuda_works():
            print("CUDA is available.")
            return
        print("That PyTorch build does not see the GPU.")
    raise SystemExit(
        "PyTorch installed, but this GPU is not visible. "
        "An RTX 50-series card needs a CUDA 12.8 or newer wheel."
    )


def _clone_depth() -> None:
    entry = VDA / "video_depth_anything" / "video_depth.py"
    if entry.exists():
        print(f"Video Depth Anything source already present at {VDA}")
        return
    if shutil.which("git") is None:
        raise SystemExit("git is required to fetch Video Depth Anything.")
    VDA.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["git", "clone", VDA_REPO, str(VDA)])
    subprocess.check_call(["git", "checkout", VDA_PIN], cwd=VDA)


def _download(url: str, dest: Path, minimum: int) -> None:
    if dest.exists() and dest.stat().st_size >= minimum:
        print(f"Already downloaded {dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    if partial.exists():
        partial.unlink()
    print(f"Downloading {dest.name}")
    request = urllib.request.Request(url, headers={"User-Agent": "map-converter"})
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as handle:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        next_report = 64 * 1024 * 1024
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            done += len(chunk)
            if done >= next_report:
                if total:
                    print(f"  {dest.name}: {done // (1024 * 1024)} / {total // (1024 * 1024)} MB")
                else:
                    print(f"  {dest.name}: {done // (1024 * 1024)} MB")
                next_report += 64 * 1024 * 1024
    if partial.stat().st_size < minimum:
        partial.unlink(missing_ok=True)
        raise SystemExit(f"Download of {dest.name} looks too small. Check the link: {url}")
    partial.replace(dest)
    print(f"Saved {dest}")


def _download_video_weights() -> None:
    for name, url, minimum in WEIGHTS:
        _download(url, CHECKPOINTS / name, minimum)


def _prefetch_still_models() -> None:
    os.environ["HF_HOME"] = str(CHECKPOINTS / "hf")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(CHECKPOINTS / "hf" / "hub")
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    for repo in STILL_REPOS:
        print(f"Fetching still depth model {repo}")
        AutoImageProcessor.from_pretrained(repo, use_fast=True)
        AutoModelForDepthEstimation.from_pretrained(repo)


def _install_inside_venv() -> None:
    print("Upgrading pip")
    _pip("install", "--upgrade", "pip")
    print("Installing Python packages")
    _pip("install", "-r", str(ROOT / "requirements.txt"))
    print("Installing a CUDA build of PyTorch")
    _install_torch()
    print("Fetching Video Depth Anything")
    _clone_depth()
    print("Downloading video depth checkpoints")
    _download_video_weights()
    print("Downloading still-image depth models")
    _prefetch_still_models()
    print("")
    print("Install finished. Start the app with:")
    print("    python start.py")


def main() -> None:
    _python_ok()
    inside = "--in-venv" in sys.argv or _in_venv()
    if not inside:
        if not VENV_PY.exists():
            print(f"Creating {ROOT / '.venv'}")
            subprocess.check_call([sys.executable, "-m", "venv", str(ROOT / ".venv")])
        print("Switching into the project environment")
        raise SystemExit(subprocess.call([str(VENV_PY), str(Path(__file__).resolve()), "--in-venv"]))
    _install_inside_venv()


if __name__ == "__main__":
    main()
