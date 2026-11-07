"""Choose a GPU, then open the converter.

    python start.py
    python start.py --gpu 0
    python start.py --cpu
    python start.py --yes
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parent
VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
PREFS = ROOT / "preferences.json"


def _relaunch_in_venv() -> None:
    if not VENV_PY.exists():
        raise SystemExit("The environment is missing. Run install.py first.")
    if Path(sys.executable).resolve() == VENV_PY.resolve():
        return
    raise SystemExit(subprocess.call([str(VENV_PY), str(Path(__file__).resolve()), *sys.argv[1:]]))


def _load_prefs() -> dict:
    if not PREFS.exists():
        return {}
    try:
        return json.loads(PREFS.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save_prefs(data: dict) -> None:
    PREFS.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _gpus() -> list[tuple[int, str, str]]:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return []
    if proc.returncode != 0:
        return []
    found = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            index = int(parts[0])
        except ValueError:
            continue
        found.append((index, parts[1], parts[2]))
    return found


def _ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except EOFError:
        return ""


def _resolve(args: argparse.Namespace) -> tuple[int | None, str, int, bool]:
    prefs = _load_prefs()
    gpus = _gpus()
    if args.cpu and args.gpu is not None:
        raise SystemExit("Pass either --cpu or --gpu, not both.")

    interactive = sys.stdin.isatty() and not args.yes and args.gpu is None and not args.cpu
    gpu: int | None
    precision: str
    port: int

    if interactive:
        print("Map Converter")
        if not gpus:
            print("No NVIDIA GPU detected. Using CPU.")
            gpu = None
        else:
            print("GPUs:")
            for index, name, memory in gpus:
                print(f"  [{index}] {name} ({memory})")
            default_gpu = prefs.get("gpu", gpus[0][0])
            if default_gpu not in {item[0] for item in gpus}:
                default_gpu = gpus[0][0]
            typed = _ask(f"GPU index, or cpu [{default_gpu}]: ")
            if typed.lower() == "cpu":
                gpu = None
            elif typed == "":
                gpu = int(default_gpu)
            else:
                gpu = int(typed)
                if gpu not in {item[0] for item in gpus}:
                    raise SystemExit(f"GPU {gpu} is not in the list above.")
        default_precision = "fp32" if gpu is None else str(prefs.get("precision") or "fp16")
        if default_precision not in {"fp16", "fp32"}:
            default_precision = "fp16"
        typed = _ask(f"Precision fp16 or fp32 [{default_precision}]: ").lower()
        precision = typed if typed in {"fp16", "fp32"} else default_precision
        default_port = int(prefs.get("port") or 7860)
        typed = _ask(f"Port [{default_port}]: ")
        port = int(typed) if typed else default_port
    else:
        if args.cpu or not gpus:
            gpu = None
        else:
            gpu = gpus[0][0] if args.gpu is None else args.gpu
            known = {item[0] for item in gpus}
            if gpu not in known:
                raise SystemExit(f"GPU {gpu} was not found. Detected: {sorted(known)}")
        if gpu is None:
            precision = "fp32"
        elif args.fp32:
            precision = "fp32"
        elif args.fp16:
            precision = "fp16"
        else:
            precision = str(prefs.get("precision") or "fp16")
            if precision not in {"fp16", "fp32"}:
                precision = "fp16"
        port = int(args.port or prefs.get("port") or 7860)

    if args.fp32:
        precision = "fp32"
    if args.fp16 and gpu is not None:
        precision = "fp16"
    if gpu is None:
        precision = "fp32"
    if args.port:
        port = int(args.port)
    if port < 1 or port > 65535:
        raise SystemExit(f"Port {port} is not valid.")

    _save_prefs({"gpu": gpu, "precision": precision, "port": port})
    return gpu, precision, port, bool(args.share)


def main() -> None:
    _relaunch_in_venv()
    parser = argparse.ArgumentParser(description="Start the map converter")
    parser.add_argument("--gpu", type=int, help="NVIDIA GPU index from nvidia-smi")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument("--fp32", action="store_true", help="Full precision")
    parser.add_argument("--fp16", action="store_true", help="Half precision (default on GPU)")
    parser.add_argument("--port", type=int, help="Local port, default 7860")
    parser.add_argument("--share", action="store_true", help="Create a temporary public Gradio link")
    parser.add_argument("--yes", action="store_true", help="Use saved settings and do not prompt")
    args = parser.parse_args()

    gpu, precision, port, share = _resolve(args)
    os.environ["HF_HOME"] = str(ROOT / "checkpoints" / "hf")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(ROOT / "checkpoints" / "hf" / "hub")
    os.environ["MAPCONV_PRECISION"] = precision
    os.environ["MAPCONV_PORT"] = str(port)
    os.environ["MAPCONV_SHARE"] = "1" if share else "0"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    if gpu is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print(f"Starting on CPU, {precision}, port {port}")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        print(f"Starting on GPU {gpu}, {precision}, port {port}")
    print("The page opens at http://127.0.0.1:%d" % port)

    from app import main as launch

    launch()


if __name__ == "__main__":
    main()
