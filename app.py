"""Launch the local interface. Prefer start.py so the GPU is chosen first."""

import logging
import os
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / "checkpoints" / "hf"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(ROOT / "checkpoints" / "hf" / "hub"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
warnings.filterwarnings("ignore", message=r"torch\.jit\.script")
warnings.filterwarnings("ignore", message=r"Importing from timm\.models")
warnings.filterwarnings("ignore", message=r"The module 'mediapipe' is not installed")
warnings.filterwarnings("ignore", message=r"Overwriting tiny_vit_")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)


def main() -> None:
    from mapconv.ui import build

    port = int(os.environ.get("MAPCONV_PORT", "7860"))
    share = os.environ.get("MAPCONV_SHARE", "") == "1"
    demo = build()
    demo.queue()
    print(f"Map Converter is at http://127.0.0.1:{port}")
    demo.launch(
        server_name="127.0.0.1",
        server_port=port,
        share=share,
        inbrowser=True,
        allowed_paths=[str(ROOT / "outputs")],
        show_error=True,
        quiet=True,
    )


if __name__ == "__main__":
    main()
