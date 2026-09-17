"""LoRA training wrapper. Same CLI as training/train.py (mode=lora)."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openvons.lm.training.train import build_parser, run
if __name__ == "__main__":
    a = build_parser().parse_args(); a.mode = "lora"; a.epochs = a.epochs if a.epochs != 10 else 2; a.lr = a.lr if a.lr != 1e-3 else 1e-4; run(a)
