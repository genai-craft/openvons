"""TASK-009 wrapper: head-only training on cached features. Same CLI as training/train.py (mode=head)."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jev.lm.training.train import build_parser, run
if __name__ == "__main__":
    a = build_parser().parse_args(); a.mode = "head"; run(a)
