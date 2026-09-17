"""EXP-003: accuracy on glaive test samples whose tool set is fully unseen in training (choice generalization)."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openvons.core.formats import read_jsonl
ROOT = Path(__file__).resolve().parents[1]
train = read_jsonl(str(ROOT / "data/store/processed/glaive_tools/train.jsonl"))
test = read_jsonl(str(ROOT / "data/store/processed/glaive_tools/test.jsonl"))
seen = {o.id for s in train for o in s.question.options}
unseen = np.array([all(o.id == "none" or o.id not in seen for o in s.question.options) for s in test])
multi = np.array([s.question.n > 2 for s in test])
y = np.array([s.label for s in test])
print(f"test={len(test)} unseen-tool samples={unseen.sum()} multi-tool samples={multi.sum()} unseen&multi={(unseen & multi).sum()}")
for exp in sys.argv[1:]:
    P = np.load(f"/data/decision_model/checkpoints/{exp}/probs_test.npy")
    pred = P.argmax(1)
    acc = lambda m: (pred[m] == y[m]).mean() if m.sum() else float("nan")
    print(f"{exp:45s} all={acc(np.ones_like(unseen)):.4f} unseen={acc(unseen):.4f} multi={acc(multi):.4f} unseen&multi={acc(unseen & multi):.4f}")
