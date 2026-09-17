"""Method C: average several teachers' soft labels -> data/store/generated/<task>/<name>/<split>.jsonl"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jev.core.formats import read_jsonl, write_jsonl
from jev.lm.teacher.ensemble import ensemble
ROOT = Path(__file__).resolve().parents[1]
task, name, *teachers = sys.argv[1:]
for split in ["train", "valid", "test"]:
    runs = [read_jsonl(str(ROOT / "data/store/generated" / task / t / f"{split}.jsonl")) for t in teachers]
    out = ensemble(runs)
    d = ROOT / "data/store/generated" / task / name; d.mkdir(parents=True, exist_ok=True)
    write_jsonl(str(d / f"{split}.jsonl"), out)
    acc = sum(max(range(s.question.n), key=lambda i: s.target_probs[i]) == s.label for s in out) / len(out)
    hi = sum((s.meta or {}).get("teacher_disagreement", 0) > 0.5 for s in out)
    print(f"{split}: n={len(out)} ensemble-acc={acc:.4f} disagreement>0.5: {hi}")
