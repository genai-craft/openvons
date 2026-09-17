"""Smoke test: encoder layout, forward, and numerical equivalence of the 4 multi-question modes."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from openvons.core import Noul, Choice, Score, to_questions
from openvons.lm.models.decision_model import DecisionModel, DecisionModelConfig
from openvons.lm.backends.model_backend import ModelBackend, MODES

name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-0.6B"
pooling = sys.argv[2] if len(sys.argv) > 2 else "last"
m = DecisionModel(DecisionModelConfig(model_name=name, pooling=pooling, head="embed"))
be = ModelBackend(m)
state = "商品がまだ届いていません。注文番号は 12345 で、3日前に発送済みと表示されています。"
qs = to_questions({
    "urgent": Noul("緊急対応が必要か"),
    "department": Choice({"shipping": "配送に関する問い合わせ", "billing": "請求・支払い", "returns": "返品・返金", "other": "その他"}),
    "severity": Score(["問題なし", "軽度", "重大"]),
    "lang_ja": Noul("日本語で書かれているか"),
})
e = m.dtok.encode(state, qs[1])
print("ids", len(e.ids), "state_len", e.state_len, "option_pos", e.option_pos, "decision_pos", e.decision_pos)
print("option tokens:", [m.dtok.tok.decode([e.ids[p]]) for p in e.option_pos], "| decision tok:", repr(m.dtok.tok.decode([e.ids[e.decision_pos]])))
ref = be.logits(state, qs, "naive")
for mode in MODES:
    for _ in range(2):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        lg = be.logits(state, qs, mode)
        torch.cuda.synchronize(); dt = (time.perf_counter() - t0) * 1000
    diff = max((a - b).abs().max().item() for a, b in zip(ref, lg))
    print(f"{mode:11s} {dt:7.1f} ms  max|diff vs naive| = {diff:.4f}  logits[0]={[round(x,3) for x in lg[0].tolist()]}")
