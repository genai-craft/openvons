"""convaiinnovations/laya を openvons.lm の DecisionBackend として使う (任意、`pip install laya`)。

Laya は ModernBERT-large (395M) + 判断 head の「生成しない判断モデル」で、state と typed questions (choice / score / noul) を
1 回の forward で答える (Apache-2.0)。openvons と同じ思想なので、学習なしの基準線・cold start 用の backend として置く。
手元のテストセットでのゼロショット精度は docs/laya_eval.md (学習した 4B 凍結 + head より 13〜25pt 低い)。
"""
from __future__ import annotations

import time

from openvons.core.primitives import Question

from .base import Decision, DecisionBackend


class LayaBackend(DecisionBackend):
    name = "laya"

    def __init__(self, repo: str = "convaiinnovations/laya", subfolder: str | None = None):
        import laya  # 遅延 import (任意依存)
        self.agent = laya.load(repo, subfolder=subfolder) if subfolder else laya.load(repo)

    @staticmethod
    def _question(q: Question) -> dict:
        ids = q.ids
        if len(ids) == 2 and set(i.lower() for i in ids) <= {"true", "false", "yes", "no"}:
            return {"type": "noul", "instructions": q.text}
        return {"type": "choice", "instructions": q.text, "criteria": {o.id: (o.description or o.id) for o in q.options}}

    def decide(self, state: str, questions: list[Question]) -> list[Decision]:
        qs = {f"q{i}": self._question(q) for i, q in enumerate(questions)}
        t0 = time.perf_counter()
        out = self.agent.predict(state, qs)
        ms = (time.perf_counter() - t0) * 1e3
        answers = out.get("answers", out)
        res = []
        for i, q in enumerate(questions):
            a = answers[f"q{i}"]
            if a.get("type") == "noul" or "noul" in a:
                p = a.get("noul")
                p = p if isinstance(p, float) else float(a.get("confidence", 0.5))
                # options の並びが (true, false) / (yes, no) 前提
                first_true = q.ids[0].lower() in ("true", "yes")
                probs = [p, 1 - p] if first_true else [1 - p, p]
            else:
                pr = a.get("probabilities") or {}
                probs = [float(pr.get(o.id, 0.0)) for o in q.options]
                s = sum(probs)
                probs = [x / s for x in probs] if s > 0 else [1.0 / len(probs)] * len(probs)
            res.append(Decision(probs=probs, latency_ms=ms / max(len(questions), 1), info={"backend": "laya", "action": a.get("action")}))
        return res
