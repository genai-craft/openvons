"""Method C: combine several teachers' distributions (mean or weighted mean)."""
from __future__ import annotations

from jev.core.formats import Sample


def ensemble(runs: list[list[Sample]], weights: list[float] | None = None) -> list[Sample]:
    weights = weights or [1.0] * len(runs)
    z = sum(weights)
    out = []
    for group in zip(*runs):
        n = group[0].question.n
        probs = [sum(w * g.target_probs[i] for w, g in zip(weights, group)) / z for i in range(n)]
        disagreement = max(abs(a.target_probs[i] - b.target_probs[i]) for a in group for b in group for i in range(n))
        meta = dict(group[0].meta or {})
        meta["teacher_disagreement"] = round(disagreement, 4)
        out.append(Sample(group[0].id, group[0].state, group[0].question, group[0].label, probs, meta))
    return out
