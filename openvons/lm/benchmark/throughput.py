"""1 GPU で 1 秒あたり何件の「問い合わせ」を捌けるか (コールセンター的 triage シナリオ)。

1 件の問い合わせ = 1 つの state に対して N 個の質問 (部署 / 緊急度 / 感情 / エスカレーション / 次アクション / 深刻度)。
Decision Model: 問い合わせをバッチにまとめ、各問い合わせ内の質問は block_diag で 1 forward。
LLM baseline : vLLM に質問ごと guided-choice リクエストを高い並列度で投げる。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from openvons.lm.benchmark.latency import make_questions, STATE_SHORT, STATE_LONG

ROOT = Path(__file__).resolve().parents[1]


def bench_model(a, states, qs):
    import torch
    from openvons.lm.models.decision_model import DecisionModel, DecisionModelConfig
    from openvons.lm.backends.model_backend import ModelBackend
    m = (DecisionModel.from_checkpoint(a.ckpt) if a.ckpt
         else DecisionModel(DecisionModelConfig(model_name=a.model, pooling=a.pooling, head="embed")))
    be = ModelBackend(m)

    if a.batch_contacts <= 1:
        run = lambda chunk: [be.logits(s, qs, a.mode) for s in chunk]
    else:
        # 複数問い合わせ × 複数質問をまとめて 1 forward (キューに溜まった分を一括処理する想定)
        @torch.no_grad()
        def run(chunk):
            encs = [m.dtok.encode(s, q) for s in chunk for q in qs]
            b = m.dtok.collate(encs, m.device)
            return m(b, qs[0].type)

    chunks = [states[i : i + max(1, a.batch_contacts)] for i in range(0, len(states), max(1, a.batch_contacts))]
    run(chunks[0]); torch.cuda.synchronize()          # warm-up
    t0 = time.perf_counter()
    for c in chunks:
        run(c)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    n = len(states)
    return {"contacts": n, "questions_per_contact": len(qs), "batch_contacts": a.batch_contacts,
            "wall_s": round(wall, 2), "contacts_per_s": round(n / wall, 1),
            "decisions_per_s": round(n * len(qs) / wall, 1), "ms_per_contact": round(wall / n * 1000, 2),
            "gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}


async def bench_llm(a, states, qs):
    from openvons.lm.backends.llm_backend import LLMBackend
    be = LLMBackend(f"http://127.0.0.1:{a.port}/v1", a.model, mode="logprob", concurrency=a.concurrency)
    await asyncio.gather(*[be.adecide_one(states[0], q) for q in qs])
    t0 = time.perf_counter()
    await asyncio.gather(*[be.adecide_one(s, q) for s in states for q in qs])
    wall = time.perf_counter() - t0
    return {"contacts": len(states), "questions_per_contact": len(qs), "wall_s": round(wall, 2),
            "contacts_per_s": round(len(states) / wall, 1), "decisions_per_s": round(len(states) * len(qs) / wall, 1),
            "ms_per_contact": round(wall / len(states) * 1000, 2), "concurrency": a.concurrency}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["model", "llm"], required=True)
    ap.add_argument("--contacts", type=int, default=200)
    ap.add_argument("--questions", type=int, default=6)
    ap.add_argument("--state", choices=["short", "long"], default="short")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--pooling", default="last")
    ap.add_argument("--mode", default="block_diag")
    ap.add_argument("--port", type=int, default=8300)
    ap.add_argument("--concurrency", type=int, default=128)
    ap.add_argument("--batch_contacts", type=int, default=1)
    ap.add_argument("--exp", default=None)
    a = ap.parse_args()
    base = STATE_SHORT if a.state == "short" else STATE_LONG
    states = [f"[case {i:04d}] " + base for i in range(a.contacts)]   # prefix caching を効かせない
    qs = make_questions(a.questions)
    r = bench_model(a, states, qs) if a.backend == "model" else asyncio.run(bench_llm(a, states, qs))
    name = a.exp or f"exp013_throughput_{a.backend}_{a.state}_q{a.questions}"
    r.update(experiment=name, backend=a.backend, model=a.model, mode=a.mode if a.backend == "model" else "logprob", state=a.state)
    json.dump(r, open(ROOT / "experiments" / f"{name}.json", "w"), indent=2)
    print(json.dumps(r, indent=None))


if __name__ == "__main__":
    main()
