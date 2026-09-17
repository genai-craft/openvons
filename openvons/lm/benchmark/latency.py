"""TASK-011 / TASK-012: latency vs number of questions (EXP-007 / EXP-008).

Decision model: naive / batched / kv_shared / block_diag for q in {1,2,4,8,16}.
LLM baseline : (a) one JSON-schema request with all questions, (b) per-question guided-choice requests fired concurrently.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from openvons.core.primitives import Question, Option

ROOT = Path(__file__).resolve().parents[1]

STATE_SHORT = ("Customer message: My order #48213 was marked as delivered yesterday but I have not received anything. "
               "The tracking page says it was left at the front door. I live in an apartment building and there is no package. "
               "I need this for a trip on Friday. Account: premium member since 2021, 14 previous orders, no prior complaints.")
STATE_LONG = STATE_SHORT + " " + " ".join(f"Previous interaction {i}: the customer asked about shipping times and was answered within one day." for i in range(40))

QUESTION_BANK = [
    ("choice", "Which department should handle this?", ["shipping", "billing", "returns", "account", "other"]),
    ("noul", "Is urgent handling required?", None),
    ("score", "How severe is the problem?", ["none", "minor", "moderate", "severe"]),
    ("noul", "Is the customer angry?", None),
    ("choice", "What is the best next action?", ["refund", "reship", "investigate_with_carrier", "ask_for_more_info", "escalate"]),
    ("noul", "Should a human agent be involved?", None),
    ("score", "How likely is fraud?", ["very unlikely", "unlikely", "possible", "likely", "very likely"]),
    ("choice", "What is the customer's sentiment?", ["positive", "neutral", "negative", "furious"]),
    ("noul", "Does the message mention a deadline?", None),
    ("choice", "Which channel should we reply on?", ["email", "sms", "phone", "chat"]),
    ("noul", "Is a discount coupon appropriate?", None),
    ("score", "Priority level", ["P4", "P3", "P2", "P1", "P0"]),
    ("noul", "Is the delivery address likely wrong?", None),
    ("choice", "Which template fits best?", ["apology_reship", "apology_refund", "clarify_address", "carrier_investigation", "generic"]),
    ("noul", "Is this a repeat complaint?", None),
    ("score", "Customer churn risk", ["low", "medium", "high"]),
]


def make_questions(n: int) -> list[Question]:
    qs = []
    for i in range(n):
        t, text, opts = QUESTION_BANK[i % len(QUESTION_BANK)]
        if t == "noul":
            q = Question("noul", text, [Option("true", "yes"), Option("false", "no")], key=f"q{i}")
        elif t == "choice":
            q = Question("choice", text, [Option(o, o.replace("_", " ")) for o in opts], key=f"q{i}")
        else:
            q = Question("score", text, [Option(str(j), o) for j, o in enumerate(opts)], key=f"q{i}")
        qs.append(q)
    return qs


def timeit(fn, reps: int, warm: int = 5) -> dict:
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1000)
    return {"p50": float(np.percentile(ts, 50)), "p95": float(np.percentile(ts, 95)), "mean": float(np.mean(ts))}


def bench_model(a, state, counts):
    import torch
    from openvons.lm.models.decision_model import DecisionModel, DecisionModelConfig
    from openvons.lm.backends.model_backend import ModelBackend, MODES
    if a.ckpt:
        m = DecisionModel.from_checkpoint(a.ckpt)
    else:
        m = DecisionModel(DecisionModelConfig(model_name=a.model, pooling=a.pooling, head=a.head))
    be = ModelBackend(m)
    out = {}
    for mode in (a.modes or MODES):
        for n in counts:
            qs = make_questions(n)

            def fn():
                be.logits(state, qs, mode)
                torch.cuda.synchronize()

            r = timeit(fn, a.reps)
            out[f"{mode}/{n}"] = r
            print(f"model {mode:11s} q={n:2d}  p50={r['p50']:7.1f} ms  p95={r['p95']:7.1f} ms")
    out["state_tokens"] = len(m.dtok.encode_state(state))
    return out


async def bench_llm(a, state, counts):
    from openvons.lm.backends.llm_backend import LLMBackend
    be = LLMBackend(f"http://127.0.0.1:{a.port}/v1", a.model, mode="logprob", concurrency=64)
    out = {}
    for n in counts:
        qs = make_questions(n)
        # (a) one JSON-schema request
        for _ in range(3):
            await be.adecide_json(state, qs)
        ts = []
        for _ in range(a.reps):
            t0 = time.perf_counter(); await be.adecide_json(state, qs); ts.append((time.perf_counter() - t0) * 1000)
        out[f"json/{n}"] = {"p50": float(np.percentile(ts, 50)), "p95": float(np.percentile(ts, 95))}
        # (b) per-question guided-choice logprob requests in parallel
        for _ in range(3):
            await asyncio.gather(*[be.adecide_one(state, q) for q in qs])
        ts = []
        for _ in range(a.reps):
            t0 = time.perf_counter(); await asyncio.gather(*[be.adecide_one(state, q) for q in qs]); ts.append((time.perf_counter() - t0) * 1000)
        out[f"parallel_logprob/{n}"] = {"p50": float(np.percentile(ts, 50)), "p95": float(np.percentile(ts, 95))}
        print(f"llm  json q={n:2d} p50={out[f'json/{n}']['p50']:7.1f} ms | parallel-logprob p50={out[f'parallel_logprob/{n}']['p50']:7.1f} ms")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["llm", "model"], required=True)
    ap.add_argument("--counts", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--state", choices=["short", "long"], default="short")
    ap.add_argument("--exp", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--pooling", default="last")
    ap.add_argument("--head", default="embed")
    ap.add_argument("--modes", nargs="+", default=None)
    ap.add_argument("--port", type=int, default=8300)
    a = ap.parse_args()
    state = STATE_SHORT if a.state == "short" else STATE_LONG
    res = bench_model(a, state, a.counts) if a.backend == "model" else asyncio.run(bench_llm(a, state, a.counts))
    name = a.exp or f"exp007_latency_{a.backend}_{Path(a.model).name}_{a.state}"
    json.dump({"experiment": name, "backend": a.backend, "model": a.model, "state": a.state, "results": res},
              open(ROOT / "experiments" / f"{name}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
