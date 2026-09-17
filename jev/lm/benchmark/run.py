"""Benchmark a backend on a task's test split: accuracy / F1 / ECE / Brier / NLL / latency (Phase 0 + §8).

  LLM baseline (EXP-001):
    python benchmark/run.py --backend llm --port 8300 --model qwen3-4b --mode logprob --task massive_scenario_en
  Decision model checkpoint:
    python benchmark/run.py --backend model --ckpt /data/decision_model/checkpoints/<exp> --task massive_scenario_en
Latency is measured per request with --concurrency 1 (true single-request latency); throughput with higher concurrency.
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
from tqdm import tqdm

from jev.core.metrics import all_metrics, latency_summary, reliability_table
from jev.core.temperature import TemperatureScaler
from jev.core.formats import read_jsonl
from jev.lm.training.dataset import task_path

ROOT = Path(__file__).resolve().parents[1]


async def run_llm(a, samples):
    from jev.lm.backends.llm_backend import LLMBackend
    be = LLMBackend(f"http://127.0.0.1:{a.port}/v1", a.model, mode=a.mode, n_samples=a.n_samples, concurrency=a.concurrency)
    # warm-up
    for s in samples[:3]:
        await be.adecide_one(s.state, s.question)
    probs, lat, usage, contents = [None] * len(samples), [0.0] * len(samples), [], [None] * len(samples)
    pbar = tqdm(total=len(samples), mininterval=2)

    async def one(i, s):
        d = await be.adecide_one(s.state, s.question)
        probs[i], lat[i] = d.probs, d.latency_ms
        u = d.info.get("usage") or {}
        usage.append((u.get("prompt_tokens", 0), u.get("completion_tokens", 0)))
        contents[i] = d.info.get("content")
        pbar.update(1)

    t0 = time.perf_counter()
    await asyncio.gather(*[one(i, s) for i, s in enumerate(samples)])
    wall = time.perf_counter() - t0
    pbar.close()
    return probs, lat, wall, {"prompt_tokens_mean": float(np.mean([u[0] for u in usage])), "completion_tokens_mean": float(np.mean([u[1] for u in usage]))}


def run_model(a, samples):
    import torch
    from jev.lm.models.decision_model import DecisionModel
    from jev.lm.backends.model_backend import ModelBackend
    m = DecisionModel.from_checkpoint(a.ckpt)
    if a.no_temperature:
        m.temperature = 1.0
    be = ModelBackend(m, mode="naive")
    for s in samples[:10]:
        be.decide(s.state, [s.question])
    probs, lat = [], []
    t0 = time.perf_counter()
    for s in tqdm(samples, mininterval=2):
        d = be.decide(s.state, [s.question])[0]
        probs.append(d.probs); lat.append(d.latency_ms)
    wall = time.perf_counter() - t0
    return probs, lat, wall, {"gpu_mem_alloc_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["llm", "model"], required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--exp", default=None)
    # llm
    ap.add_argument("--port", type=int, default=8300)
    ap.add_argument("--model", default="qwen3-4b")
    ap.add_argument("--mode", default="logprob", choices=["logprob", "greedy", "sample"])
    ap.add_argument("--n_samples", type=int, default=20)
    ap.add_argument("--concurrency", type=int, default=1)
    # model
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--no_temperature", action="store_true")
    ap.add_argument("--calibrate_on_valid", action="store_true", help="fit temperature on the valid split predictions (LLM)")
    a = ap.parse_args()

    samples = read_jsonl(str(task_path(a.task, a.split)), a.limit)
    N = max(s.question.n for s in samples)
    labels = np.array([s.label for s in samples])
    if a.backend == "llm":
        probs, lat, wall, extra = asyncio.run(run_llm(a, samples))
        name = a.exp or f"exp001_{a.task}_{a.model}_{a.mode}_c{a.concurrency}"
        meta = {"model": a.model, "mode": a.mode, "concurrency": a.concurrency, "head": "lm_head", "pooling": "autoregressive"}
    else:
        probs, lat, wall, extra = run_model(a, samples)
        cfg = json.load(open(Path(a.ckpt) / "config.json"))
        name = a.exp or f"bench_{Path(a.ckpt).name}"
        meta = {"model": cfg["model_name"], "pooling": cfg["pooling"], "head": cfg["head"], "ckpt": a.ckpt}
    P = np.zeros((len(samples), N))
    for i, p in enumerate(probs):
        P[i, : len(p)] = p
    res = {"experiment": name, "dataset": a.task, "split": a.split, "n": len(samples), **meta, **extra,
           **all_metrics(P, labels, N), **latency_summary(lat), "wall_seconds": round(wall, 1),
           "throughput_qps": round(len(samples) / wall, 2), "reliability": reliability_table(P, labels)}
    if a.calibrate_on_valid and a.backend == "llm":
        vs = read_jsonl(str(task_path(a.task, "valid")), min(1000, a.limit or 1000))
        vp, _, _, _ = asyncio.run(run_llm(argparse.Namespace(**{**vars(a), "concurrency": 64}), vs))
        VP = np.zeros((len(vs), N))
        for i, p in enumerate(vp):
            VP[i, : len(p)] = p
        ts = TemperatureScaler().fit(VP, np.array([s.label for s in vs]))
        res["temperature"] = ts.T
        res["calibrated"] = all_metrics(ts.transform(P), labels, N)
    (ROOT / "experiments").mkdir(exist_ok=True)
    json.dump(res, open(ROOT / "experiments" / f"{name}.json", "w"), indent=2, ensure_ascii=False)
    np.save(ROOT / "experiments" / f"{name}_probs.npy", P)
    print(json.dumps({k: v for k, v in res.items() if k != "reliability"}, indent=None, ensure_ascii=False))


if __name__ == "__main__":
    main()
