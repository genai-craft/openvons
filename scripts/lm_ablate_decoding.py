"""「max_tokens=1 にするだけ」と「guided choice + logprobs」で何が違うのかを実測する。

4 条件:
  A plain_1tok   : max_tokens=1 のみ (制約なし)            -> 出たトークンがラベルとは限らない
  B plain_8tok   : max_tokens=8 のみ (制約なし)            -> 文章の書き出しが返る
  C guided       : guided choice + max_tokens=1            -> 必ず妥当なラベル。ただし答えだけ
  D guided+logp  : C + logprobs                            -> 選択肢すべての確率が 1 forward で得られる
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

from openvons.core.metrics import all_metrics
from openvons.core.formats import llm_single_prompt, SYSTEM_PROMPT
from openvons.lm.backends.llm_backend import LLMBackend, _softmax_from_logprobs
from openvons.core.formats import read_jsonl
from openvons.lm.training.dataset import task_path

ROOT = Path(__file__).resolve().parents[1]
SLOPPY = False


async def run(be: LLMBackend, samples, cond: str):
    out = [None] * len(samples)
    lat = [0.0] * len(samples)
    pbar = tqdm(total=len(samples), desc=cond, mininterval=2)

    async def one(i, s):
        prompt, labels = llm_single_prompt(s.state, s.question)
        sys_msg = SYSTEM_PROMPT
        if SLOPPY:      # 「ラベルだけで答えろ」という指示を外した、素朴なプロンプト
            prompt = prompt.replace("\n\nAnswer with the option label only.", "")
            sys_msg = "You are a helpful assistant."
        body = {"messages": [{"role": "system", "content": sys_msg}, {"role": "user", "content": prompt}],
                "temperature": 0}
        if cond == "plain_1tok":
            body["max_tokens"] = 1
        elif cond == "plain_8tok":
            body["max_tokens"] = 8
        elif cond == "plain_4tok":
            body["max_tokens"] = 4
        else:
            body["max_tokens"] = 1 if max(len(l) for l in labels) == 1 else 4
            body["structured_outputs"] = {"choice": labels}
            if cond == "guided_logprob":
                body.update(logprobs=True, top_logprobs=min(max(len(labels), 5), 100))
        r = await be._chat(body)
        ch = r["choices"][0]
        txt = ch["message"]["content"]
        rec = {"text": txt, "usage": r.get("usage", {}), "labels": labels}
        if cond == "guided_logprob" and ch.get("logprobs", {}).get("content"):
            lp = {l: -np.inf for l in labels}
            for t in ch["logprobs"]["content"][0]["top_logprobs"]:
                tk = t["token"].strip()
                if tk in lp and t["logprob"] > lp[tk] and t["logprob"] > -9000:
                    lp[tk] = t["logprob"]
            rec["probs"] = _softmax_from_logprobs([lp[l] for l in labels])
        out[i] = rec
        lat[i] = r["_latency_ms"]
        pbar.update(1)

    await asyncio.gather(*[one(i, s) for i, s in enumerate(samples)])
    pbar.close()
    return out, lat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="massive_scenario_en")
    ap.add_argument("--port", type=int, default=8300)
    ap.add_argument("--model", default="qwen3-4b")
    ap.add_argument("--limit", type=int, default=600)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--sloppy", action="store_true")
    a = ap.parse_args()
    global SLOPPY
    SLOPPY = a.sloppy
    samples = read_jsonl(str(task_path(a.task, "test")), a.limit)
    y = np.array([s.label for s in samples])
    N = max(s.question.n for s in samples)
    results = {}

    async def all_conditions():
        be = LLMBackend(f"http://127.0.0.1:{a.port}/v1", a.model, concurrency=a.concurrency)
        return [(c, await run(be, samples, c)) for c in ["plain_1tok", "plain_4tok", "plain_8tok", "guided", "guided_logprob"]]

    for cond, (recs, lat) in asyncio.run(all_conditions()):
        valid = [i for i, r in enumerate(recs) if r["text"].strip().rstrip(".") in r["labels"]]
        pred = np.array([r["labels"].index(r["text"].strip().rstrip(".")) if i in set(valid) else -1
                         for i, r in enumerate(recs)])
        row = {"parse_ok_rate": len(valid) / len(recs),
               "accuracy_counting_failures_as_wrong": float((pred == y).mean()),
               "accuracy_on_parsable": float((pred[pred >= 0] == y[pred >= 0]).mean()) if len(valid) else None,
               "completion_tokens_mean": float(np.mean([r["usage"].get("completion_tokens", 0) for r in recs])),
               "latency_p50_ms": float(np.percentile(lat, 50)),
               "gives_probabilities": cond == "guided_logprob",
               "sample_outputs": [r["text"] for r in recs[:5]]}
        if cond == "guided_logprob" and all(r.get("probs") for r in recs):
            P = np.zeros((len(recs), N))
            for i, r in enumerate(recs):
                P[i, : len(r["probs"])] = r["probs"]
            row["metrics_from_probs"] = all_metrics(P, y, N)
        results[cond] = row
        print(f"\n[{cond}] parse_ok={row['parse_ok_rate']:.3f} acc={row['accuracy_counting_failures_as_wrong']:.4f} "
              f"out_tokens={row['completion_tokens_mean']:.1f} p50={row['latency_p50_ms']:.1f}ms")
        print("  例:", [t[:30] for t in row["sample_outputs"][:3]])
    json.dump({"experiment": f"exp016_decoding_ablation_{a.model}_{a.task}" + ("_sloppy" if a.sloppy else ""), "task": a.task, "n": len(samples),
               "model": a.model, "conditions": results},
              open(ROOT / "experiments" / f"exp016_decoding_ablation_{a.model}.json", "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
