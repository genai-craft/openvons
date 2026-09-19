"""Phase 4 実装順 1〜2: verifier + candidate generator (prior top-1) で、通常 decode と全 token 一致 (§17.2) を確認し、
実測 wall-clock speedup と受理長を記録する。scorer なし (= §12 baseline 3: prior 順選択)。"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from openvons.jevpick.candidates.grammar import build_grammar_index  # noqa: E402
from openvons.jevpick.candidates.ngram import SuffixIndex, rank  # noqa: E402
from openvons.jevpick.runtime.verifier import GreedyVerifier  # noqa: E402

MAX_N = 6
SEP = 151643


def run(verifier, prompt_ids, max_new, stop_ids, block_len, indexes, timing):
    """block_len=0 で通常 decode。indexes: 候補 source の SuffixIndex 群 (ctx は内部で作る)。"""
    ctx = SuffixIndex(MAX_N)
    ctx.extend(prompt_ids[:-1])
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    x = verifier.prefill(prompt_ids)
    out: list[int] = []
    steps = 0
    accepted = []
    t_cand = 0.0
    while len(out) < max_new:
        draft = ()
        if block_len:
            tc = time.perf_counter()
            full = prompt_ids + out
            suffix = full[-MAX_N:]
            cands = []
            ctx_occ = ctx.occurrences(suffix)
            cands += ctx.blocks(ctx_occ, block_len, "ngram")
            for name, idx in indexes.items():
                cands += idx.blocks(idx.occurrences(suffix), block_len, name)
            top = rank(cands, 1)
            if top:
                draft = tuple(t for t in top[0].token_ids if t != SEP)
            t_cand += time.perf_counter() - tc
        new, a = verifier.step(x, draft)
        ctx.extend([x])
        steps += 1
        accepted.append(a)
        for t in new:
            out.append(t)
            if t in stop_ids or len(out) >= max_new:
                break
        if out[-1] in stop_ids or len(out) >= max_new:
            break
        # ctx には確定 token を x 以外すべて入れる (x は次 step で入る)
        ctx.extend(new[:-1])
        x = new[-1]
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    timing.append({"tokens": len(out), "steps": steps, "sec": dt, "cand_sec": t_cand, "accepted": accepted})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--traces", default="/data/openvons/jevpick/traces_Qwen3-4B-Instruct-2507.jsonl")
    ap.add_argument("--per-domain", type=int, default=20)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--blocks", default="4,8")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", default="experiments/jevpick/phase4_runtime/golden_Qwen3-4B-Instruct-2507.json")
    args = ap.parse_args()
    dev = "cuda:0"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype)).to(dev).eval()
    stop_ids = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
    traces = [json.loads(l) for l in open(args.traces)]
    domains = sorted({t["domain"] for t in traces})
    test, corpus = {}, {}
    for d in domains:
        ss = [t for t in traces if t["domain"] == d]
        test[d] = ss[: args.per_domain]
        idx = SuffixIndex(MAX_N, max_positions=128)
        for s in ss[args.per_domain :]:
            idx.extend(s["output_ids"] + [SEP])
        corpus[d] = {"corpus": idx, "grammar": build_grammar_index(tok, d, MAX_N)}
    ver = GreedyVerifier(model, dev)
    # warm-up
    run(ver, test[domains[0]][0]["prompt_ids"], 16, stop_ids, 0, {}, [])
    results = {}
    blocks = [int(b) for b in args.blocks.split(",")]
    for d in domains:
        res = {"baseline": [], **{f"L{L}": [] for L in blocks}}
        match = {f"L{L}": 0 for L in blocks}
        for s in test[d]:
            base = run(ver, s["prompt_ids"], args.max_new, stop_ids, 0, {}, res["baseline"])
            for L in blocks:
                spec = run(ver, s["prompt_ids"], args.max_new, stop_ids, L, corpus[d], res[f"L{L}"])
                match[f"L{L}"] += int(spec == base)
        summary = {"n": len(test[d])}
        bt = sum(r["sec"] for r in res["baseline"])
        btok = sum(r["tokens"] for r in res["baseline"])
        summary["baseline_tok_s"] = btok / bt
        for L in blocks:
            rr = res[f"L{L}"]
            st = sum(r["sec"] for r in rr)
            stok = sum(r["tokens"] for r in rr)
            acc = np.concatenate([r["accepted"] for r in rr])
            summary[f"L{L}"] = {
                "exact_match": match[f"L{L}"] / len(test[d]),
                "tok_s": stok / st, "speedup": (stok / st) / summary["baseline_tok_s"],
                "tok_per_step": stok / sum(r["steps"] for r in rr),
                "mean_accepted": float(acc.mean()), "zero_accept_rate": float((acc == 0).mean()),
                "cand_overhead_frac": sum(r["cand_sec"] for r in rr) / st,
            }
        results[d] = summary
        print(d, json.dumps(summary, indent=1), flush=True)
    json.dump(results, open(args.out, "w"), indent=1)
    print("->", args.out)


if __name__ == "__main__":
    main()
