"""§12 baseline 8: DFlash (公式実装 dflash_generate) の greedy 受理長と tok/s を test split で実測し、
同条件の通常 decode (自前 verifier, L=0) と比較する。出力一致も確認。"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from dflash.model import DFlashDraftModel, dflash_generate
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from openvons.jevpick.runtime.verifier import GreedyVerifier  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--draft", default="z-lab/Qwen3-4B-DFlash-b16")
    ap.add_argument("--prompts", default="/data/openvons/jevpick/prompts_v2.jsonl")
    ap.add_argument("--per-domain", type=int, default=30)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--blocks", default="16,8")
    ap.add_argument("--out", default="experiments/jevpick/phase4_runtime/dflash_Qwen3-4B.json")
    args = ap.parse_args()
    dev = "cuda:0"
    tok = AutoTokenizer.from_pretrained(args.model)
    target = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
    draft = DFlashDraftModel.from_pretrained(args.draft, attn_implementation="sdpa", dtype=torch.bfloat16).to(dev).eval()
    stop = [tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")]
    rows = [json.loads(l) for l in open(args.prompts)]
    test = {}
    for r in rows:
        if r.get("split") == "test" and len(test.setdefault(r["domain"], [])) < args.per_domain:
            test[r["domain"]].append(r)
    ver = GreedyVerifier(target, dev)
    results = {}
    for dom, rs in test.items():
        base_t = base_n = 0.0
        stats = {b: {"t": 0.0, "n": 0, "acc": [], "match": 0} for b in map(int, args.blocks.split(","))}
        for i, r in enumerate(rs):
            kw = {"tools": r["tools"]} if r["tools"] else {}
            text = tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False, **kw)
            ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
            # baseline: 自前 verifier で通常 decode (DFlash と同じ HF eager 条件)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            x = ver.prefill(ids[0].tolist())
            out = []
            while len(out) < args.max_new:
                new, _ = ver.step(x, ())
                out.append(new[0]); x = new[0]
                if x in stop:
                    break
            torch.cuda.synchronize(); dt = time.perf_counter() - t0
            if i > 0:  # warm-up 1 本除外
                base_t += dt; base_n += len(out)
            for b, st in stats.items():
                torch.cuda.synchronize(); t0 = time.perf_counter()
                res = dflash_generate(draft, target, ids, args.max_new, stop, block_size=b, return_stats=True)
                torch.cuda.synchronize(); dt = time.perf_counter() - t0
                gen = res.output_ids[0, ids.shape[1]:].tolist()
                if i > 0:
                    st["t"] += dt; st["n"] += len(gen); st["acc"] += res.acceptance_lengths
                st["match"] += int(gen[: len(out)] == out[: len(gen)])
        summ = {"n": len(rs) - 1, "baseline_tok_s": base_n / base_t}
        for b, st in stats.items():
            acc = np.array(st["acc"])
            summ[f"b{b}"] = {"tok_s": st["n"] / st["t"], "speedup": (st["n"] / st["t"]) / summ["baseline_tok_s"],
                             "mean_produced_per_step": float(acc.mean()), "mean_accepted": float(acc.mean() - 1),
                             "zero_accept_rate": float((acc == 1).mean()), "exact_match": st["match"] / len(rs)}
        results[dom] = summ
        print(dom, json.dumps(summ, indent=1), flush=True)
    json.dump(results, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
