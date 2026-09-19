"""§13 Phase 3 用の実測 timing: 1 token decode と (1+L) token 一括 verification forward の時間。
KV cache あり、batch=1、context ~1024。CUDA event と wall clock を併記 (§17.3)。"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--out", default="/data/openvons/jevpick/bench_verify.json")
    args = ap.parse_args()
    dev = "cuda:0"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(dev).eval()
    ids = torch.randint(1000, 100000, (1, args.ctx), device=dev)
    res = {"model": args.model, "ctx": args.ctx, "iters": args.iters, "steps": {}}
    with torch.no_grad():
        cache = DynamicCache()
        model(input_ids=ids, past_key_values=cache, use_cache=True)
        base_len = cache.get_seq_length()
        for k in [1, 2, 3, 5, 9, 17]:
            new = torch.randint(1000, 100000, (1, k), device=dev)
            pos = torch.arange(base_len, base_len + k, device=dev)[None]
            walls, evs = [], []
            for i in range(10 + args.iters):
                cache.crop(base_len)
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                s.record()
                out = model(input_ids=new, past_key_values=cache, position_ids=pos, use_cache=True)
                _ = out.logits.argmax(-1)
                e.record()
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                if i >= 10:
                    walls.append((t1 - t0) * 1e3)
                    evs.append(s.elapsed_time(e))
            w = np.array(walls)
            res["steps"][k] = {"wall_median_ms": float(np.median(w)), "wall_p90_ms": float(np.percentile(w, 90)),
                               "wall_p95_ms": float(np.percentile(w, 95)), "wall_std_ms": float(w.std()),
                               "cuda_median_ms": float(np.median(evs))}
            print(k, res["steps"][k], flush=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print("->", args.out)


if __name__ == "__main__":
    main()
