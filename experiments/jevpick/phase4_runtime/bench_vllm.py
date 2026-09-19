"""vLLM (0.29) 上での候補 source 比較: 通常 decode / ngram (prompt lookup) / DFlash / suffix を
短文脈 (v2 test, ~300 token) と 16k 文脈で測る。greedy, batch=1, decode 専用 tok/s (max_tokens=1 の TTFT を引く)。
ChoiceSpec の scorer は vLLM 未統合なので、ここでは「有限候補 (ngram) vs 生成型 draft (DFlash)」の
長文脈での挙動を、kernel が最適化された runtime 上で確認する目的。"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
from openvons.jevpick.paths import DATA  # noqa: E402

import argparse
import gc
import json
import time

import numpy as np
import torch


def run_config(name, spec, prompts, max_tokens, model, max_model_len, gpu_util=0.6, tp=1, pp=1, block_size=0):
    from vllm import LLM, SamplingParams
    extra = {"block_size": block_size} if block_size else {}
    llm = LLM(model=model, dtype="bfloat16", gpu_memory_utilization=gpu_util, max_model_len=max_model_len, tensor_parallel_size=tp, pipeline_parallel_size=pp, **extra,
              speculative_config=spec, enforce_eager=False, max_num_seqs=1, enable_prefix_caching=False,
              disable_log_stats=False)
    sp1 = SamplingParams(temperature=0, max_tokens=1)
    spN = SamplingParams(temperature=0, max_tokens=max_tokens)
    out = {}
    for dom, rows in prompts.items():
        recs = []
        for i, r in enumerate(rows):
            ids = r["prompt_ids"]
            # warm-up 1 回目は捨てる
            t0 = time.perf_counter(); llm.generate({"prompt_token_ids": ids}, sp1, use_tqdm=False); t1 = time.perf_counter() - t0
            t0 = time.perf_counter(); o = llm.generate({"prompt_token_ids": ids}, spN, use_tqdm=False); tN = time.perf_counter() - t0
            n = len(o[0].outputs[0].token_ids)
            if i > 0 and n > 1:
                recs.append({"n": n, "ttft": t1, "total": tN, "decode": tN - t1, "text": o[0].outputs[0].text[:200], "ids": list(o[0].outputs[0].token_ids)})
        tok = sum(r["n"] - 1 for r in recs); dec = sum(r["decode"] for r in recs)
        out[dom] = {"n": len(recs), "decode_tok_s": tok / dec, "ttft_mean": float(np.mean([r["ttft"] for r in recs])),
                    "e2e_tok_s": sum(r["n"] for r in recs) / sum(r["total"] for r in recs), "outputs": [r["ids"] for r in recs]}
        print(name, dom, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in out[dom].items() if k != "outputs"}, flush=True)
    # spec decode metrics (accepted tokens / drafts)
    try:
        ms = {m.name: m for m in llm.get_metrics()}
        keys = [k for k in ms if "spec_decode" in k]
        out["metrics"] = {k: getattr(ms[k], "value", None) if hasattr(ms[k], "value") else str(ms[k])[:200] for k in keys}
        print(name, "metrics", out["metrics"], flush=True)
    except Exception as e:  # noqa: BLE001
        out["metrics"] = {"error": str(e)[:200]}
    del llm
    gc.collect(); torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--draft", default="z-lab/Qwen3-4B-DFlash-b16")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--configs", default="none,ngram,dflash")
    ap.add_argument("--out", default="experiments/jevpick/phase4_runtime/vllm_Qwen3-4B.json")
    ap.add_argument("--k-dflash", type=int, default=15)
    ap.add_argument("--domains", default="toolcall,python")
    ap.add_argument("--no-long", action="store_true")
    ap.add_argument("--gpu-util", type=float, default=0.6)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--pp", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=0)
    ap.add_argument("--block-size", type=int, default=0)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    def load(path, split_filter=True):
        rows = [json.loads(l) for l in open(path)]
        by = {}
        for r in rows:
            if split_filter and r.get("split") != "test":
                continue
            if r["domain"] not in args.domains.split(","):
                continue
            if len(by.setdefault(r["domain"], [])) >= args.n + 1:
                continue
            kw = {"tools": r["tools"]} if r["tools"] else {}
            text = tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False, **kw)
            by[r["domain"]].append({"sample_id": r["sample_id"], "prompt_ids": tok.encode(text, add_special_tokens=False)})
        return by

    short = load(f"{DATA}/prompts_v2.jsonl")
    long_ = {} if args.no_long else load(f"{DATA}/prompts_long.jsonl")
    prompts = {f"{d}_short": v for d, v in short.items()} | {f"{d}_16k": v for d, v in long_.items()}
    for d, v in prompts.items():
        print(d, len(v), "mean prompt tokens", int(np.mean([len(x["prompt_ids"]) for x in v])), flush=True)
    specs = {
        "none": None,
        "ngram": {"method": "ngram", "num_speculative_tokens": 8, "prompt_lookup_max": 6, "prompt_lookup_min": 1},
        "ngram16": {"method": "ngram", "num_speculative_tokens": 16, "prompt_lookup_max": 6, "prompt_lookup_min": 1},
        "dflash": {"method": "dflash", "model": args.draft, "num_speculative_tokens": args.k_dflash},
        "dflash8": {"method": "dflash", "model": args.draft, "num_speculative_tokens": 7},
        "suffix": {"method": "suffix", "num_speculative_tokens": 8},
        "mtp": {"method": "mtp", "num_speculative_tokens": 3},
        "mtp1": {"method": "mtp", "num_speculative_tokens": 1},
        "mtp7": {"method": "mtp", "num_speculative_tokens": 7},
        "mtp15": {"method": "mtp", "num_speculative_tokens": 15},
    }
    res = {}
    for name in args.configs.split(","):
        try:
            res[name] = run_config(name, specs[name], prompts, args.max_tokens, args.model, args.max_model_len or (20000 if not args.no_long else 4096), args.gpu_util, args.tp, args.pp, args.block_size)
        except Exception as e:  # noqa: BLE001
            import traceback; traceback.print_exc()
            res[name] = {"error": str(e)[:500]}
        json.dump(res, open(args.out, "w"), indent=1)
    # 出力一致 (none との greedy 一致率)
    if "none" in res:
        for name, r in res.items():
            if name == "none" or "error" in r:
                continue
            for d in prompts:
                if d in r and d in res["none"]:
                    a, b = r[d]["outputs"], res["none"][d]["outputs"]
                    r[d]["exact_match_vs_none"] = float(np.mean([x == y for x, y in zip(a, b)]))
        json.dump(res, open(args.out, "w"), indent=1)
    print("->", args.out)


if __name__ == "__main__":
    main()
