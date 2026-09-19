"""文脈長別の step コスト実測 (§13 Phase 6 「context 4K/16K/64K」)。
target 1-token decode / target verify (k=9, 17) / DFlash draft forward (block 16) / 有限候補生成 (CPU, SuffixIndex)
を ctx ∈ {1k, 4k, 16k, 32k} で測る。batch=1, HF eager sdpa, warm-up 5 + 30 回, 中央値 ms。"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from dflash.model import DFlashDraftModel, extract_context_feature
from transformers import AutoModelForCausalLM, DynamicCache

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from openvons.jevpick.candidates.ngram import SuffixIndex  # noqa: E402


def timed(fn, iters=30, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--draft", default="z-lab/Qwen3-4B-DFlash-b16")
    ap.add_argument("--ctxs", default="1024,4096,16384,32768")
    ap.add_argument("--out", default="/data/openvons/jevpick/bench_ctx.json")
    args = ap.parse_args()
    dev = "cuda:0"
    target = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
    draft = DFlashDraftModel.from_pretrained(args.draft, attn_implementation="sdpa", dtype=torch.bfloat16).to(dev).eval()
    emb = target.get_input_embeddings().weight
    B = draft.block_size
    res = {}
    with torch.inference_mode():
        for ctx in map(int, args.ctxs.split(",")):
            ids = torch.randint(1000, 100000, (1, ctx), device=dev)
            cache = DynamicCache()
            o = target(input_ids=ids, past_key_values=cache, use_cache=True, output_hidden_states=True)
            th = extract_context_feature(o.hidden_states, draft.target_layer_ids)
            base = cache.get_seq_length()
            r = {}
            for k in (1, 9, 17):
                new = torch.randint(1000, 100000, (1, k), device=dev)
                pos = torch.arange(base, base + k, device=dev)[None]

                def f():
                    cache.crop(base)
                    out = target(input_ids=new, past_key_values=cache, position_ids=pos, use_cache=True, output_hidden_states=(k > 1))
                    _ = out.logits.argmax(-1)
                r[f"target_k{k}_ms"] = timed(f)
            # DFlash draft: context KV を一度作ってから、block だけの forward (実運用の毎 step コスト)
            dcache = DynamicCache(config=draft.config)
            block = torch.full((1, B), draft.mask_token_id, dtype=torch.long, device=dev)
            block[0, 0] = 1000
            pos_all = torch.arange(0, ctx + B, device=dev)[None]
            draft(target_hidden=th, noise_embedding=F.embedding(block, emb), position_ids=pos_all[:, : ctx + B], past_key_values=dcache, use_cache=True)
            dcache.crop(-B)
            dlen = dcache.get_seq_length()

            def g():
                dcache.crop(dlen)
                dh = draft(target_hidden=th[:, ctx - 1 : ctx - 1], noise_embedding=F.embedding(block, emb),
                           position_ids=pos_all[:, ctx - 1 : ctx - 1 + B], past_key_values=dcache, use_cache=True)[:, 1:, :]
                _ = draft.compute_logits(dh, target.lm_head).argmax(-1)
            r["dflash_draft_ms"] = timed(g)
            # 有限候補生成 (CPU): ctx token の SuffixIndex を作り、suffix lookup + blocks(L=8)
            toks = ids[0].tolist()
            idx = SuffixIndex(6)
            t0 = time.perf_counter(); idx.extend(toks); r["ngram_index_build_ms"] = (time.perf_counter() - t0) * 1e3
            # 実データに近づけるため、suffix が必ず当たるように文脈中の n-gram を使う
            rng = np.random.default_rng(0)
            ts = []
            for _ in range(50):
                p = int(rng.integers(6, ctx - 8))
                suffix = toks[p - 6 : p]
                t0 = time.perf_counter()
                occ = idx.occurrences(suffix)
                _ = idx.blocks(occ, 8, "ngram")
                ts.append((time.perf_counter() - t0) * 1e3)
            r["ngram_lookup_ms"] = float(np.median(ts))
            res[ctx] = r
            print(ctx, {k: round(v, 3) for k, v in r.items()}, flush=True)
            del cache, dcache, o, th
            torch.cuda.empty_cache()
    json.dump(res, open(args.out, "w"), indent=1)
    print("->", args.out)


if __name__ == "__main__":
    main()
