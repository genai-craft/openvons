"""Phase 1 Oracle Study (§13 Phase 1, §22)。

trace (prompt_ids, greedy output_ids) を replay し、各 decode 位置で
source 別に候補を作り、oracle 最長一致 / prior top-1 一致を記録する。

出力: {DATA}/oracle_<tag>.pkl
  samples: [{sample_id, domain, N, res: {(source, L): int8 array (N, 4) = [m@1, m@4, m@16, n_cands]}}]
"""
from __future__ import annotations

import argparse
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from openvons.jevpick.paths import DATA  # noqa: E402
from openvons.jevpick.candidates.base import match_len  # noqa: E402
from openvons.jevpick.candidates.grammar import build_grammar_index  # noqa: E402
from openvons.jevpick.candidates.ngram import SuffixIndex, rank  # noqa: E402

BLOCKS = (2, 4, 8, 16)
KS = (1, 4, 16)
MAX_N = 6
SEP = 151643  # <|endoftext|>

G = {}  # fork 後に子プロセスへ共有する静的 index


def best_match(cands, truth):
    """prior 順 top-K それぞれの最長一致長。"""
    ranked = rank(cands, KS[-1])
    out, best = [], 0
    for i, c in enumerate(ranked):
        best = max(best, match_len(c.token_ids, truth))
        if i + 1 in KS:
            out.append(best)
    while len(out) < len(KS):
        out.append(best)
    return out


def process(sample):
    dom = sample["domain"]
    prompt, out = sample["prompt_ids"], sample["output_ids"]
    P, N = len(prompt), len(out)
    full = prompt + out
    ctx = SuffixIndex(MAX_N)
    ctx.extend(prompt)
    gram: SuffixIndex = G["grammar"][dom]
    corp: SuffixIndex = G["corpus"][(dom, sample["fold"])]
    sources = ("prompt_ngram", "output_ngram", "ngram", "grammar", "corpus", "mixed")
    res = {(s, L): np.zeros((N, 4), dtype=np.int16) for s in sources for L in BLOCKS}
    for t in range(N):
        suffix = full[max(0, P + t - MAX_N) : P + t]
        occ_ctx = ctx.occurrences(suffix)
        occ_gram = gram.occurrences(suffix)
        occ_corp = corp.occurrences(suffix)
        for L in BLOCKS:
            truth = out[t : t + L]
            c_ctx = ctx.blocks(occ_ctx, L, "ngram", limit=P + t)
            c_prompt = [c for c in c_ctx if c.metadata["p"] <= P]
            c_out = [c for c in c_ctx if c.metadata["p"] > P]
            c_gram = gram.blocks(occ_gram, L, "grammar")
            c_corp = corp.blocks(occ_corp, L, "corpus")
            for s, cands in (("prompt_ngram", c_prompt), ("output_ngram", c_out), ("ngram", c_ctx),
                             ("grammar", c_gram), ("corpus", c_corp), ("mixed", c_ctx + c_gram + c_corp)):
                row = res[(s, L)][t]
                if cands:
                    row[:3] = best_match(cands, truth)
                row[3] = min(len(cands), 32767)
        ctx.extend([out[t]])
    return {"sample_id": sample["sample_id"], "domain": dom, "N": N, "res": res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default=f"{DATA}/traces_Qwen3-4B-Instruct-2507.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    import json
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    samples = [json.loads(l) for l in open(args.traces)]
    if args.limit:
        samples = samples[: args.limit]
    domains = sorted({s["domain"] for s in samples})
    # 2-fold: corpus index は自分と異なる fold の出力から作る (leakage 防止)
    by_dom = {d: [s for s in samples if s["domain"] == d] for d in domains}
    G["grammar"] = {d: build_grammar_index(tok, d, MAX_N) for d in domains}
    G["corpus"] = {}
    for d, ss in by_dom.items():
        for i, s in enumerate(ss):
            s["fold"] = i % 2
        for f in (0, 1):
            idx = SuffixIndex(MAX_N, max_positions=128)
            for s in ss:
                if s["fold"] != f:
                    idx.extend(s["output_ids"] + [SEP])
            G["corpus"][(d, f)] = idx
    print("indexes built; samples", len(samples), flush=True)
    with Pool(args.workers) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(process, samples, chunksize=4)):
            results.append(r)
            if (i + 1) % 100 == 0:
                print(i + 1, flush=True)
    tag = Path(args.traces).stem.replace("traces_", "")
    out = Path(args.out or f"{DATA}/oracle_{tag}.pkl")
    pickle.dump({"blocks": BLOCKS, "ks": KS, "samples": results}, open(out, "wb"))
    print("->", out)


if __name__ == "__main__":
    main()
