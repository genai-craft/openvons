"""Phase 1 v2: source 拡張版 oracle study。

source: ngram (prompt+output) / grammar / corpus (同 domain の train split 出力) /
        schema (Source D, toolcall) / macro (Source E, python) / repo (Source B, python) /
        dflash (Source F: DFlash draft の argmax chain + 先頭 token の top-2..4 差し替え chain) /
        finite (= dflash 以外すべて) / union (= finite + dflash)
出力 pkl: samples[i]["res"][(source, L)] = int16 (N, 4) [m@1, m@4, m@16, n_cands]
         samples[i]["cands"][(source, L)] = 各位置の top-16 候補 (token tuple, source, prior, match) … scorer 学習用
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from openvons.jevpick.candidates.base import Candidate, match_len  # noqa: E402
from openvons.jevpick.candidates.grammar import build_grammar_index  # noqa: E402
from openvons.jevpick.candidates.macro_copy import build_macro_index  # noqa: E402
from openvons.jevpick.candidates.ngram import SuffixIndex, rank  # noqa: E402
from openvons.jevpick.candidates.repository import build_repo_index  # noqa: E402
from openvons.jevpick.candidates.tool_schema import build_schema_index, detect_format, tools_from_prompt_text, user_text_from_prompt_text  # noqa: E402

BLOCKS = (2, 4, 8, 16)
KS = (1, 4, 16)
MAX_N = 6
SEP = 151643
SITE = Path(sys.prefix) / "lib/python3.12/site-packages"
G = {}


def best_match(cands, truth, preranked=False):
    ranked = cands[: KS[-1]] if preranked else rank(cands, KS[-1])
    out, best = [], 0
    for i, c in enumerate(ranked):
        best = max(best, match_len(c.token_ids, truth))
        if i + 1 in KS:
            out.append(best)
    while len(out) < len(KS):
        out.append(best)
    return out, ranked


def dflash_cands(dtok, dlp, t, L):
    """位置 t の DFlash 出力から候補列を作る。chain0 = argmax、chain k = 先頭 token を top-(k+1) に差し替え。"""
    toks, lps = dtok[t], dlp[t]  # (15, 4)
    out = []
    base = [int(x) for x in toks[:, 0]]
    for k in range(4):
        chain = list(base[:L])
        chain[0] = int(toks[0, k])
        # 先頭が同じなら重複
        if k > 0 and chain[0] == base[0]:
            continue
        score = float(lps[0, k]) + float(lps[1:L, 0].sum())
        out.append(Candidate(tuple(chain), "dflash", (score,), metadata={"k": k}))
    return out


def process(sample):
    dom, tok = sample["domain"], G["tok"]
    prompt, out = sample["prompt_ids"], sample["output_ids"]
    P, N = len(prompt), len(out)
    full = prompt + out
    ctx = SuffixIndex(MAX_N)
    ctx.extend(prompt)
    gram = G["grammar"][dom]
    corp = G["corpus"][(dom, sample.get("fold", "all"))]
    ptext = tok.decode(prompt)
    extra = {}
    if dom == "toolcall":
        tools = tools_from_prompt_text(ptext)
        extra["schema"] = build_schema_index(tok, tools, user_text_from_prompt_text(ptext), MAX_N, detect_format(ptext))
    if dom == "python":
        extra["macro"] = build_macro_index(tok, ptext, MAX_N)
        meta = sample.get("meta") or {}
        if meta.get("repo") and meta["repo"] in G["repo"]:
            extra["repo"] = G["repo"][meta["repo"]]
    own_path = (sample.get("meta") or {}).get("path")
    has_df = "dflash" in sample
    has_mtp = "mtp" in sample
    dtok, dlp = (sample["dflash"] if has_df else (None, None))
    mtok, mlp = (sample["mtp"] if has_mtp else (None, None))
    sources = ["ngram", "grammar", "corpus", *extra.keys(), "finite"]
    if has_df:
        sources += ["dflash", "union"]
    if has_mtp:
        sources += ["mtp", "union_mtp"]
    if has_df and has_mtp:
        sources += ["union_all"]
    res = {(s, L): np.zeros((N, 4), dtype=np.int16) for s in sources for L in BLOCKS}
    keep = {(s, L): [] for s in ("finite", "union", "union_mtp", "union_all") for L in BLOCKS if s in sources}
    macro_refresh = 0
    for t in range(N):
        if dom == "python" and t - macro_refresh >= 32:
            extra["macro"] = build_macro_index(tok, tok.decode(full[max(0, P + t - 1500) : P + t]), MAX_N)
            macro_refresh = t
        suffix = full[max(0, P + t - MAX_N) : P + t]
        occ = {"ngram": ctx.occurrences(suffix), "grammar": gram.occurrences(suffix), "corpus": corp.occurrences(suffix)}
        for k, ix in extra.items():
            occ[k] = ix.occurrences_excluding(suffix, own_path) if k == "repo" else ix.occurrences(suffix)
        for L in BLOCKS:
            truth = out[t : t + L]
            cs = {"ngram": ctx.blocks(occ["ngram"], L, "ngram", limit=P + t)}
            cs["grammar"] = gram.blocks(occ["grammar"], L, "grammar")
            cs["corpus"] = corp.blocks(occ["corpus"], L, "corpus")
            for k, ix in extra.items():
                cs[k] = ix.blocks(occ[k], L, k)
            for k in list(cs):  # SEP を含む候補は SEP 手前で切る
                cs[k] = [Candidate(c.token_ids[: c.token_ids.index(SEP)], c.source, c.prior_score, metadata=c.metadata)
                         if SEP in c.token_ids else c for c in cs[k]]
                cs[k] = [c for c in cs[k] if c.token_ids]
            cs["finite"] = [c for k in cs if k != "finite" for c in cs[k]]
            if has_df:
                cs["dflash"] = dflash_cands(dtok, dlp, t, L)
                # union の prior 順: DFlash chain (argmax 優先) を先頭、残り枠を有限候補の prior 順で埋める
                # (prior tuple が source 間で比較不能なため rank() に混ぜない)
                cs["union"] = cs["dflash"] + rank(cs["finite"], KS[-1] - len(cs["dflash"]))
            if has_mtp:
                cs["mtp"] = [Candidate(c.token_ids, "mtp", c.prior_score, metadata=c.metadata) for c in dflash_cands(mtok, mlp, t, L)]
                cs["union_mtp"] = cs["mtp"] + rank(cs["finite"], KS[-1] - len(cs["mtp"]))
            if has_df and has_mtp:
                gen = cs["dflash"] + [c for c in cs["mtp"] if all(c.token_ids != d.token_ids for d in cs["dflash"])]
                cs["union_all"] = gen + rank(cs["finite"], max(KS[-1] - len(gen), 4))
            for s in sources:
                row = res[(s, L)][t]
                cands = cs[s]
                if cands:
                    bm, ranked = best_match(cands, truth, preranked=s.startswith("union"))
                    row[:3] = bm
                    if (s, L) in keep:
                        keep[(s, L)].append([(c.token_ids, c.source, match_len(c.token_ids, truth), tuple(float(v) for v in c.prior_score)) for c in ranked])
                elif (s, L) in keep:
                    keep[(s, L)].append([])
                row[3] = min(len(cands), 32767)
        ctx.extend([out[t]])
    return {"sample_id": sample["sample_id"], "domain": dom, "split": sample["split"], "meta": sample.get("meta"), "N": N, "P": P, "res": res, "cands": keep}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="/data/openvons/jevpick/traces_v2_Qwen3-4B.jsonl")
    ap.add_argument("--prompts", default="/data/openvons/jevpick/prompts_v2.jsonl")
    ap.add_argument("--extract", default="/data/openvons/jevpick/extract_v2_Qwen3-4B", help="extract.py の出力 (dflash 候補)。無ければ dflash 抜き")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--domains", default="toolcall,python")
    ap.add_argument("--corpus-traces", default="", help="corpus index を別の traces (train split) から作る (量子化モデルの test 評価用)")
    args = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    G["tok"] = tok
    prompts = {json.loads(l)["sample_id"]: json.loads(l) for l in open(args.prompts)}
    samples = [json.loads(l) for l in open(args.traces)]
    samples = [s for s in samples if s["domain"] in args.domains.split(",")]
    for s in samples:
        p = prompts[s["sample_id"]]
        s["split"] = p.get("split", "test")
        s["meta"] = p.get("meta")
    for exd in args.extract.split(","):
        ex = Path(exd)
        if not (ex / "index.jsonl").exists() or not (ex / "dflash_tok.npy").exists():
            continue
        dtok = np.load(ex / "dflash_tok.npy", mmap_mode="r")
        dlp = np.load(ex / "dflash_lp.npy", mmap_mode="r")
        idx = {json.loads(l)["sample_id"]: json.loads(l) for l in open(ex / "index.jsonl")}
        for s in samples:
            e = idx.get(s["sample_id"])
            if e:
                s["dflash"] = (np.array(dtok[e["offset"] : e["offset"] + e["N"]]), np.array(dlp[e["offset"] : e["offset"] + e["N"]]))
    print("dflash candidates attached:", sum("dflash" in s for s in samples))
    for exd in args.extract.split(","):
        ex = Path(exd)
        if not (ex / "mtp_tok.npy").exists():
            continue
        mtok = np.load(ex / "mtp_tok.npy", mmap_mode="r")
        mlp = np.load(ex / "mtp_lp.npy", mmap_mode="r")
        idx = {json.loads(l)["sample_id"]: json.loads(l) for l in open(ex / "index.jsonl")}
        for s in samples:
            e = idx.get(s["sample_id"])
            if e:
                s["mtp"] = (np.array(mtok[e["offset"] : e["offset"] + e["N"]]), np.array(mlp[e["offset"] : e["offset"] + e["N"]]))
    print("mtp candidates attached:", sum("mtp" in s for s in samples))
    if args.limit:
        samples = samples[: args.limit]
    domains = sorted({s["domain"] for s in samples})
    G["grammar"] = {d: build_grammar_index(tok, d, MAX_N) for d in domains}
    repos = sorted({(s.get("meta") or {}).get("repo") for s in samples if s["domain"] == "python"} - {None})
    G["repo"] = {}
    for r in repos:
        G["repo"][r] = build_repo_index(tok, str(SITE / r), 150, 30000, MAX_N)
        print("repo index", r, len(G["repo"][r].seq), "tokens", flush=True)
    # corpus: test は train 全体、train は 2-fold で自分と異なる fold から作る (leakage 防止)
    G["corpus"] = {}
    corpus_src = samples
    if args.corpus_traces:
        corpus_src = [json.loads(l) for l in open(args.corpus_traces)]
        for s in corpus_src:
            s["split"] = prompts[s["sample_id"]].get("split", "test")
    for d in domains:
        tr = [s for s in corpus_src if s["domain"] == d and s["split"] == "train"]
        for i, s in enumerate(tr):
            s["fold"] = i % 2
        for f in ("all", 0, 1):
            idx = SuffixIndex(MAX_N, max_positions=128)
            for s in tr:
                if f == "all" or s["fold"] != f:
                    idx.extend(s["output_ids"] + [SEP])
            G["corpus"][(d, f)] = idx
    print("indexes built; samples", len(samples), flush=True)
    with Pool(args.workers) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(process, samples, chunksize=2)):
            results.append(r)
            if (i + 1) % 200 == 0:
                print(i + 1, flush=True)
    tag = Path(args.traces).stem.replace("traces_", "")
    out = Path(args.out or f"/data/openvons/jevpick/oracle_{tag}.pkl")
    pickle.dump({"blocks": BLOCKS, "ks": KS, "samples": results}, open(out, "wb"))
    print("->", out)


if __name__ == "__main__":
    main()
