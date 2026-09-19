"""Oracle Study の集計 (§22 出力表, §11.1, G0 判定)。"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

SOURCES = ("prompt_ngram", "output_ngram", "ngram", "grammar", "corpus", "mixed")


def replay(acc, has, L, cost, skip: bool):
    """cost(k): k token 一括 forward の時間 / 1 token decode の時間。返り値 (tokens, steps, cost)。"""
    n = len(acc)
    t = steps = 0
    c = 0.0
    while t < n:
        steps += 1
        if skip and not has[t]:
            c += 1.0
            t += 1
        else:
            c += cost(L + 1)
            t += int(acc[t]) + 1
    return n, steps, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="/data/openvons/jevpick/oracle_Qwen3-4B-Instruct-2507.pkl")
    ap.add_argument("--bench", default="/data/openvons/jevpick/bench_verify.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    d = pickle.load(open(args.pkl, "rb"))
    blocks, ks = d["blocks"], d["ks"]
    bench = json.load(open(args.bench))
    st = {int(k): v["wall_median_ms"] for k, v in bench["steps"].items()}
    base = st[1]

    def cost_meas(k):
        return st[k] / base

    def cost_ideal(k):
        return 1.0

    domains = sorted({s["domain"] for s in d["samples"]})
    rows = {}
    for dom in domains:
        ss = [s for s in d["samples"] if s["domain"] == dom]
        for src in SOURCES:
            for L in blocks:
                m = np.concatenate([s["res"][(src, L)] for s in ss])  # (T, 4)
                full = np.concatenate([np.arange(s["N"]) + L <= s["N"] for s in ss])
                r = {"n_pos": int(len(m)), "n_full": int(full.sum()), "mean_len": float(np.mean([s["N"] for s in ss]))}
                for i, K in enumerate(ks):
                    r[f"recall@{K}"] = float((m[full, i] == L).mean())
                r["mean_max@16"] = float(m[:, 2].mean())
                r["mean_top1"] = float(m[:, 0].mean())
                r["zero_hit@16"] = float((m[:, 2] == 0).mean())
                r["has_cand"] = float((m[:, 3] > 0).mean())
                r["mean_ncand"] = float(m[:, 3].mean())
                for name, col in (("oracle", 2), ("top1", 0)):
                    for cname, cost in (("ideal", cost_ideal), ("meas", cost_meas)):
                        for skip in (False, True):
                            tot = np.zeros(3)
                            for s in ss:
                                a = s["res"][(src, L)]
                                tot += replay(a[:, col], a[:, 3] > 0, L, cost, skip)
                            key = f"{name}_{cname}{'_skip' if skip else ''}"
                            r[key] = float(tot[0] / tot[2])
                            if cname == "ideal" and not skip:
                                r[f"{name}_tok_per_step"] = float(tot[0] / tot[1])
                rows[(dom, src, L)] = r

    tag = Path(args.pkl).stem.replace("oracle_", "")
    out = Path(args.out or f"experiments/jevpick/phase1_oracle/results_{tag}.md")
    lines = [f"# Oracle Study 結果: {tag}", "",
             f"- trace: {args.pkl}", f"- verify cost (HF eager, ctx={bench['ctx']}, batch=1, 中央値 ms): "
             + ", ".join(f"k={k}: {v:.2f}" for k, v in sorted(st.items())), "",
             "## §22 出力表 (source=mixed, K=16, L=4 / 推定 speedup は L 別 oracle・実測コスト・候補無し時は通常 decode)", "",
             "| Domain | Oracle recall@16 L=4 | Mean max match (L=4) | Zero-hit率 (L=4) | 推定最大 speedup (best L) | prior top-1 speedup (best L) | 平均出力長 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for dom in domains:
        r4 = rows[(dom, "mixed", 4)]
        bestL = max(blocks, key=lambda L: rows[(dom, "mixed", L)]["oracle_meas_skip"])
        bestL1 = max(blocks, key=lambda L: rows[(dom, "mixed", L)]["top1_meas_skip"])
        lines.append(f"| {dom} | {r4['recall@16']*100:.1f}% | {r4['mean_max@16']:.2f} | {r4['zero_hit@16']*100:.1f}% | "
                     f"{rows[(dom,'mixed',bestL)]['oracle_meas_skip']:.2f}x (L={bestL}) | "
                     f"{rows[(dom,'mixed',bestL1)]['top1_meas_skip']:.2f}x (L={bestL1}) | {r4['mean_len']:.0f} |")
    lines += ["", "## G0 判定 (oracle recall@16, block=4 ≥ 50%)", "", "| Domain | " + " | ".join(SOURCES) + " |", "|---|" + "---:|" * len(SOURCES)]
    for dom in domains:
        cells = []
        for src in SOURCES:
            v = rows[(dom, src, 4)]["recall@16"]
            cells.append(f"{'**' if v >= 0.5 else ''}{v*100:.1f}%{'**' if v >= 0.5 else ''}")
        lines.append(f"| {dom} | " + " | ".join(cells) + " |")
    lines += ["", "## domain × source × block_len", "",
              "| Domain | Source | L | recall@1 | recall@4 | recall@16 | mean max@16 | mean top-1 | zero-hit@16 | 候補あり率 | 平均候補数 | tok/step oracle | tok/step top-1 | speedup oracle (理想) | speedup oracle (実測+skip) | speedup top-1 (実測+skip) |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for dom in domains:
        for src in SOURCES:
            for L in blocks:
                r = rows[(dom, src, L)]
                lines.append(f"| {dom} | {src} | {L} | {r['recall@1']*100:.1f}% | {r['recall@4']*100:.1f}% | {r['recall@16']*100:.1f}% | "
                             f"{r['mean_max@16']:.2f} | {r['mean_top1']:.2f} | {r['zero_hit@16']*100:.1f}% | {r['has_cand']*100:.1f}% | {r['mean_ncand']:.1f} | "
                             f"{r['oracle_tok_per_step']:.2f} | {r['top1_tok_per_step']:.2f} | {r['oracle_ideal']:.2f}x | {r['oracle_meas_skip']:.2f}x | {r['top1_meas_skip']:.2f}x |")
    out.write_text("\n".join(lines) + "\n")
    json.dump({f"{k[0]}|{k[1]}|{k[2]}": v for k, v in rows.items()}, open(out.with_suffix(".json"), "w"), indent=1)
    print("\n".join(lines[:20]))
    print("->", out)


if __name__ == "__main__":
    main()
