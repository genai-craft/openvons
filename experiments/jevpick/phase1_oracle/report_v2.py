"""Oracle Study v2 の集計。domain × (split group) × source × L。"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
from openvons.jevpick.paths import DATA  # noqa: E402

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def replay(acc, has, L, cost):
    n = len(acc); t = 0; c = 0.0
    while t < n:
        if has[t]:
            c += cost; t += int(acc[t]) + 1
        else:
            c += 1.0; t += 1
    return n, c


def group_of(s):
    if s["domain"] == "toolcall":
        return f"toolcall/test/{'known' if (s.get('meta') or {}).get('schema_known') else 'unknown'}" if s["split"] == "test" else "toolcall/train"
    return f"python/{s['split']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default=f"{DATA}/oracle_v2_Qwen3-4B.pkl")
    ap.add_argument("--bench", default=f"{DATA}/bench_verify.json")
    ap.add_argument("--out", default="experiments/jevpick/phase1_oracle/results_v2_Qwen3-4B.md")
    args = ap.parse_args()
    d = pickle.load(open(args.pkl, "rb"))
    blocks, ks = d["blocks"], d["ks"]
    st = {int(k): v["wall_median_ms"] for k, v in json.load(open(args.bench))["steps"].items()}

    def cost(L):
        k = min(st, key=lambda k: abs(k - (L + 1)))
        return st[k] / st[1]

    groups = sorted({group_of(s) for s in d["samples"]})
    sources = []
    for s in d["samples"]:
        for k in s["res"]:
            if k[0] not in sources:
                sources.append(k[0])
    rows = {}
    for g in groups:
        ss = [s for s in d["samples"] if group_of(s) == g]
        for src in sources:
            for L in blocks:
                sel = [s for s in ss if (src, L) in s["res"]]
                if not sel:
                    continue
                m = np.concatenate([s["res"][(src, L)] for s in sel])
                full = np.concatenate([np.arange(s["N"]) + L <= s["N"] for s in sel])
                r = {"n_samples": len(sel), "n_pos": int(len(m))}
                for i, K in enumerate(ks):
                    r[f"recall@{K}"] = float((m[full, i] == L).mean()) if full.any() else 0.0
                r["mean_max@16"] = float(m[:, 2].mean()); r["mean_top1"] = float(m[:, 0].mean())
                r["zero_hit@16"] = float((m[:, 2] == 0).mean()); r["has_cand"] = float((m[:, 3] > 0).mean())
                r["mean_ncand"] = float(m[:, 3].mean())
                for name, col in (("oracle", 2), ("top1", 0)):
                    n = c = 0.0
                    for s in sel:
                        a = s["res"][(src, L)]
                        nn_, cc = replay(a[:, col], a[:, 3] > 0, L, cost(L))
                        n += nn_; c += cc
                    r[f"speedup_{name}"] = n / c
                rows[(g, src, L)] = r
    lines = [f"# Oracle Study v2: {Path(args.pkl).stem}", "",
             "推定 speedup = replay (候補なし位置は通常 decode) × 実測 forward コスト比 (HF eager, ctx=1024)。", ""]
    lines += ["## G0 表 (L=4, oracle recall@16) と mean max match", "",
              "| Group | " + " | ".join(sources) + " |", "|---|" + "---:|" * len(sources)]
    for g in groups:
        cells = []
        for src in sources:
            r = rows.get((g, src, 4))
            cells.append(f"{r['recall@16']*100:.1f}% / {r['mean_max@16']:.2f}" if r else "-")
        lines.append(f"| {g} | " + " | ".join(cells) + " |")
    lines += ["", "## 推定 speedup (oracle / prior top-1), source × L", "",
              "| Group | Source | " + " | ".join(f"L={L}" for L in blocks) + " |", "|---|---|" + "---:|" * len(blocks)]
    for g in groups:
        for src in sources:
            cells = []
            for L in blocks:
                r = rows.get((g, src, L))
                cells.append(f"{r['speedup_oracle']:.2f}x / {r['speedup_top1']:.2f}x" if r else "-")
            if any(c != "-" for c in cells):
                lines.append(f"| {g} | {src} | " + " | ".join(cells) + " |")
    lines += ["", "## 詳細", "", "| Group | Source | L | n_pos | recall@1 | recall@4 | recall@16 | mean max@16 | mean top-1 | zero-hit@16 | 候補あり率 | 平均候補数 | speedup oracle | speedup top-1 |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for (g, src, L), r in rows.items():
        lines.append(f"| {g} | {src} | {L} | {r['n_pos']} | {r['recall@1']*100:.1f}% | {r['recall@4']*100:.1f}% | {r['recall@16']*100:.1f}% | {r['mean_max@16']:.2f} | {r['mean_top1']:.2f} | {r['zero_hit@16']*100:.1f}% | {r['has_cand']*100:.1f}% | {r['mean_ncand']:.1f} | {r['speedup_oracle']:.2f}x | {r['speedup_top1']:.2f}x |")
    Path(args.out).write_text("\n".join(lines) + "\n")
    json.dump({f"{k[0]}|{k[1]}|{k[2]}": v for k, v in rows.items()}, open(Path(args.out).with_suffix(".json"), "w"), indent=1)
    print("\n".join(lines[:12 + len(groups)]))
    print("->", args.out)


if __name__ == "__main__":
    main()
