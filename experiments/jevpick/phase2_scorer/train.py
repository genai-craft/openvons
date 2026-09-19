"""Phase 2: offline scorer 学習と評価 (§13 Phase 2, G1)。

データ: oracle pkl (候補 + 実受理長) と extract dir (hidden)。
評価: prior top-1 / scorer top-1 / oracle の平均受理長・top-1 正解率、replay 推定 speedup (実測コスト, controller skip)。
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from openvons.jevpick.paths import DATA  # noqa: E402
from openvons.jevpick.scorer.model import SOURCES, BlockScorer, scorer_loss  # noqa: E402

SRC_ID = {s: i + 1 for i, s in enumerate(SOURCES)}


def load_embeddings(model_id: str) -> torch.Tensor:
    from huggingface_hub import snapshot_download
    d = Path(snapshot_download(model_id, allow_patterns=["*.json", "*.safetensors"]))
    idx = json.load(open(d / "model.safetensors.index.json"))["weight_map"]
    key = next(k for k in idx if k.endswith("embed_tokens.weight") and not k.startswith("mtp"))
    with safe_open(d / idx[key], "pt") as sf:
        return sf.get_tensor(key)


def prior_feats(src, prior):
    if src in ("dflash", "mtp"):
        return [0.0, 0.0, float(prior[0]) / 10.0]
    n = float(prior[0]) if prior else 0.0
    c = float(prior[1]) if len(prior) > 1 else 0.0
    return [n / 6.0, np.log1p(c), 0.0]


class Data:
    """1 decode 位置 = 1 example。候補 ≤ K、token ≤ L。"""

    def __init__(self, samples, ex_dirs, source_key, L, K, layer_i, hidden_all, index_all, max_pos=0):
        self.L, self.K = L, K
        rows = []  # (hidden_offset, anchor_token, cands)
        for s in samples:
            e = index_all.get(s["sample_id"])
            if e is None:
                continue
            cl = s["cands"].get((source_key, L))
            if cl is None:
                continue
            for t, cands in enumerate(cl):
                if not cands:
                    continue
                rows.append((e["ex"], e["offset"] + t, s["anchor"][t], cands[:K], s["domain"], s.get("meta") or {}, s["sample_id"], t))
        if max_pos and len(rows) > max_pos:
            rng = np.random.default_rng(0)
            rows = [rows[i] for i in sorted(rng.choice(len(rows), max_pos, replace=False))]
        self.rows = rows
        self.hidden_all = hidden_all
        self.layer_i = layer_i

    def __len__(self):
        return len(self.rows)

    def batch(self, idxs, emb, dev):
        B, K, L = len(idxs), self.K, self.L
        h = np.stack([self.hidden_all[self.rows[i][0]][self.rows[i][1], self.layer_i] for i in idxs]).astype(np.float32)
        x = torch.tensor([self.rows[i][2] for i in idxs], device=dev)
        tok = torch.zeros(B, K, L, dtype=torch.long)
        cm = torch.zeros(B, K, L, dtype=torch.bool)
        km = torch.zeros(B, K, dtype=torch.bool)
        src = torch.zeros(B, K, dtype=torch.long)
        pr = torch.zeros(B, K, 3)
        match = torch.zeros(B, K, dtype=torch.long)
        for b, i in enumerate(idxs):
            for c, (ids, s, m, prior) in enumerate(self.rows[i][3]):
                n = min(len(ids), L)
                tok[b, c, :n] = torch.tensor(ids[:n])
                cm[b, c, :n] = True
                km[b, c] = True
                src[b, c] = SRC_ID.get(s, 0)
                pr[b, c] = torch.tensor(prior_feats(s, prior))
                match[b, c] = min(m, n)
        tok, cm, km, src, pr, match = (t.to(dev) for t in (tok, cm, km, src, pr, match))
        return (torch.tensor(h, device=dev), emb[x], emb[tok], cm, km, src, pr, match)


def evaluate(model, data, emb, dev, bs=256):
    model.eval()
    out = {"prior": [], "scorer": [], "oracle": [], "exp": [], "n": 0, "pick_src": [], "guard0.5": [], "guard1.0": [], "guard2.0": []}
    rows_meta = []
    with torch.no_grad():
        for s in range(0, len(data), bs):
            idxs = list(range(s, min(s + bs, len(data))))
            h, xe, ce, cm, km, src, pr, match = data.batch(idxs, emb, dev)
            logits = model(h, xe, ce, cm, src, pr)
            e = BlockScorer.expected_len(logits, cm).masked_fill(~km, -1e4)
            pick = e.argmax(-1)
            out["scorer"].append(match.gather(1, pick[:, None])[:, 0].cpu().numpy())
            # chain0 = 候補 0 が dflash なら (union は dflash を先頭に並べている)
            is_df0 = (src[:, 0] == SRC_ID["dflash"]) & km[:, 0]
            e0 = e[:, 0]
            for mname, mg in (("guard0.5", 0.5), ("guard1.0", 1.0), ("guard2.0", 2.0)):
                use0 = is_df0 & (e.max(-1).values - e0 < mg)
                gp = torch.where(use0, torch.zeros_like(pick), pick)
                out[mname].append(match.gather(1, gp[:, None])[:, 0].cpu().numpy())
            out["pick_src"].append(src.gather(1, pick[:, None])[:, 0].cpu().numpy())
            out["prior"].append(match[:, 0].cpu().numpy())  # 候補は prior 順で保存されている
            out["oracle"].append(match.max(-1).values.cpu().numpy())
            out["exp"].append(e.max(-1).values.cpu().numpy())
            rows_meta += [(data.rows[i][4], data.rows[i][5], data.rows[i][6], data.rows[i][7]) for i in idxs]
    for k in ("prior", "scorer", "oracle", "exp", "pick_src", "guard0.5", "guard1.0", "guard2.0"):
        out[k] = np.concatenate(out[k])
    out["meta"] = rows_meta
    return out


def summarize(ev, L, cost_ratio, tag=""):
    """位置別 (候補あり位置のみ) の指標と、sample 単位 replay speedup。"""
    r = {}
    oracle = ev["oracle"]
    for k in ("prior", "scorer", "oracle", "guard0.5", "guard1.0", "guard2.0"):
        if k not in ev:
            continue
        m = ev[k]
        r[f"{k}_mean_acc"] = float(m.mean())
        r[f"{k}_top1_correct"] = float((m == oracle).mean())
        r[f"{k}_zero"] = float((m == 0).mean())
    if "pick_src" in ev:
        inv = {v: k for k, v in SRC_ID.items()}
        vals, cnts = np.unique(ev["pick_src"], return_counts=True)
        r["pick_src"] = {inv.get(int(v), "?"): float(c / len(ev["pick_src"])) for v, c in zip(vals, cnts)}
    r["exp_mae"] = float(np.abs(ev["exp"] - ev["scorer"]).mean())
    # replay: sample ごとに位置順に辿る。候補なし位置は通常 decode。controller: exp < th なら通常 decode
    by_sample = {}
    for i, (dom, meta, sid, t) in enumerate(ev["meta"]):
        by_sample.setdefault(sid, {})[t] = i
    ths = [0.0, 0.25, 0.5, 0.75, 1.0]
    tot = {("prior", 0.0): [0, 0.0], ("oracle", 0.0): [0, 0.0], **{("scorer", th): [0, 0.0] for th in ths}}
    for g in ("guard0.5", "guard1.0", "guard2.0"):
        if g in ev:
            tot[(g, 0.0)] = [0, 0.0]
    for sid, pos in by_sample.items():
        N = max(pos) + 1
        for key, acc in tot.items():
            who, th = key
            t = 0
            c = 0.0
            while t < N:
                i = pos.get(t)
                if i is None or (who == "scorer" and ev["exp"][i] < th):
                    c += 1.0
                    t += 1
                else:
                    c += cost_ratio
                    t += int(ev[who][i]) + 1
            acc[0] += N
            acc[1] += c
    for (who, th), (n, c) in tot.items():
        r[f"speedup_{who}" + (f"_th{th}" if who == "scorer" else "")] = n / c
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default=f"{DATA}/oracle_v2_Qwen3-4B.pkl")
    ap.add_argument("--extract", default=f"{DATA}/extract_v2_Qwen3-4B_0,{DATA}/extract_v2_Qwen3-4B_1")
    ap.add_argument("--traces", default=f"{DATA}/traces_v2_Qwen3-4B.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--bench", default=f"{DATA}/bench_verify.json")
    ap.add_argument("--domain", default="toolcall")
    ap.add_argument("--source", default="finite", help="finite / union")
    ap.add_argument("--L", type=int, default=8)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--layer", type=int, default=2, help="extract LAYERS のインデックス (0:9 1:18 2:27 3:36)")
    ap.add_argument("--encoder", default="pool")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--save", default=None)
    ap.add_argument("--test-pkl", default="", help="test を別 pkl から (例: FP8 モデルの trace)")
    ap.add_argument("--test-extract", default="", help="test 用 extract dir (カンマ区切り)")
    ap.add_argument("--test-traces", default="")
    ap.add_argument("--load", default="", help="学習済み scorer を読み込み、学習をスキップして評価だけ行う")
    args = ap.parse_args()
    dev = "cuda:0"
    d = pickle.load(open(args.pkl, "rb"))
    samples = [s for s in d["samples"] if s["domain"] == args.domain]
    traces = {json.loads(l)["sample_id"]: json.loads(l) for l in open(args.traces)}
    for s in samples:  # anchor token x at position t = full[P+t-1]
        tr = traces[s["sample_id"]]
        full = tr["prompt_ids"] + tr["output_ids"]
        s["anchor"] = [full[s["P"] + t - 1] for t in range(s["N"])]
    hidden_all, index_all = [], {}
    for ei, exd in enumerate(args.extract.split(",")):
        ex = Path(exd)
        hidden_all.append(np.load(ex / "hidden.npy", mmap_mode="r"))
        for l in open(ex / "index.jsonl"):
            e = json.loads(l)
            e["ex"] = ei
            index_all[e["sample_id"]] = e
    emb = load_embeddings(args.model).to(dev, torch.float32)
    train = Data([s for s in samples if s["split"] == "train"], None, args.source, args.L, args.K, args.layer, hidden_all, index_all, args.max_train)
    if args.test_pkl:
        td = pickle.load(open(args.test_pkl, "rb"))
        tsamples = [s for s in td["samples"] if s["domain"] == args.domain]
        ttraces = {json.loads(l)["sample_id"]: json.loads(l) for l in open(args.test_traces or args.traces)}
        for s in tsamples:
            tr = ttraces[s["sample_id"]]
            full = tr["prompt_ids"] + tr["output_ids"]
            s["anchor"] = [full[s["P"] + t - 1] for t in range(s["N"])]
        thidden, tindex = [], {}
        for ei, exd in enumerate((args.test_extract or args.extract).split(",")):
            ex = Path(exd)
            thidden.append(np.load(ex / "hidden.npy", mmap_mode="r"))
            for l in open(ex / "index.jsonl"):
                e = json.loads(l); e["ex"] = ei; tindex[e["sample_id"]] = e
        test = Data([s for s in tsamples if s["split"] == "test"], None, args.source, args.L, args.K, args.layer, thidden, tindex)
    else:
        test = Data([s for s in samples if s["split"] == "test"], None, args.source, args.L, args.K, args.layer, hidden_all, index_all)
    print(f"train positions {len(train)}  test positions {len(test)}", flush=True)
    model = BlockScorer(emb.shape[1], encoder=args.encoder, max_len=args.L).to(dev)
    print("scorer params", sum(p.numel() for p in model.parameters()) / 1e6, "M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * (len(train) // args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=max(steps, 1), pct_start=0.1)
    rng = np.random.default_rng(0)
    t0 = time.time()
    step = 0
    if args.load:
        model.load_state_dict(torch.load(args.load, map_location=dev)["state"])
        args.epochs = 0
    for ep in range(args.epochs):
        model.train()
        perm = rng.permutation(len(train))
        for s in range(0, len(perm) - args.bs + 1, args.bs):
            h, xe, ce, cm, km, src, pr, match = train.batch(perm[s : s + args.bs].tolist(), emb, dev)
            logits = model(h, xe, ce, cm, src, pr)
            l_ord, l_rank = scorer_loss(logits, cm, km, match)
            loss = l_ord + l_rank
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 200 == 0:
                print(f"ep{ep} step{step} ord={l_ord.item():.3f} rank={l_rank.item():.3f} {time.time()-t0:.0f}s", flush=True)
    bench = json.load(open(args.bench))
    st = {int(k): v["wall_median_ms"] for k, v in bench["steps"].items()}
    kk = min(st, key=lambda k: abs(k - (args.L + 1)))
    cost_ratio = st[kk] / st[1]
    ev = evaluate(model, test, emb, dev)
    res = {"config": vars(args), "train_positions": len(train), "test_positions": len(test), "cost_ratio": cost_ratio}
    res["all"] = summarize(ev, args.L, cost_ratio)
    # split 別 (toolcall: schema_known / python: repo)
    keyf = (lambda m: f"schema_known={m.get('schema_known')}") if args.domain == "toolcall" else (lambda m: m.get("repo"))
    groups = {}
    for i, (dom, meta, sid, t) in enumerate(ev["meta"]):
        groups.setdefault(keyf(meta), []).append(i)
    for g, idxs in sorted(groups.items(), key=lambda kv: -len(kv[1]))[:8]:
        sub = {k: ev[k][idxs] for k in ev if k not in ("meta", "n")}
        sub["meta"] = [ev["meta"][i] for i in idxs]
        res[f"group:{g}"] = summarize(sub, args.L, cost_ratio)
    print(json.dumps({k: v for k, v in res.items() if k != "config"}, indent=1))
    out = Path(args.out or f"experiments/jevpick/phase2_scorer/result_{args.domain}_{args.source}_L{args.L}_layer{args.layer}_{args.encoder}.json")
    json.dump(res, open(out, "w"), indent=1)
    if args.save:
        torch.save({"state": model.state_dict(), "config": vars(args)}, args.save)
    print("->", out)


if __name__ == "__main__":
    main()
