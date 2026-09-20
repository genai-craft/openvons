"""judge の学習 head (openvons の本筋): 凍結 VLM の hidden state に小さな head を載せ、ラベル付き動画で学習する。

  CUDA_VISIBLE_DEVICES=5 .venv/bin/python examples/judge/train_head.py --extract     # 窓ごとの特徴を抽出 (state/judge/features.npz)
  .venv/bin/python examples/judge/train_head.py --train                                # 項目別 head を学習・評価 (clip index 0-6 学習 / 7-9 評価)

特徴 = 窓の prefill の最終 token の hidden (最終層と 3/4 層を連結) + 質問方式の 10 項目 × 3 logit (ゼロショットの答え)。
ラベル (窓単位) = 問題あり動画の事象区間に重なる窓 → 1、それ以外 → 0。
head = 項目ごとの 2 層 MLP (数万パラメータ)。評価は窓 AUROC と、動画単位 (最大確率が閾値以上 → 検出) の正解率。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.judge.checks import BY_KEY, CHECKS, HEAD_KEYS  # noqa: E402

DATA = Path(os.environ.get("JUDGE_DATA", str(Path(__file__).resolve().parents[2] / "state" / "judge")))
CLIPS = DATA / "clips"
FEAT = DATA / "features.npz"


def extract(model: str, shard: tuple[int, int] = (0, 1)):
    from examples.judge.server import vq
    from openvons.vision.video_judge import VideoJudge, read_frames
    judge = VideoJudge(model); judge.T = {}
    man = json.load(open(CLIPS / "manifest.json"))
    qs = [vq(BY_KEY[k]) for k in HEAD_KEYS]
    feats, zs, meta = [], [], []
    man = [m for k, m in enumerate(man) if k % shard[1] == shard[0]]
    with torch.inference_mode():
        for m in man:
            # その動画の場面 (店舗・街 / スポーツ …) にある項目の窓設定だけ抽出する (+ 共通の 2 fps × 4 秒)。スポーツの 10 fps 窓を全動画で作ると 5 倍かかる
            scene = BY_KEY[m["check"]].scene
            for (fps, win) in sorted({(c.fps, c.window_s) for c in CHECKS if c.scene == scene} | {(2.0, 4.0)}):
                frames, ts, dur = read_frames(str(CLIPS / m["file"]), fps, 60)
                n_win = max(1, int(round(win * fps))); step = max(1, n_win // 2)
                for s in range(0, max(1, len(frames) - n_win + 1), step):
                    fr = frames[s : s + n_win]
                    inp, cache = judge._prefill(fr, "Fixed camera footage.")
                    # 最終 token の hidden (最終層 + 3/4 層)
                    out = judge.model(**inp, output_hidden_states=True, use_cache=False)
                    hs = out.hidden_states; L = len(hs) - 1
                    h = torch.cat([hs[L][0, -1], hs[(3 * L) // 4][0, -1]]).float().cpu().numpy()
                    z = np.concatenate([judge._ask(inp, cache, q) for q in qs if (q.fps, q.window_s) == (fps, win)])  # 同じ窓設定の質問のみ
                    zfull = np.zeros(len(qs) * 3, np.float32)
                    j = 0
                    for qi, q in enumerate(qs):
                        if (q.fps, q.window_s) == (fps, win):
                            zfull[qi * 3 : qi * 3 + 3] = z[j : j + 3]; j += 3
                    t0, t1 = ts[s], ts[min(s + n_win, len(ts)) - 1]
                    ev = m["label"] == 1 and m.get("event_start") is not None and t1 > m["event_start"] and t0 < m["event_end"]
                    feats.append(h.astype(np.float16)); zs.append(zfull)
                    meta.append({"file": m["file"], "check": m["check"], "label": m["label"], "idx": int(m["file"].rsplit("problem", 1)[-1].rsplit("ok", 1)[-1].split(".")[0]),
                                 "fps": fps, "win": win, "t0": t0, "t1": t1, "event": int(ev)})
                    del cache
            print(m["file"], len(meta), flush=True)
    out = FEAT if shard[1] == 1 else FEAT.with_name(f"features_shard{shard[0]}of{shard[1]}.npz")
    np.savez_compressed(out, h=np.stack(feats), z=np.stack(zs), meta=json.dumps(meta))
    print("->", out, np.stack(feats).shape)


def merge_shards(n: int):
    """features_shard*.npz を features.npz にまとめる。"""
    hs, zs, metas = [], [], []
    for i in range(n):
        d = np.load(FEAT.with_name(f"features_shard{i}of{n}.npz"), allow_pickle=True)
        hs.append(d["h"]); zs.append(d["z"]); metas += json.loads(str(d["meta"]))
    np.savez_compressed(FEAT, h=np.concatenate(hs), z=np.concatenate(zs), meta=json.dumps(metas))
    print("->", FEAT, np.concatenate(hs).shape)


class Head(torch.nn.Module):
    def __init__(self, d_h, d_z, d=128):
        super().__init__()
        self.norm = torch.nn.LayerNorm(d_h)
        self.net = torch.nn.Sequential(torch.nn.Linear(d_h + d_z, d), torch.nn.GELU(), torch.nn.Dropout(0.2), torch.nn.Linear(d, 1))

    def forward(self, h, z):
        return self.net(torch.cat([self.norm(h), z], -1)).squeeze(-1)


def auroc(s, y):
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return None
    return float(np.mean([[1.0 if a > b else 0.5 if a == b else 0.0 for b in neg] for a in pos]))


def neighbor_index(meta: list[dict], n: int) -> np.ndarray:
    """窓 i の前後 n 個の窓の index (同じ動画・同じ窓設定、時刻順)。端は自分自身で埋める。形 [N, 2n+1] (中央が自分)。"""
    groups: dict[tuple, list[int]] = {}
    for i, m in enumerate(meta):
        groups.setdefault((m["file"], m["fps"], m["win"]), []).append(i)
    nb = np.zeros((len(meta), 2 * n + 1), np.int64)
    for g in groups.values():
        g = sorted(g, key=lambda i: meta[i]["t0"])
        for k, i in enumerate(g):
            for o in range(-n, n + 1):
                nb[i, o + n] = g[min(max(k + o, 0), len(g) - 1)]
    return nb


def with_ctx(H: torch.Tensor, Z: torch.Tensor, meta: list[dict], n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """前後 n 窓の特徴を連結 → 「前後の状況」を head に見せる (n=0 なら元のまま)。"""
    if n <= 0:
        return H, Z
    nb = torch.tensor(neighbor_index(meta, n), device=H.device)
    return H[nb].flatten(1), Z[nb].flatten(1)


def train(test_from: int = 7, epochs: int = 300, use_z: bool = True, mil_epochs: int = 150, agg: str = 'top2', ctx: int = 0, tag: str = ""):
    torch.manual_seed(0); np.random.seed(0)   # 再現性 (項目ごとの数字が走らせるたびに ±0.1 動いていた)
    d = np.load(FEAT, allow_pickle=True)
    H = torch.tensor(d["h"].astype(np.float32)); Z = torch.tensor(d["z"]); meta = json.loads(str(d["meta"]))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    H, Z = H.to(dev), Z.to(dev)
    H, Z = with_ctx(H, Z, meta, ctx)
    results = {}
    for c in [BY_KEY[k] for k in HEAD_KEYS]:
        # この項目の窓設定の窓だけ使う。負例は同じシーン (+ 他シーン少量) の全動画から
        idx_all = [i for i, m in enumerate(meta) if (m["fps"], m["win"]) == (c.fps, c.window_s)]
        y = np.array([1 if (meta[i]["check"] == c.key and meta[i]["event"]) else 0 for i in idx_all])
        same = np.array([meta[i]["check"] == c.key for i in idx_all])
        clipidx = np.array([meta[i]["idx"] for i in idx_all])
        # 動画が少ない項目 (20 本 = idx 0〜9) は分割点を半分にする (40 本の項目は idx 14 から、20 本の項目は 7 から)
        n_idx = int(clipidx[same].max()) + 1 if same.any() else 20
        tf = test_from if n_idx >= 20 else max(1, test_from // 2)
        tr = np.array([(clipidx[k] < tf) for k in range(len(idx_all))]) & (same | (clipidx < tf))
        te = (~tr) & same  # 評価はこの項目の動画 (idx >= test_from) のみ
        ii = torch.tensor(idx_all, device=dev)
        Zc = Z[ii] if use_z else torch.zeros(len(idx_all), 0, device=dev)
        head = Head(H.shape[1], Zc.shape[1]).to(dev)
        opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.05)
        yt = torch.tensor(y, dtype=torch.float32, device=dev)
        trm = torch.tensor(tr, device=dev)
        pos_w = torch.tensor(max(1.0, float((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1))), device=dev)
        for ep in range(epochs):
            head.train(); opt.zero_grad()
            logit = head(H[ii][trm], Zc[trm])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, yt[trm], pos_weight=pos_w)
            loss.backward(); opt.step()
        # MIL 微調整: 動画ごとに窓 logit の top-2 平均を動画スコアにし、動画ラベルで BCE
        files = np.array([meta[i]["file"] for i in idx_all]); ylab = np.array([meta[i]["label"] if meta[i]["check"] == c.key else 0 for i in idx_all])
        tr_files = sorted({files[k] for k in range(len(idx_all)) if tr[k]})
        groups = [torch.tensor(np.where(files == f)[0], device=dev) for f in tr_files]
        glabel = torch.tensor([float(ylab[np.where(files == f)[0][0]]) for f in tr_files], device=dev)
        opt2 = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=0.05)
        for ep in range(mil_epochs):
            head.train(); opt2.zero_grad()
            lg = head(H[ii], Zc)
            cs = torch.stack([lg[g].topk(min(2, len(g))).values.mean() for g in groups])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(cs, glabel)
            loss.backward(); opt2.step()
        head.eval()
        with torch.no_grad():
            p = torch.sigmoid(head(H[ii], Zc)).cpu().numpy()
        # 窓 AUROC (test) と 動画単位の正解率 (test)
        a = auroc(p[te], y[te])
        clips = {}
        for k in np.where(te)[0]:
            m = meta[idx_all[k]]; clips.setdefault(m["file"], {"label": m["label"], "p": []})["p"].append(p[k])
        zs = Z[ii][:, HEAD_KEYS.index(c.key) * 3 : HEAD_KEYS.index(c.key) * 3 + 3].cpu().numpy()
        zp = np.exp(zs - zs.max(1, keepdims=True)); zp = zp[:, 0] / zp.sum(1)   # ゼロショットの p(yes)
        zclips = {}
        for k in np.where(te)[0]:
            m = meta[idx_all[k]]; zclips.setdefault(m["file"], {"label": m["label"], "p": []})["p"].append(zp[k])
        # 閾値は学習側の動画で決める (max p の中間)
        trclips = {}
        for k in np.where(tr & same)[0]:
            m = meta[idx_all[k]]; trclips.setdefault(m["file"], {"label": m["label"], "p": []})["p"].append(p[k])
        # 閾値は学習側の動画で正解率が最大になる点 (max p を使う)
        def score(ps):
            ps = sorted(ps, reverse=True); return float(np.mean(ps[:2])) if agg == "top2" else float(ps[0])
        tr_scores = np.array([score(v["p"]) for v in trclips.values()]); tr_labels = np.array([v["label"] for v in trclips.values()])
        thr = 0.5
        if len(tr_scores):
            cands = np.unique(np.concatenate([tr_scores, [0.5]]))
            accs = [((tr_scores >= t) == (tr_labels == 1)).mean() for t in cands]
            best = max(accs); thr = float(np.median([t for t, a_ in zip(cands, accs) if a_ == best]))
        acc = float(np.mean([(score(v["p"]) >= thr) == (v["label"] == 1) for v in clips.values()])) if clips else None
        zacc = float(np.mean([(max(v["p"]) >= 0.4) == (v["label"] == 1) for v in zclips.values()])) if zclips else None
        za = auroc(zp[te], y[te])
        results[c.key] = {"title": c.title, "n_train_windows": int(tr.sum()), "n_test_clips": len(clips), "head_window_auroc": a, "head_clip_acc": acc, "thr": thr,
                          "zeroshot_window_auroc": za, "zeroshot_clip_acc": zacc}
        print(f"{c.title:14s} head AUROC {a if a is None else round(a,3)} acc {acc if acc is None else round(acc,2)} | zero-shot AUROC {za if za is None else round(za,3)} acc {zacc if zacc is None else round(zacc,2)}  (test clips {len(clips)})", flush=True)
        torch.save({"state": head.state_dict(), "d_h": H.shape[1], "d_z": Zc.shape[1], "thr": thr, "ctx": ctx}, DATA / f"head_{c.key}{tag}.pt")
    json.dump(results, open(DATA / f"head_eval{tag}.json", "w"), ensure_ascii=False, indent=1)
    hs = [v["head_clip_acc"] for v in results.values() if v["head_clip_acc"] is not None]; zs_ = [v["zeroshot_clip_acc"] for v in results.values() if v["zeroshot_clip_acc"] is not None]
    print("mean clip acc: head %.3f  zero-shot %.3f" % (np.mean(hs), np.mean(zs_)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--shard", default="0/1", help="抽出を分割する i/n (GPU ごとに)")
    ap.add_argument("--merge", type=int, default=0, help="n 個の shard をまとめる")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--model", default=os.environ.get("JUDGE_MODEL", "Qwen/Qwen3-VL-4B-Instruct"))
    ap.add_argument("--test-from", type=int, default=7)
    ap.add_argument("--no-z", action="store_true")
    ap.add_argument("--mil-epochs", type=int, default=150)
    ap.add_argument("--agg", default="top2")
    ap.add_argument("--ctx", type=int, default=0, help="前後 N 窓の特徴も連結して学習する (時間文脈)")
    ap.add_argument("--tag", default="", help="出力ファイル名の接尾辞 (例 _ctx1)")
    a = ap.parse_args()
    if a.extract:
        i, n = a.shard.split("/"); extract(a.model, (int(i), int(n)))
    if a.merge:
        merge_shards(a.merge)
    if a.train:
        train(a.test_from, use_z=not a.no_z, mil_epochs=a.mil_epochs, agg=a.agg, ctx=a.ctx, tag=a.tag)
