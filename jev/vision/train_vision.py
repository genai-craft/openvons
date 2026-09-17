"""視覚エンコーダ単体 Decision Model の学習 (TASK-006 の画像版)。

視覚タワーを凍結したまま特徴を 1 回抽出してキャッシュし、質問ごとの小さな head だけを学習する。

usage:
  python training/train_vision.py --exp vis_fairface_2b --data /data/decision_model/data/vision/fairface \
      --model Qwen/Qwen3-VL-2B-Instruct --epochs 15
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from jev.core.metrics import all_metrics, reliability_table
from jev.core.temperature import TemperatureScaler
from jev.vision.vision_model import VisionDecisionConfig, VisionDecisionModel

ROOT = Path(__file__).resolve().parents[1]
FEAT = Path("/data/decision_model/features/vision")


def load_split(data: Path, split: str, limit=None):
    rows = [json.loads(l) for l in open(data / f"{split}.jsonl") if l.strip()]
    return rows[:limit] if limit else rows


@torch.no_grad()
def extract(model: VisionDecisionModel, data: Path, rows, bs: int, desc: str) -> torch.Tensor:
    out = []
    for i in tqdm(range(0, len(rows), bs), desc=desc, mininterval=5):
        imgs = [Image.open(data / r["image"]).convert("RGB") for r in rows[i : i + bs]]
        out.append(model.patch_features(imgs).cpu())
    return torch.cat(out)


def cached(model, data: Path, split: str, rows, bs, tag) -> torch.Tensor:
    FEAT.mkdir(parents=True, exist_ok=True)
    p = FEAT / f"{tag}__{split}__{len(rows)}.pt"
    if p.exists():
        return torch.load(p)
    f = extract(model, data, rows, bs, f"features {split}")
    torch.save(f, p)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--data", default="/data/decision_model/data/vision/fairface")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--pooling", default="meanmax", choices=["mean", "meanmax", "attn"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--feat_bs", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--balanced", action="store_true", help="クラス頻度の逆数で重み付けした損失を使う (不均衡データ用)")
    a = ap.parse_args()
    data = Path(a.data)
    tasks = json.load(open(data / "tasks.json"))["questions"]
    cfg = VisionDecisionConfig(model_name=a.model, pooling=a.pooling, questions=tasks)
    model = VisionDecisionModel(cfg)
    print(f"凍結: 視覚エンコーダ {model.n_frozen()/1e6:.0f}M / 学習: head {model.n_trainable()/1e3:.1f}K params")

    t0 = time.time()
    tag = f"{Path(a.data).name}__{a.model.replace('/', '_')}__{a.pooling}"
    splits = {s: load_split(data, s, a.limit if s == "train" else None) for s in ["train", "valid", "test"]}
    feats = {s: cached(model, data, s, splits[s], a.feat_bs, tag) for s in splits}
    t_feat = time.time() - t0
    labels = {s: {k: torch.tensor([r["labels"][k] for r in splits[s]]) for k in tasks} for s in splits}

    result = {"experiment": a.exp, "model": a.model, "pooling": a.pooling, "kind": "vision_only", "balanced": a.balanced,
              "frozen_params": model.n_frozen(), "trainable_params": model.n_trainable(),
              "n_train": len(splits["train"]), "feature_seconds": round(t_feat, 1), "tasks": {}}
    t1 = time.time()
    for key in tasks:
        head = model.heads[key]
        params = list(head.parameters()) + list(model.norm.parameters())
        opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd)
        cw = None
        if a.balanced:
            cnt = torch.bincount(labels["train"][key], minlength=len(tasks[key]["choices"])).float()
            cw = (cnt.sum() / (len(cnt) * cnt.clamp(min=1))).to(model.device)
        X, y = feats["train"].to(model.device), labels["train"][key].to(model.device)
        n = len(y)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * ((n + a.bs - 1) // a.bs), pct_start=0.1)
        for ep in range(a.epochs):
            perm = torch.randperm(n, device=model.device)
            for i in range(0, n, a.bs):
                idx = perm[i : i + a.bs]
                loss = torch.nn.functional.cross_entropy(model.logits(X[idx], key), y[idx], weight=cw)
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        with torch.no_grad():
            pv = torch.softmax(model.logits(feats["valid"].to(model.device), key), -1).cpu().numpy()
            pt = torch.softmax(model.logits(feats["test"].to(model.device), key), -1).cpu().numpy()
        yv, yt = labels["valid"][key].numpy(), labels["test"][key].numpy()
        ts = TemperatureScaler().fit(pv, yv)
        model.temperature[key] = ts.T
        m_raw, m_cal = all_metrics(pt, yt), all_metrics(ts.transform(pt), yt)
        pred = pt.argmax(1)
        per_class = []
        for c in range(len(tasks[key]["choices"])):
            mask = yt == c
            per_class.append({"id": tasks[key]["choices"][c]["id"], "support": int(mask.sum()),
                              "recall": float((pred[mask] == c).mean()) if mask.any() else None,
                              "precision": float((yt[pred == c] == c).mean()) if (pred == c).any() else None})
        majority = float(np.bincount(yt, minlength=len(tasks[key]["choices"])).max() / len(yt))
        result["tasks"][key] = {"n_options": len(tasks[key]["choices"]), "test": m_raw, "test_calibrated": m_cal,
                                "temperature": ts.T, "reliability": reliability_table(pt, yt),
                                "per_class": per_class, "majority_baseline": majority}
        print(f"  {key:12s} ({len(tasks[key]['choices'])}択) acc={m_raw['accuracy']:.4f} (多数派={majority:.4f}) "
              f"macroF1={m_raw['macro_f1']:.4f} ece={m_raw['ece']:.4f}->{m_cal['ece']:.4f}")
        print("               クラス別: " + "  ".join(
              f"{c['id']}(n={c['support']}) recall={c['recall']:.3f}" if c["recall"] is not None else f"{c['id']}(n=0)"
              for c in per_class))
    result["head_train_seconds"] = round(time.time() - t1, 1)

    ck = Path("/data/decision_model/checkpoints") / a.exp
    ck.mkdir(parents=True, exist_ok=True)
    model.save(str(ck))
    for key in tasks:
        with torch.no_grad():
            pt = torch.softmax(model.logits(feats["test"].to(model.device), key), -1).cpu().numpy()
        np.save(ck / f"probs_test_{key}.npy", pt)

    # 単画像レイテンシ
    imgs = [Image.open(data / r["image"]).convert("RGB") for r in splits["test"][:60]]
    for im in imgs[:10]:
        model.decide([im])
    lat = []
    for im in imgs:
        torch.cuda.synchronize(); t = time.perf_counter(); model.decide([im]); torch.cuda.synchronize()
        lat.append((time.perf_counter() - t) * 1000)
    result["latency_p50_ms"] = float(np.percentile(lat, 50))
    result["latency_p95_ms"] = float(np.percentile(lat, 95))
    result["gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    json.dump(result, open(ROOT / "experiments" / f"{a.exp}.json", "w"), indent=2, ensure_ascii=False)
    print(f"特徴抽出 {t_feat:.0f}s / head 学習 {result['head_train_seconds']:.1f}s / "
          f"単画像 p50 {result['latency_p50_ms']:.1f}ms / VRAM {result['gpu_mem_gb']}GB")


if __name__ == "__main__":
    main()
