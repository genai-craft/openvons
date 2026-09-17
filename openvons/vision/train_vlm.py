"""小型 VLM Decision Model の学習 (画像 + 自然言語の質問)。

VLM を凍結したまま、画像を 1 回読んで全質問ぶんの (問い合わせベクトル, 選択肢ベクトル) を抽出・キャッシュし、
embed head だけを学習する。質問と選択肢がテキストなので、学習に無かった質問にもある程度対応できる。
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

from openvons.core.metrics import all_metrics, reliability_table
from openvons.core.temperature import TemperatureScaler
from openvons.core.primitives import Question
from openvons.vision.vlm_decision_model import VLMDecisionConfig, VLMDecisionModel

ROOT = Path(__file__).resolve().parents[1]
FEAT = Path("/data/decision_model/features/vlm")


def load_split(data: Path, split: str, limit=None):
    rows = [json.loads(l) for l in open(data / f"{split}.jsonl") if l.strip()]
    return rows[:limit] if limit else rows


def to_questions(tasks: dict) -> list[Question]:
    return [Question.from_dict(v, key=k) for k, v in tasks.items()]


@torch.no_grad()
def extract(model: VLMDecisionModel, data: Path, rows, qs, desc: str):
    """1 画像につき len(qs) 本の (pooled, opt_vecs) を得る。"""
    P, O, M = [], [], []
    for r in tqdm(rows, desc=desc, mininterval=10):
        img = Image.open(data / r["image"]).convert("RGB")
        f = model.features(img, qs)
        P.append(f["pooled"].to(torch.bfloat16).cpu())
        O.append(f["opt_vecs"].to(torch.bfloat16).cpu())
        M.append(f["opt_mask"].cpu())
    return torch.stack(P), torch.stack(O), torch.stack(M)     # (N, Q, H), (N, Q, Nopt, H), (N, Q, Nopt)


def cached(model, data, split, rows, qs, tag):
    FEAT.mkdir(parents=True, exist_ok=True)
    p = FEAT / f"{tag}__{split}__{len(rows)}.pt"
    if p.exists():
        d = torch.load(p)
        return d["P"], d["O"], d["M"]
    P, O, M = extract(model, data, rows, qs, f"features {split}")
    torch.save({"P": P, "O": O, "M": M}, p)
    return P, O, M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--data", default="/data/decision_model/data/vision/fairface")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--head", default="embed", choices=["embed", "mlp"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--limit", type=int, default=20000)
    ap.add_argument("--eval_limit", type=int, default=2000)
    a = ap.parse_args()
    data = Path(a.data)
    tasks = json.load(open(data / "tasks.json"))["questions"]
    qs = to_questions(tasks)
    keys = [q.key for q in qs]
    model = VLMDecisionModel(VLMDecisionConfig(model_name=a.model, head=a.head))
    n_frozen = sum(p.numel() for p in model.backbone.parameters())
    n_head = sum(p.numel() for p in model.head.parameters())
    print(f"凍結: VLM {n_frozen/1e9:.2f}B / 学習: head {n_head/1e6:.2f}M params")

    splits = {"train": load_split(data, "train", a.limit),
              "valid": load_split(data, "valid", a.eval_limit),
              "test": load_split(data, "test", a.eval_limit)}
    tag = f"{Path(a.data).name}__{a.model.replace('/', '_')}"
    t0 = time.time()
    F = {s: cached(model, data, s, splits[s], qs, tag) for s in splits}
    t_feat = time.time() - t0
    Y = {s: {k: torch.tensor([r["labels"][k] for r in splits[s]]) for k in keys} for s in splits}

    dev = model.device
    params = list(model.head.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    n = len(splits["train"])
    steps = a.epochs * ((n + a.bs - 1) // a.bs) * len(qs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    P, O, M = F["train"]
    for ep in range(a.epochs):
        perm = torch.randperm(n)
        for i in range(0, n, a.bs):
            idx = perm[i : i + a.bs]
            for qi, q in enumerate(qs):                      # 質問ごとに選択肢数が違うのでまとめて扱わない
                lg = model.head(P[idx, qi].to(dev).float(), O[idx, qi].to(dev).float(), M[idx, qi].to(dev), q.type)
                loss = torch.nn.functional.cross_entropy(lg[:, : q.n], Y["train"][q.key][idx].to(dev))
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()

    @torch.no_grad()
    def predict(split, qi, q):
        P, O, M = F[split]
        out = []
        for i in range(0, len(P), 512):
            lg = model.head(P[i:i+512, qi].to(dev).float(), O[i:i+512, qi].to(dev).float(), M[i:i+512, qi].to(dev), q.type)
            out.append(torch.softmax(lg[:, : q.n], -1).cpu())
        return torch.cat(out).numpy()

    result = {"experiment": a.exp, "model": a.model, "head": a.head, "kind": "small_vlm",
              "frozen_params": n_frozen, "trainable_params": n_head, "n_train": n,
              "feature_seconds": round(t_feat, 1), "tasks": {}}
    for qi, q in enumerate(qs):
        pv, pt = predict("valid", qi, q), predict("test", qi, q)
        yv, yt = Y["valid"][q.key].numpy(), Y["test"][q.key].numpy()
        ts = TemperatureScaler().fit(pv, yv)
        m_raw, m_cal = all_metrics(pt, yt), all_metrics(ts.transform(pt), yt)
        result["tasks"][q.key] = {"n_options": q.n, "test": m_raw, "test_calibrated": m_cal,
                                  "temperature": ts.T, "reliability": reliability_table(pt, yt)}
        print(f"  {q.key:8s} ({q.n}択) acc={m_raw['accuracy']:.4f} f1={m_raw['macro_f1']:.4f} "
              f"ece={m_raw['ece']:.4f} -> 補正後 {m_cal['ece']:.4f}  brier={m_raw['brier']:.4f}")
        ck = Path("/data/decision_model/checkpoints") / a.exp
        ck.mkdir(parents=True, exist_ok=True)
        np.save(ck / f"probs_test_{q.key}.npy", pt)
    model.save_head(str(Path("/data/decision_model/checkpoints") / a.exp))

    imgs = [Image.open(data / r["image"]).convert("RGB") for r in splits["test"][:60]]
    for im in imgs[:8]:
        model.decide(im, qs)
    lat = []
    for im in imgs:
        torch.cuda.synchronize(); t = time.perf_counter(); model.decide(im, qs); torch.cuda.synchronize()
        lat.append((time.perf_counter() - t) * 1000)
    result["latency_p50_ms_2q"] = float(np.percentile(lat, 50))
    result["gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    json.dump(result, open(ROOT / "experiments" / f"{a.exp}.json", "w"), indent=2, ensure_ascii=False)
    print(f"特徴抽出 {t_feat:.0f}s / 2質問同時 p50 {result['latency_p50_ms_2q']:.1f}ms / VRAM {result['gpu_mem_gb']}GB")


if __name__ == "__main__":
    main()
