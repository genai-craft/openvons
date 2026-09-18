"""端末に載せる「学習済みヘッド」を作る (凍結 SigLIP2 + 小さな head).

    CUDA_VISIBLE_DEVICES=4 .venv/bin/python scripts/train_ondevice_head.py --task fairface

端末にはすでに SigLIP2 の画像側 (int8) が載っている。その出力 (768 次元の正規化済み埋め込み) の
上に小さな head を足すだけで、年齢・性別は学習なしの零ショットより大幅に良くなる。
head は数万パラメータなので、ONNX にせず JSON で配って端末側で計算する。

出力: <out>/head_<task>.json
  {dim, tasks: {age: {labels, w1, b1, w2, b2, temperature, metrics}, gender: {...}}}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FEAT = Path("/data/openjev/features")


class Head(nn.Module):
    """LayerNorm → Linear → GELU → 質問ごとの Linear。

    正規化を入れないと学習が進まない (実験済み)。幹は質問間で共有する。
    別々に持つと 768×hidden が質問の数だけ増えて、端末に配る JSON が無駄に太る。
    """

    def __init__(self, dim: int, n_outs: dict[str, int], hidden: int = 128):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.out = nn.ModuleDict({k: nn.Linear(hidden, n) for k, n in n_outs.items()})

    def trunk(self, x):
        return torch.nn.functional.gelu(self.fc1(self.norm(x)))

    def forward(self, x, key: str):
        return self.out[key](self.trunk(x))


def embed_onnx(rows, data: Path, onnx_dir: Path, bs: int = 16) -> torch.Tensor:
    """端末が実際に動かすのと同じ int8 ONNX で埋め込む。

    int8 は fp32 との コサイン類似が 0.89 程度しかなく、学習済みヘッドはこのズレに弱い
    (fp32 特徴で学習したヘッドを int8 に載せると年齢 0.57 → 0.32 に落ちた)。
    端末で使う表現そのもので学習する。
    """
    import numpy as np
    import onnxruntime as ort
    pre = json.loads((onnx_dir / "preprocess.json").read_text())
    size, mean, std = pre["size"], np.array(pre["mean"]), np.array(pre["std"])
    so = ort.SessionOptions(); so.intra_op_num_threads = 16
    sess = ort.InferenceSession(str(onnx_dir / "image_encoder_quantized.onnx"), so, providers=["CPUExecutionProvider"])

    def prep(path):
        a = (np.asarray(Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)) / 255.0 - mean) / std
        return a.transpose(2, 0, 1).astype(np.float32)

    out = []
    for i in range(0, len(rows), bs):
        px = np.stack([prep(data / r["image"]) for r in rows[i : i + bs]])
        out.append(torch.from_numpy(sess.run(None, {"pixel_values": px})[0]))
        if i % (bs * 200) == 0:
            print(f"  {i}/{len(rows)}", flush=True)
    return torch.cat(out).float()


@torch.no_grad()
def embed(rows, data: Path, model, proc, device: str, bs: int = 256) -> torch.Tensor:
    """端末と同じ表現 (正規化済みプール埋め込み) を取り出す。"""
    out = []
    for i in range(0, len(rows), bs):
        imgs = [Image.open(data / r["image"]).convert("RGB") for r in rows[i : i + bs]]
        px = proc(images=imgs, return_tensors="pt")["pixel_values"].to(device)
        f = model.get_image_features(pixel_values=px)
        f = f.pooler_output if hasattr(f, "pooler_output") else f
        out.append((f / f.norm(dim=-1, keepdim=True)).float().cpu())
        if i % (bs * 20) == 0:
            print(f"  {i}/{len(rows)}", flush=True)
    return torch.cat(out)


def cached_embed(tag: str, rows, data, model, proc, device, onnx_dir: Path | None = None) -> torch.Tensor:
    FEAT.mkdir(parents=True, exist_ok=True)
    p = FEAT / f"{tag}_{len(rows)}.pt"
    if p.exists():
        return torch.load(p)
    t0 = time.time()
    f = embed_onnx(rows, data, onnx_dir) if onnx_dir else embed(rows, data, model, proc, device)
    torch.save(f, p)
    print(f"  {tag}: {len(rows)} 枚 {time.time() - t0:.0f}s -> {p}")
    return f


def fit_all(xtr, ytr: dict, xva, yva: dict, xte, yte: dict, labels: dict, epochs: int, device: str):
    """幹を共有して全部の質問を同時に学習する。返り値は質問ごとの重みと実測値。"""
    n_outs = {k: len(v) for k, v in labels.items()}
    head = Head(xtr.shape[1], n_outs).to(device)
    # 少数クラスを潰さないようにクラス重みを入れる (年齢の 0-2 や 70+ は数が少ない)
    ws = {}
    for k, y in ytr.items():
        cnt = np.bincount(y.numpy(), minlength=n_outs[k]).astype(np.float32)
        w = torch.tensor((cnt.sum() / np.maximum(cnt, 1)) ** 0.5, device=device)
        ws[k] = w / w.mean()
    opt = torch.optim.AdamW(head.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    xtr_d = xtr.to(device)
    ytr_d = {k: v.to(device) for k, v in ytr.items()}
    xva_d = xva.to(device)
    best, best_state = -1.0, None
    for ep in range(epochs):
        head.train()
        perm = torch.randperm(len(xtr_d), device=device)
        for i in range(0, len(perm), 1024):
            idx = perm[i : i + 1024]
            h = head.trunk(xtr_d[idx])
            loss = sum(torch.nn.functional.cross_entropy(head.out[k](h), ytr_d[k][idx], weight=ws[k]) for k in labels)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        head.eval()
        with torch.no_grad():
            accs = {k: (head(xva_d, k).argmax(1).cpu() == yva[k]).float().mean().item() for k in labels}
        score = float(np.mean(list(accs.values())))
        if score > best:
            best, best_state = score, {k: v.detach().clone() for k, v in head.state_dict().items()}
        print(f"  epoch {ep + 1}/{epochs} valid " + " ".join(f"{k} {v:.4f}" for k, v in accs.items()), flush=True)
    head.load_state_dict(best_state)
    head.eval()
    return head, {k: calibrate_and_score(head, k, xva, yva[k], xte, yte[k], labels[k], device) for k in labels}


def calibrate_and_score(head, name: str, xva, yva, xte, yte, labels, device: str) -> dict:
    n = len(labels)
    # 温度スケーリング (valid で合わせる)。確率をそのまま判断に使うので校正は必須
    with torch.no_grad():
        lv = head(xva.to(device), name).cpu()
    logT = torch.zeros(1, requires_grad=True)
    optT = torch.optim.LBFGS([logT], lr=0.1, max_iter=60)

    def closure():
        optT.zero_grad()
        loss = torch.nn.functional.cross_entropy(lv / logT.exp(), yva)
        loss.backward()
        return loss

    optT.step(closure)
    T = float(logT.exp().item())

    with torch.no_grad():
        lt = head(xte.to(device), name).cpu() / T
        pr = torch.softmax(lt, 1)
        pred = lt.argmax(1)
    acc = (pred == yte).float().mean().item()
    f1 = []
    for c in range(n):
        tp = ((pred == c) & (yte == c)).sum().item()
        fp = ((pred == c) & (yte != c)).sum().item()
        fn = ((pred != c) & (yte == c)).sum().item()
        f1.append(2 * tp / max(2 * tp + fp + fn, 1))
    conf = pr.max(1).values
    hi = conf >= 0.9
    metrics = {"test_acc": round(acc, 4), "macro_f1": round(float(np.mean(f1)), 4), "temperature": round(T, 3),
               "coverage@0.9": round(float(hi.float().mean()), 4),
               "precision@0.9": round(float((pred[hi] == yte[hi]).float().mean()) if hi.any() else 0.0, 4)}
    print(f"  [{name}] test acc {acc:.4f} macroF1 {np.mean(f1):.4f} T={T:.2f} "
          f"(確信度0.9以上: {metrics['coverage@0.9']:.0%} を {metrics['precision@0.9']:.0%} で)")
    sd = head.state_dict()
    r5 = lambda t: [[round(v, 5) for v in row] for row in t.cpu().tolist()]
    r5b = lambda t: [round(v, 5) for v in t.cpu().tolist()]
    return {
        "labels": labels,
        "w2": r5(sd[f"out.{name}.weight"]), "b2": r5b(sd[f"out.{name}.bias"]),
        "temperature": round(T, 4), "metrics": metrics,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data/decision_model/data/vision/fairface")
    ap.add_argument("--model", default="google/siglip2-base-patch16-224")
    ap.add_argument("--out", default="/data/openjev/models/ondevice/vision-choices")
    ap.add_argument("--task", default="fairface")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--limit", type=int, default=0, help="動作確認用に枚数を絞る")
    ap.add_argument("--onnx-dir", default="", help="端末と同じ int8 ONNX で特徴を作る (推奨)")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = Path(args.data)
    tasks = json.loads((data / "tasks.json").read_text())["questions"]

    def rows(split):
        r = [json.loads(l) for l in open(data / f"{split}.jsonl") if l.strip()]
        return r[: args.limit] if args.limit else r

    tr, va, te = rows("train"), rows("valid"), rows("test")
    print(f"train {len(tr)} valid {len(va)} test {len(te)}")
    onnx_dir = Path(args.onnx_dir) if args.onnx_dir else None
    model = proc = None
    if onnx_dir is None:
        from transformers import AutoModel, AutoProcessor
        model = AutoModel.from_pretrained(args.model).eval().to(device)
        proc = AutoProcessor.from_pretrained(args.model)
    tag = f"{args.task}_siglip2" + ("_int8" if onnx_dir else "")
    xtr = cached_embed(f"{tag}_train", tr, data, model, proc, device, onnx_dir)
    xva = cached_embed(f"{tag}_valid", va, data, model, proc, device, onnx_dir)
    xte = cached_embed(f"{tag}_test", te, data, model, proc, device, onnx_dir)

    #: 画面に出す日本語。年齢は幅なので、そのまま帯で見せる
    JA = {
        "age": {"0-2": "0-2 歳", "3-9": "3-9 歳", "10-19": "10 代", "20-29": "20 代", "30-39": "30 代",
                "40-49": "40 代", "50-59": "50 代", "60-69": "60 代", "more than 70": "70 歳以上"},
        "gender": {"Male": "男性", "Female": "女性"},
    }
    labels, ytr, yva, yte = {}, {}, {}, {}
    for key, spec in tasks.items():
        labels[key] = [{"id": c["id"], "label": JA.get(key, {}).get(c["id"], c["id"])} for c in spec["choices"]]
        ytr[key] = torch.tensor([r["labels"][key] for r in tr])
        yva[key] = torch.tensor([r["labels"][key] for r in va])
        yte[key] = torch.tensor([r["labels"][key] for r in te])
    head, per_task = fit_all(xtr, ytr, xva, yva, xte, yte, labels, args.epochs, device)

    sd = head.state_dict()
    r5 = lambda t: [[round(v, 5) for v in row] for row in t.cpu().tolist()]
    r5b = lambda t: [round(v, 5) for v in t.cpu().tolist()]
    titles = {"age": "年齢層", "gender": "性別"}
    doc = {"model": args.model, "dim": int(xtr.shape[1]), "trained_on": args.task,
           "features": "int8-onnx" if onnx_dir else "fp32",
           "title": "顔の年齢・性別 (学習済み)",
           # 共有の幹 (LayerNorm → Linear → GELU)
           "trunk": {"norm_w": r5b(sd["norm.weight"]), "norm_b": r5b(sd["norm.bias"]),
                     "w1": r5(sd["fc1.weight"]), "b1": r5b(sd["fc1.bias"])},
           "tasks": {k: {"title": titles.get(k, k), **v} for k, v in per_task.items()}}

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    p = out / f"head_{args.task}.json"
    p.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    n_par = sum(p_.numel() for p_ in head.parameters())
    print(f"-> {p} ({p.stat().st_size / 1e3:.0f} KB, 学習したのは {n_par / 1000:.0f}k パラメータ)")


if __name__ == "__main__":
    main()
