"""TASK-009: train a decision model.

modes:
  head   : backbone frozen, features cached once, only the head is trained   (Step 1)
  lora   : LoRA on the backbone + head                                         (Step 2)
  topN   : last N transformer layers + head trainable                          (Step 3)
  full   : everything trainable                                                (Step 4)

Writes experiments/<exp>.json in the spec §13 format and saves the head (+ adapter) under
/data/decision_model/checkpoints/<exp>/.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from jev.core.metrics import all_metrics, latency_summary, reliability_table
from jev.core.temperature import TemperatureScaler
from jev.lm.models.decision_model import DecisionModel, DecisionModelConfig
from jev.lm.training.dataset import DecisionDataset, load_split
from jev.lm.training.losses import compute_loss

ROOT = Path(__file__).resolve().parents[1]
FEAT_DIR = Path("/data/decision_model/features")
CKPT_DIR = Path("/data/decision_model/checkpoints")


# ----------------------------------------------------------------------------- features
@torch.no_grad()
def extract_features(model: DecisionModel, ds: DecisionDataset, bs: int, desc: str) -> dict[str, torch.Tensor]:
    """Frozen-backbone features, kept on CPU: pooled (n,H), opt_vecs (n,N,H) bf16, opt_mask, targets, labels."""
    order = sorted(range(len(ds)), key=lambda i: len(ds.samples[i].state))
    pooled, opts, omask, tgts, labels = [], [], [], [], []
    N = max(s.question.n for s in ds.samples)
    for i in tqdm(range(0, len(order), bs), desc=desc, mininterval=5):
        idx = order[i : i + bs]
        b = ds.collate([ds[j] for j in idx])
        b = {k: v.to(model.device) for k, v in b.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            f = model.features(b)
        n_here = f["opt_vecs"].shape[1]
        ov = torch.zeros((len(idx), N, f["opt_vecs"].shape[2]), dtype=torch.bfloat16, device=model.device)
        ov[:, :n_here] = f["opt_vecs"].to(torch.bfloat16)
        om = torch.zeros((len(idx), N), dtype=torch.bool, device=model.device)
        om[:, :n_here] = f["opt_mask"]
        tg = torch.zeros((len(idx), N))
        tg[:, :n_here] = b["targets"].cpu()
        pooled.append(f["pooled"].to(torch.bfloat16).cpu()); opts.append(ov.cpu()); omask.append(om.cpu())
        tgts.append(tg); labels.append(b["labels"].cpu())
    inv = torch.empty(len(order), dtype=torch.long)
    inv[torch.tensor(order)] = torch.arange(len(order))
    cat = lambda xs: torch.cat(xs)[inv]
    return {"pooled": cat(pooled), "opt_vecs": cat(opts), "opt_mask": cat(omask), "targets": cat(tgts), "labels": cat(labels)}


def cached_features(model, ds, task, split, soft, limit, bs, target):
    key = f"{model.cfg.model_name.replace('/', '_')}__{model.cfg.pooling}__{task}__{split}__{soft or 'hard'}__{target}__{limit or 'all'}.pt"
    p = FEAT_DIR / key
    if p.exists():
        return torch.load(p)
    f = extract_features(model, ds, bs, f"features {task}/{split}")
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(f, p)
    return f


def head_logits_from_cache(model, f, idx, qtype, device):
    feats = {"pooled": f["pooled"][idx].to(device), "opt_vecs": f["opt_vecs"][idx].to(device), "opt_mask": f["opt_mask"][idx].to(device)}
    return model.logits_from_features(feats, qtype)


# ----------------------------------------------------------------------------- eval
@torch.no_grad()
def predict_cached(model, f, qtype, device, bs=1024):
    out = []
    for i in range(0, len(f["labels"]), bs):
        idx = torch.arange(i, min(i + bs, len(f["labels"])))
        out.append(torch.softmax(head_logits_from_cache(model, f, idx, qtype, device), -1).cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def predict_live(model, ds, qtype, bs=64):
    dl = DataLoader(ds, batch_size=bs, shuffle=False, collate_fn=ds.collate, num_workers=2)
    out, N = [], max(s.question.n for s in ds.samples)
    for b in dl:
        b = {k: v.to(model.device) for k, v in b.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(b, qtype)
        p = torch.softmax(lg, -1)
        pad = torch.zeros((p.shape[0], N), device=p.device)
        pad[:, : p.shape[1]] = p
        out.append(pad.cpu())
    return torch.cat(out).numpy()


def evaluate(probs, labels, teacher=None, n_classes=None):
    m = all_metrics(probs, labels, n_classes, teacher)
    return m


# ----------------------------------------------------------------------------- main
def run(a) -> dict:
    t_start = time.time()
    torch.manual_seed(a.seed)
    device = "cuda"
    cfg = DecisionModelConfig(model_name=a.model, pooling=a.pooling, head=a.head, head_dim=a.head_dim)
    model = DecisionModel(cfg, device)
    tasks = a.task.split(",")                     # multi-task: --task a,b,c  (one shared head)
    splits, per_task = {}, {}
    for split, lim in [("train", a.limit), ("valid", a.valid_limit), ("test", a.test_limit)]:
        soft = a.soft if split != "test" else (a.soft if a.teacher_test else None)
        per_task[split] = {t: load_split(t, split, soft, lim) for t in tasks}
        samples = [s for t in tasks for s in per_task[split][t]]
        splits[split] = DecisionDataset(samples, model.dtok, target=a.target)
    types = {s.question.type for s in splits["train"].samples}
    qtype = types.pop() if len(types) == 1 else "choice"
    if len(tasks) > 1:
        assert a.head != "linear", "multi-task training needs an option-embedding head (embed / mlp)"
    n_classes = max(s.question.n for s in splits["train"].samples) if a.fixed_classes else None
    fixed = len({tuple(s.question.ids) for s in splits["train"].samples}) == 1
    print(f"task={a.task} type={qtype} fixed_options={fixed} train={len(splits['train'])} valid={len(splits['valid'])} test={len(splits['test'])}")

    result = {"experiment": a.exp, "model": a.model, "pooling": a.pooling, "head": a.head, "loss": a.loss,
              "mode": a.mode, "target": a.target, "soft_labels": a.soft, "quantization": "bf16", "dataset": a.task,
              "n_train": len(splits["train"]), "epochs": a.epochs, "lr": a.lr, "gpu": torch.cuda.get_device_name(0)}

    # ---------------------------------------------------------------- Step 1: head-only on cached features
    if a.mode == "head" and a.pooling != "attn":
        def multi_features(split):
            soft = a.soft if split != "test" else (a.soft if a.teacher_test else None)
            lim = getattr(a, "limit" if split == "train" else f"{split}_limit")
            fs = [cached_features(model, DecisionDataset(per_task[split][t], model.dtok, a.target), t, split, soft, lim, a.feat_bs, a.target)
                  for t in tasks]
            N = max(f["opt_mask"].shape[1] for f in fs)
            out = {}
            for k in ("pooled", "opt_vecs", "opt_mask", "targets", "labels"):
                parts = []
                for f in fs:
                    x = f[k]
                    if k in ("opt_vecs", "opt_mask", "targets") and x.shape[1] < N:
                        pad = list(x.shape); pad[1] = N - x.shape[1]
                        x = torch.cat([x, torch.zeros(pad, dtype=x.dtype)], 1)
                    parts.append(x)
                out[k] = torch.cat(parts)
            return out

        feats = {s: multi_features(s) for s in splits}
        params = model.trainable_head_parameters()
        opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd)
        n = len(feats["train"]["labels"])
        steps = a.epochs * math.ceil(n / a.bs)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
        step = 0
        for ep in range(a.epochs):
            perm = torch.randperm(n)
            tot = 0.0
            for i in range(0, n, a.bs):
                idx = perm[i : i + a.bs]
                lg = head_logits_from_cache(model, feats["train"], idx, qtype, device)
                loss = compute_loss(a.loss, lg, feats["train"]["targets"][idx].to(device), feats["train"]["opt_mask"][idx].to(device))
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
                tot += loss.item() * len(idx); step += 1
            pv = predict_cached(model, feats["valid"], qtype, device)
            mv = evaluate(pv, feats["valid"]["labels"].numpy())
            print(f"ep {ep+1}/{a.epochs} loss={tot/n:.4f} valid acc={mv['accuracy']:.4f} ece={mv['ece']:.4f} nll={mv['nll']:.4f}")
        probs_valid = predict_cached(model, feats["valid"], qtype, device)
        probs_test = predict_cached(model, feats["test"], qtype, device)
        y_valid, y_test = feats["valid"]["labels"].numpy(), feats["test"]["labels"].numpy()
        teacher_test = feats["test"]["targets"].numpy() if a.teacher_test else None
    else:
        # ------------------------------------------------------------ Step 2-4: live training (lora / topN / full / attn-pooling)
        trainable = list(model.trainable_head_parameters())
        if a.mode == "lora":
            from peft import LoraConfig, get_peft_model
            lcfg = LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.05, bias="none",
                              target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
            model.backbone = get_peft_model(model.backbone, lcfg)
            trainable += [p for p in model.backbone.parameters() if p.requires_grad]
        elif a.mode.startswith("top"):
            k = int(a.mode[3:])
            for p in model.backbone.parameters():
                p.requires_grad_(False)
            for layer in model.backbone.layers[-k:]:
                layer.float()
                for p in layer.parameters():
                    p.requires_grad_(True); trainable.append(p)
            model.backbone.norm.float()
            trainable += list(model.backbone.norm.parameters())
        elif a.mode == "full":
            model.backbone.float()
            for p in model.backbone.parameters():
                p.requires_grad_(True)
            trainable += list(model.backbone.parameters())
        else:  # head with attn pooling: backbone frozen
            for p in model.backbone.parameters():
                p.requires_grad_(False)
        if a.pooling == "decision" and a.mode != "head":
            emb = model.backbone.get_input_embeddings()
            emb.weight.requires_grad_(True)
        n_tr = sum(p.numel() for p in trainable)
        print(f"trainable params: {n_tr/1e6:.2f}M")
        opt = torch.optim.AdamW(trainable, lr=a.lr, weight_decay=a.wd)
        ds = splits["train"]
        dl = DataLoader(ds, batch_size=a.bs, shuffle=True, collate_fn=ds.collate, num_workers=2, drop_last=False)
        steps = a.epochs * len(dl)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
        model.train(); model.backbone.train()
        for ep in range(a.epochs):
            tot, cnt = 0.0, 0
            for b in tqdm(dl, desc=f"epoch {ep+1}", mininterval=5):
                b = {k: v.to(device) for k, v in b.items()}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg = model(b, qtype)
                loss = compute_loss(a.loss, lg, b["targets"], b["option_mask"])
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step(); sched.step()
                tot += loss.item() * len(b["labels"]); cnt += len(b["labels"])
            model.eval(); model.backbone.eval()
            pv = predict_live(model, splits["valid"], qtype, a.eval_bs)
            yv = np.array([s.label for s in splits["valid"].samples])
            mv = evaluate(pv, yv)
            print(f"ep {ep+1}/{a.epochs} loss={tot/cnt:.4f} valid acc={mv['accuracy']:.4f} ece={mv['ece']:.4f} nll={mv['nll']:.4f}")
            model.train(); model.backbone.train()
        model.eval(); model.backbone.eval()
        probs_valid = predict_live(model, splits["valid"], qtype, a.eval_bs)
        probs_test = predict_live(model, splits["test"], qtype, a.eval_bs)
        y_valid = np.array([s.label for s in splits["valid"].samples])
        y_test = np.array([s.label for s in splits["test"].samples])
        teacher_test = None
        if a.teacher_test:
            N = probs_test.shape[1]
            teacher_test = np.array([s.targets() + [0.0] * (N - s.question.n) for s in splits["test"].samples])

    # ---------------------------------------------------------------- metrics + temperature scaling (TASK-010)
    result["valid"] = evaluate(probs_valid, y_valid)
    result["test"] = evaluate(probs_test, y_test, teacher_test)
    if len(tasks) > 1:
        result["per_task"] = {}
        off = 0
        for t in tasks:
            n_t = len(per_task["test"][t])
            result["per_task"][t] = evaluate(probs_test[off : off + n_t], y_test[off : off + n_t])
            off += n_t
    ts = TemperatureScaler().fit(probs_valid, y_valid)
    model.temperature = ts.T
    probs_test_cal = ts.transform(probs_test)
    result["temperature"] = ts.T
    result["test_calibrated"] = evaluate(probs_test_cal, y_test, teacher_test)
    result["reliability_test"] = reliability_table(probs_test, y_test)
    result["reliability_test_calibrated"] = reliability_table(probs_test_cal, y_test)
    for k in ("accuracy", "ece", "brier", "nll", "macro_f1"):
        result[k] = result["test"][k]
    result["ece_calibrated"] = result["test_calibrated"]["ece"]
    if qtype == "score":
        lv = np.arange(probs_test.shape[1])
        result["score_mae"] = float(np.abs((probs_test * lv).sum(1) - y_test).mean())

    # ---------------------------------------------------------------- single-question latency (B=1, naive)
    from jev.lm.backends.model_backend import ModelBackend
    be = ModelBackend(model, mode="naive")
    lat = []
    test_samples = splits["test"].samples[:100]
    for s in test_samples[:10]:
        be.decide(s.state, [s.question])
    for s in test_samples:
        lat.append(be.decide(s.state, [s.question])[0].latency_ms)
    result.update(latency_summary(lat))
    result["train_seconds"] = round(time.time() - t_start, 1)
    result["gpu_mem_alloc_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)

    # ---------------------------------------------------------------- save
    ck = CKPT_DIR / a.exp
    ck.mkdir(parents=True, exist_ok=True)
    if a.mode == "lora":
        model.backbone.save_pretrained(str(ck))
    model.save_head(str(ck))
    np.save(ck / "probs_test.npy", probs_test)
    np.save(ck / "probs_valid.npy", probs_valid)
    (ROOT / "experiments").mkdir(exist_ok=True)
    json.dump(result, open(ROOT / "experiments" / f"{a.exp}.json", "w"), indent=2, ensure_ascii=False)
    print(json.dumps({k: result[k] for k in ("experiment", "accuracy", "macro_f1", "ece", "ece_calibrated", "brier", "nll",
                                             "temperature", "latency_p50_ms", "latency_p95_ms", "train_seconds")}, indent=None))
    if "per_task" in result:
        for t, m in result["per_task"].items():
            print(f"  {t:28s} acc={m['accuracy']:.4f} f1={m['macro_f1']:.4f} ece={m['ece']:.4f} brier={m['brier']:.4f}")
    return result


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--soft", default=None, help="teacher dir name under data/store/generated/<task>/ e.g. qwen3.8-27b_logprob")
    ap.add_argument("--target", default="soft", choices=["soft", "hard", "mix"])
    ap.add_argument("--teacher_test", action="store_true", help="test split also from the teacher dir (for KL-to-teacher)")
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--pooling", default="last")
    ap.add_argument("--head", default="embed")
    ap.add_argument("--head_dim", type=int, default=512)
    ap.add_argument("--loss", default="kl")
    ap.add_argument("--mode", default="head")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--feat_bs", type=int, default=64)
    ap.add_argument("--eval_bs", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--valid_limit", type=int, default=None)
    ap.add_argument("--test_limit", type=int, default=None)
    ap.add_argument("--fixed_classes", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    return ap


if __name__ == "__main__":
    run(build_parser().parse_args())
