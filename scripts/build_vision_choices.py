"""カメラの「状態」を当てる質問セットを作る (SigLIP2 のテキスト側をサーバーで先に計算しておく).

    .venv/bin/python scripts/build_vision_choices.py --set all --out /data/openjev/models/ondevice/vision-choices

端末には**画像エンコーダだけ**を置き、選択肢のテキスト埋め込みはこのスクリプトが JSON にして配る。
質問を足したり言い換えたりしても、アプリの更新は要らない (JSON を配り直すだけ)。

検出 (物体の箱) ではなく、場面の**状態**を有限の選択肢で答えさせるのが狙い。
質問の中身は scripts/vision_question_sets.py にある。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vision_question_sets import QUESTION_SETS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/siglip2-base-patch16-224")
    ap.add_argument("--out", default="/data/openjev/models/ondevice/vision-choices")
    ap.add_argument("--set", default="all", choices=["all", *sorted(QUESTION_SETS)])
    args = ap.parse_args()
    from transformers import AutoModel, AutoProcessor
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    model = AutoModel.from_pretrained(args.model).eval()
    proc = AutoProcessor.from_pretrained(args.model)
    keys = sorted(QUESTION_SETS) if args.set == "all" else [args.set]
    for key in keys:
        build_one(key, model, proc, out, args.model)


def build_one(set_key: str, model, proc, out: Path, model_name: str) -> None:
    title, questions = QUESTION_SETS[set_key]
    # 1 つの選択肢に複数の言い方を持たせ、正規化した埋め込みの平均を使う (prompt ensembling)。
    # 夜の監視カメラ画像で「乱れ・ノイズ」に倒れるなど、言い方 1 つだと外す場面が減る。
    prompts, index = [], []
    for q in questions:
        for c in q["choices"]:
            for pr in (c.get("prompts") or [c["prompt"]]):
                index.append((q["key"], c["id"])); prompts.append(pr)
    with torch.no_grad():
        tin = proc(text=prompts, padding="max_length", max_length=64, return_tensors="pt")
        emb = model.get_text_features(**tin)
        emb = emb.pooler_output if hasattr(emb, "pooler_output") else (emb.last_hidden_state if hasattr(emb, "last_hidden_state") else emb)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    acc: dict[str, dict[str, list[list[float]]]] = {}
    for (qk, cid), v in zip(index, emb.tolist()):
        acc.setdefault(qk, {}).setdefault(cid, []).append(v)
    by_key: dict[str, dict[str, list[float]]] = {}
    for qk, per_choice in acc.items():
        for cid, vs in per_choice.items():
            t = torch.tensor(vs).mean(0)
            t = t / t.norm()
            by_key.setdefault(qk, {})[cid] = [round(x, 5) for x in t.tolist()]
    doc = {"model": model_name, "set": set_key, "title": title, "logit_scale": float(model.logit_scale.exp()), "logit_bias": float(getattr(model, "logit_bias", torch.tensor(0.0))),
           "dim": emb.shape[1], "questions": [{**q, "choices": [{k: v for k, v in c.items() if k != "prompts"} for c in q["choices"]],
                          "embeddings": by_key[q["key"]]} for q in questions]}   # requires はそのまま配る
    name = "choices.json" if set_key == "general" else f"choices_{set_key}.json"
    (out / name).write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    print(f"[{set_key}] questions {len(questions)}  choices {len(prompts)}  dim {emb.shape[1]}  -> {out/name}")


if __name__ == "__main__":
    main()
