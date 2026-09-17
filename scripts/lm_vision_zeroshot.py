"""画像タスクの zero-shot 基準線: VLM に選択肢を提示して LM head からラベルトークンの確率を読む。"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

from jev.core.metrics import all_metrics, reliability_table
from jev.core.formats import option_labels
from jev.core.primitives import Question

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--data", default="/data/decision_model/data/vision/fairface")
    ap.add_argument("--limit", type=int, default=2000)
    a = ap.parse_args()
    data = Path(a.data)
    tasks = json.load(open(data / "tasks.json"))["questions"]
    qs = [Question.from_dict(v, key=k) for k, v in tasks.items()]
    rows = [json.loads(l) for l in open(data / "test.jsonl") if l.strip()][: a.limit]
    proc = AutoProcessor.from_pretrained(a.model)
    tok = proc.tokenizer
    m = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16).to("cuda").eval()

    def ask(img, q):
        msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                             {"type": "text", "text": "State:\nA photo of a person.\n"}]}]
        inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to("cuda")
        labs = option_labels(q.n)
        txt = f"Question: {q.text}\nOptions:\n" + "".join(f"{l}. {o.label()}\n" for l, o in zip(labs, q.options)) + "Decision:"
        e = tok.encode(txt, add_special_tokens=False)
        ids = torch.cat([inp["input_ids"], torch.tensor([e], device="cuda")], 1)
        kw = {k: v for k, v in inp.items() if k not in ("input_ids", "attention_mask", "mm_token_type_ids")}
        ex = {"mm_token_type_ids": torch.cat([inp["mm_token_type_ids"], torch.zeros(1, len(e), dtype=torch.long, device="cuda")], 1)} if "mm_token_type_ids" in inp else {}
        with torch.no_grad():
            o = m(input_ids=ids, attention_mask=torch.ones_like(ids), **ex, **kw)
        li = [tok.encode(" " + c, add_special_tokens=False)[0] for c in labs]
        return torch.softmax(o.logits[0, -1, li].float(), -1).cpu().numpy()

    res = {"experiment": a.exp, "model": a.model, "kind": "zero_shot", "n": len(rows), "tasks": {}}
    lat = []
    for q in qs:
        P = np.zeros((len(rows), q.n))
        for i, r in enumerate(tqdm(rows, desc=f"{q.key}", mininterval=10)):
            img = Image.open(data / r["image"]).convert("RGB")
            t0 = time.perf_counter(); P[i] = ask(img, q); lat.append((time.perf_counter() - t0) * 1000)
        y = np.array([r["labels"][q.key] for r in rows])
        mm = all_metrics(P, y)
        res["tasks"][q.key] = {"n_options": q.n, "test": mm, "reliability": reliability_table(P, y)}
        print(f"  {q.key:8s} ({q.n}択) acc={mm['accuracy']:.4f} f1={mm['macro_f1']:.4f} ece={mm['ece']:.4f} brier={mm['brier']:.4f}")
    res["latency_p50_ms"] = float(np.percentile(lat, 50))
    json.dump(res, open(ROOT / "experiments" / f"{a.exp}.json", "w"), indent=2, ensure_ascii=False)
    print(f"1質問あたり p50 {res['latency_p50_ms']:.1f}ms")


if __name__ == "__main__":
    main()
