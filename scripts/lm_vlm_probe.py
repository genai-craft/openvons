"""§17 VLM Decision Model の feasibility 検証。

(1) 画像に対して選択肢の確率を出せるか (LM head 経由の zero-shot)
(2) 1 フレームに複数の判断を出すとき、画像エンコードの共有でどれだけ速くなるか (端から端まで)

usage: HF_HOME=... CUDA_VISIBLE_DEVICES=4 python scripts/vlm_probe.py [--model Qwen/Qwen3-VL-4B-Instruct]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForImageTextToText, AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
QS = ["Is a person lying on the floor?", "Is anyone present?", "Is there an intruder?", "Should staff be alerted?",
      "Is the room dark?", "Is the person moving?", "Is an object blocking the exit?", "Is the scene normal?"] * 2


def qtext(q: str, options=("yes", "no")) -> str:
    labs = "ABCDEFG"[: len(options)]
    return f"Question: {q}\nOptions:\n" + "".join(f"{l}. {o}\n" for l, o in zip(labs, options)) + "Decision:"


def shapes(color: str, shape: str, size=(448, 448)) -> Image.Image:
    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)
    (d.ellipse if shape == "circle" else d.rectangle)([120, 120, 330, 330], fill=color)
    return img


def room(size=(1280, 720)) -> Image.Image:
    img = Image.new("RGB", size, (40, 44, 52))
    d = ImageDraw.Draw(img)
    d.rectangle([0, int(size[1] * 0.7), size[0], size[1]], fill=(90, 90, 96))
    d.ellipse([500, 520, 760, 620], fill=(200, 170, 140))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--counts", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--reps", type=int, default=8)
    a = ap.parse_args()
    proc = AutoProcessor.from_pretrained(a.model)
    tok = proc.tokenizer
    m = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16).to("cuda").eval()

    def state(img, text="State:\nLive frame from a room camera.\n"):
        msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": text}]}]
        return proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to("cuda")

    # ---- (1) zero-shot 判断 -------------------------------------------------
    zs = {}
    for color, shape in [("red", "circle"), ("blue", "square")]:
        inp = state(shapes(color, shape), "State:\nA photo from a sensor.\n")
        row = {}
        for q, opts in [("What shape is the object?", ("a circle", "a square", "a triangle")),
                        ("What color is the object?", ("red", "blue", "green"))]:
            e = tok.encode(qtext(q, opts), add_special_tokens=False)
            ids = torch.cat([inp["input_ids"], torch.tensor([e], device="cuda")], 1)
            with torch.no_grad():
                o = m(input_ids=ids, attention_mask=torch.ones_like(ids),
                      mm_token_type_ids=torch.cat([inp["mm_token_type_ids"], torch.zeros(1, len(e), dtype=torch.long, device="cuda")], 1),
                      pixel_values=inp["pixel_values"], image_grid_thw=inp["image_grid_thw"])
            lab_ids = [tok.encode(" " + c, add_special_tokens=False)[0] for c in "ABCDEFG"[: len(opts)]]
            p = torch.softmax(o.logits[0, -1, lab_ids].float(), -1)
            row[q] = {opt: round(float(x), 3) for opt, x in zip(opts, p)}
        zs[f"{color}_{shape}"] = row
        print(f"{color} {shape}: {json.dumps(row)}")

    # ---- (2) 画像共有による multi-question スケーリング ---------------------
    img = room()
    S = state(img)["input_ids"].shape[1]
    n_img = int((state(img)["input_ids"][0] == tok.convert_tokens_to_ids("<|image_pad|>")).sum())
    print(f"\n1280x720 -> 画像トークン {n_img} / state {S} tokens")

    def pad(texts):
        enc = [tok.encode(t, add_special_tokens=False) for t in texts]
        L = max(len(e) for e in enc)
        return torch.tensor([e + [tok.pad_token_id] * (L - len(e)) for e in enc], device="cuda"), L

    @torch.no_grad()
    def naive(n):
        inp = state(img)
        for q in QS[:n]:
            e = tok.encode(qtext(q), add_special_tokens=False)
            ids = torch.cat([inp["input_ids"], torch.tensor([e], device="cuda")], 1)
            m.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                    mm_token_type_ids=torch.cat([inp["mm_token_type_ids"], torch.zeros(1, len(e), dtype=torch.long, device="cuda")], 1),
                    pixel_values=inp["pixel_values"], image_grid_thw=inp["image_grid_thw"])

    @torch.no_grad()
    def shared(n):
        inp = state(img)
        s = inp["input_ids"].shape[1]
        cache = m.model(**inp, use_cache=True).past_key_values
        cache.batch_repeat_interleave(n)
        qi, L = pad([qtext(q) for q in QS[:n]])
        m.model.language_model(input_ids=qi, attention_mask=torch.ones(n, s + L, dtype=torch.long, device="cuda"),
                               past_key_values=cache, cache_position=s + torch.arange(L, device="cuda"), use_cache=True)

    def timeit(fn):
        for _ in range(2):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(a.reps):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / a.reps * 1000

    rows = []
    print(f"{'質問数':>6} {'naive':>10} {'state共有':>10} {'倍率':>7}")
    for n in a.counts:
        t_n, t_s = timeit(lambda: naive(n)), timeit(lambda: shared(n))
        rows.append({"questions": n, "naive_ms": round(t_n, 1), "shared_ms": round(t_s, 1), "speedup": round(t_n / t_s, 2)})
        print(f"{n:6d} {t_n:9.1f}ms {t_s:9.1f}ms {t_n/t_s:6.1f}x")

    res = {"experiment": "exp014_vlm_feasibility", "model": a.model, "image": "1280x720 synthetic room",
           "image_tokens": n_img, "state_tokens": S, "zero_shot": zs, "scaling": rows,
           "gpu": torch.cuda.get_device_name(0)}
    json.dump(res, open(ROOT / "experiments" / "exp014_vlm_feasibility.json", "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
