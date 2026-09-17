"""動画クリップに対する Decision Model の feasibility と、フレームサンプリングの落とし穴の検証。

(1) frame-drop 検証: N 枚を 1 つの video として投げたとき、一瞬だけ映る異常を検出できるか
(2) 複数質問のスケーリング: 同じクリップに N 個の判断を出す端から端までの時間

usage: HF_HOME=... CUDA_VISIBLE_DEVICES=4 python scripts/video_probe.py
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForImageTextToText, AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
QL = ["Is a person lying on the floor?", "Is anyone present?", "Is there an intruder?", "Should staff be alerted?",
      "Is the room dark?", "Is the person moving?", "Is an object blocking the exit?", "Is the scene normal?"] * 2
Q_FLASH = "Question: Is there a large red square anywhere in this clip?\nOptions:\nA. yes\nB. no\nDecision:"


def qtext(q: str) -> str:
    return f"Question: {q}\nOptions:\nA. yes\nB. no\nDecision:"


def clip(n: int, flash=(), size=(640, 360)):
    """通常は人が立っているだけ。flash に入れたフレーム番号でだけ大きな赤い四角 (異常) が映る。"""
    out = []
    w, h = size
    for i in range(n):
        im = Image.new("RGB", size, (40, 44, 52))
        d = ImageDraw.Draw(im)
        d.rectangle([0, int(h * 0.7), w, h], fill=(90, 90, 96))
        d.ellipse([w // 2 - 20, int(h * 0.42), w // 2 + 40, int(h * 0.72)], fill=(200, 170, 140))
        if i in flash:
            d.rectangle([40, 40, 240, 240], fill=(255, 0, 0))
        out.append(im)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--frames", type=int, nargs="+", default=[4, 8, 16, 32])
    ap.add_argument("--counts", type=int, nargs="+", default=[1, 4, 8, 16])
    ap.add_argument("--reps", type=int, default=15)
    a = ap.parse_args()
    proc = AutoProcessor.from_pretrained(a.model)
    tok = proc.tokenizer
    m = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16).to("cuda").eval()
    VID = tok.convert_tokens_to_ids("<|video_pad|>")

    def build(content):
        msgs = [{"role": "user", "content": content + [{"type": "text", "text": "State:\nSecurity camera clip.\n"}]}]
        return proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to("cuda")

    def forward(inp, q):
        e = tok.encode(q, add_special_tokens=False)
        ids = torch.cat([inp["input_ids"], torch.tensor([e], device="cuda")], 1)
        kw = {k: v for k, v in inp.items() if k not in ("input_ids", "attention_mask", "mm_token_type_ids")}
        with torch.no_grad():
            return m(input_ids=ids, attention_mask=torch.ones_like(ids),
                     mm_token_type_ids=torch.cat([inp["mm_token_type_ids"], torch.zeros(1, len(e), dtype=torch.long, device="cuda")], 1), **kw)

    def p_yes(content, q=Q_FLASH):
        inp = build(content)
        o = forward(inp, q)
        labs = [tok.encode(" " + c, add_special_tokens=False)[0] for c in "AB"]
        return float(torch.softmax(o.logits[0, -1, labs].float(), -1)[0]), int((inp["input_ids"][0] == VID).sum())

    # ---- (1) frame drop ----------------------------------------------------
    drop = []
    print("N 枚を 1 つの video として投げたとき、異常が 1 枚だけ映る場合の P(yes)")
    for n in a.frames:
        p1, tk = p_yes([{"type": "video", "video": clip(n, (n // 2,))}])
        pa, _ = p_yes([{"type": "video", "video": clip(n, tuple(range(n)))}])
        pi, _ = p_yes([{"type": "image", "image": f} for f in clip(n, (n // 2,))])
        drop.append({"frames": n, "video_tokens": tk, "video_one_frame": round(p1, 3),
                     "video_all_frames": round(pa, 3), "images_one_frame": round(pi, 3)})
        print(f"  {n:2d}枚 video (token {tk:3d}): 1枚だけ異常 P={p1:.3f} / 全枚異常 P={pa:.3f} | image ばら投げ P={pi:.3f}")

    # ---- (2) multi-question scaling ---------------------------------------
    fr = clip(16, (4, 5))
    prep = lambda: build([{"type": "video", "video": fr}])

    @torch.no_grad()
    def naive(n):
        inp = prep()
        for q in QL[:n]:
            forward(inp, qtext(q))

    @torch.no_grad()
    def shared(n):
        inp = prep()
        S = inp["input_ids"].shape[1]
        c = m.model(**inp, use_cache=True).past_key_values
        c.batch_repeat_interleave(n)
        enc = [tok.encode(qtext(q), add_special_tokens=False) for q in QL[:n]]
        L = max(len(e) for e in enc)
        qi = torch.tensor([e + [tok.pad_token_id] * (L - len(e)) for e in enc], device="cuda")
        m.model.language_model(input_ids=qi, attention_mask=torch.ones(n, S + L, dtype=torch.long, device="cuda"),
                               past_key_values=c, cache_position=S + torch.arange(L, device="cuda"), use_cache=True)

    def med(fn):
        for _ in range(5):
            fn()
        ts = []
        for _ in range(a.reps):
            torch.cuda.synchronize(); t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000)
        ts.sort()
        return ts[len(ts) // 2]

    scaling = []
    print(f"\n{'質問数':>6} {'naive':>10} {'クリップ共有':>12} {'倍率':>7}")
    for n in a.counts:
        t_n, t_s = med(lambda: naive(n)), med(lambda: shared(n))
        scaling.append({"questions": n, "naive_ms": round(t_n, 1), "shared_ms": round(t_s, 1), "speedup": round(t_n / t_s, 2)})
        print(f"{n:6d} {t_n:9.1f}ms {t_s:11.1f}ms {t_n/t_s:6.1f}x")

    json.dump({"experiment": "exp015_video_feasibility", "model": a.model, "frame_drop": drop, "scaling": scaling,
               "note": "transformers の既定 video 前処理 (video_metadata なし) での結果"},
              open(ROOT / "experiments" / "exp015_video_feasibility.json", "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
