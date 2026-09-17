"""OCR 的タスクの検証: 「開いた転記」と「閉じた選択」で挙動がどう違うか。

(1) per-digit: 「左から N 桁目は何か」を Choice(0-9) で聞く  -> 位置指定が難しく当たらない
(2) closed-set: 「登録済み M 台のどれか」を Choice で聞く    -> 高精度。選択肢外は none で棄却される
(3) 劣化耐性: ぼかし・縮小を強めたとき、誤答するか棄却するか

usage: HF_HOME=... CUDA_VISIBLE_DEVICES=5 python scripts/ocr_probe.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from transformers import AutoModelForImageTextToText, AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
FLEET = ["12-34", "56-78", "90-12", "34-56", "78-90"]
NONE = "none of these"


def _font(size: int):
    f = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    if not os.path.exists(f):
        f = glob.glob("/usr/share/fonts/**/*.ttf", recursive=True)[0]
    return ImageFont.truetype(f, size)


def plate(num: str, blur: float = 0, shrink: float = 1.0) -> Image.Image:
    W, H = 440, 220
    im = Image.new("RGB", (W, H), (250, 250, 245))
    d = ImageDraw.Draw(im)
    d.rectangle([6, 6, W - 6, H - 6], outline=(20, 60, 140), width=6)
    d.text((110, 18), "500", font=_font(48), fill=(20, 60, 140))
    d.text((30, 90), num, font=_font(96), fill=(20, 60, 140))
    if blur:
        im = im.filter(ImageFilter.GaussianBlur(blur))
    if shrink != 1.0:
        im = im.resize((max(4, int(W * shrink)), max(2, int(H * shrink))), Image.BILINEAR).resize((W, H), Image.BILINEAR)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    a = ap.parse_args()
    proc = AutoProcessor.from_pretrained(a.model)
    tok = proc.tokenizer
    m = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16).to("cuda").eval()

    def ask(img, question, options):
        msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                             {"type": "text", "text": "State:\nA vehicle number plate photo.\n"}]}]
        inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to("cuda")
        labs = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[: len(options)]
        q = f"Question: {question}\nOptions:\n" + "".join(f"{l}. {o}\n" for l, o in zip(labs, options)) + "Decision:"
        e = tok.encode(q, add_special_tokens=False)
        ids = torch.cat([inp["input_ids"], torch.tensor([e], device="cuda")], 1)
        kw = {k: v for k, v in inp.items() if k not in ("input_ids", "attention_mask", "mm_token_type_ids")}
        with torch.no_grad():
            o = m(input_ids=ids, attention_mask=torch.ones_like(ids),
                  mm_token_type_ids=torch.cat([inp["mm_token_type_ids"], torch.zeros(1, len(e), dtype=torch.long, device="cuda")], 1), **kw)
        li = [tok.encode(" " + c, add_special_tokens=False)[0] for c in labs]
        p = torch.softmax(o.logits[0, -1, li].float(), -1)
        b = int(p.argmax())
        return options[b], float(p[b])

    out = {"experiment": "exp017_ocr_probe", "model": a.model}

    DIG = [str(i) for i in range(10)]
    per_digit = []
    print("(1) 開いた転記: '12-34' を 1 桁ずつ Choice(0-9) で聞く")
    for label, kw in [("鮮明", {}), ("ぼかし1.5px", {"blur": 1.5}), ("ぼかし3px", {"blur": 3.0})]:
        img = plate("12-34", **kw)
        row = []
        for i, ordinal in enumerate(["1st", "2nd", "3rd", "4th"]):
            pred, conf = ask(img, f"What is the {ordinal} digit of the large 4-digit number at the bottom?", DIG)
            row.append({"position": i + 1, "pred": pred, "conf": round(conf, 3), "truth": "1234"[i]})
        n_ok = sum(r["pred"] == r["truth"] for r in row)
        per_digit.append({"condition": label, "correct": n_ok, "digits": row})
        print(f"  {label:12s} -> " + " ".join(f"{r['pred']}({r['conf']:.2f})" for r in row) + f"   正解 1 2 3 4  ({n_ok}/4)")
    out["per_digit"] = per_digit

    closed = []
    print("\n(2) 閉じた選択: 登録 5 台のどれか (選択肢に none あり)")
    opts = FLEET + [NONE]
    for truth in FLEET:
        pred, conf = ask(plate(truth), "Which registered vehicle is this plate?", opts)
        closed.append({"truth": truth, "pred": pred, "conf": round(conf, 3)})
        print(f"  正解 {truth} -> {pred:14s} ({conf:.3f}) {'OK' if pred == truth else 'NG'}")
    for truth in ["11-11"]:
        pred, conf = ask(plate(truth), "Which registered vehicle is this plate?", opts)
        closed.append({"truth": truth + " (登録外)", "pred": pred, "conf": round(conf, 3)})
        print(f"  登録外 {truth} -> {pred:14s} ({conf:.3f}) {'OK(棄却)' if pred == NONE else 'NG'}")
        pred2, conf2 = ask(plate(truth), "Which registered vehicle is this plate?", FLEET)
        print(f"     none 無しだと -> {pred2} ({conf2:.3f})  ※ 確信度が落ちる")
    out["closed_set"] = closed

    deg = []
    print("\n(3) 劣化を強めたとき (正解 56-78)")
    for label, kw in [("鮮明", {}), ("ぼかし8px", {"blur": 8}), ("ぼかし20px", {"blur": 20}),
                      ("1/8縮小", {"shrink": 1 / 8}), ("1/16縮小", {"shrink": 1 / 16}), ("1/32縮小", {"shrink": 1 / 32})]:
        pred, conf = ask(plate("56-78", **kw), "Which registered vehicle is this plate?", opts)
        deg.append({"condition": label, "pred": pred, "conf": round(conf, 3)})
        print(f"  {label:12s} -> {pred:14s} ({conf:.3f}) {'OK' if pred == '56-78' else '棄却' if pred == NONE else 'NG'}")
    out["degradation"] = deg
    json.dump(out, open(ROOT / "experiments" / "exp017_ocr_probe.json", "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
