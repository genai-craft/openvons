"""各構成の VRAM とレイテンシを同条件で実測する (画像 1 枚 + 4 質問)。"""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/data/decision_model/data/vision/pa100k")


def nvsmi_mb() -> int:
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    import os
    me = str(os.getpid())
    for line in out.strip().splitlines():
        pid, mem = [x.strip() for x in line.split(",")]
        if pid == me:
            return int(mem)
    return 0


def timed(fn, reps=20):
    for _ in range(5):
        fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(reps):
        torch.cuda.synchronize(); t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort(); return ts[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["vision", "vlm", "zeroshot"])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--label", required=True)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(DATA / "test.jsonl")][:40]
    imgs = [Image.open(DATA / r["image"]).convert("RGB") for r in rows]
    tasks = json.load(open(DATA / "tasks.json"))["questions"]
    from openvons.core.primitives import Question
    qs = [Question.from_dict(v, key=k) for k, v in tasks.items()]
    i = 0

    if a.kind == "vision":
        from openvons.vision.vision_model import VisionDecisionModel
        m = VisionDecisionModel.from_checkpoint(a.ckpt)
        params = m.n_frozen() + m.n_trainable()
        run = lambda: m.decide([imgs[0]])
    elif a.kind == "vlm":
        from openvons.vision.vlm_decision_model import VLMDecisionModel
        m = VLMDecisionModel.from_checkpoint(a.ckpt)
        params = sum(p.numel() for p in m.backbone.parameters()) + sum(p.numel() for p in m.head.parameters())
        run = lambda: m.decide(imgs[0], qs)
    else:
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from openvons.core.formats import option_labels
        proc = AutoProcessor.from_pretrained(a.model); tok = proc.tokenizer
        mm = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16).to("cuda").eval()
        params = sum(p.numel() for p in mm.parameters())

        def run():
            for q in qs:
                msgs = [{"role": "user", "content": [{"type": "image", "image": imgs[0]},
                                                     {"type": "text", "text": "State:\nA photo of a person.\n"}]}]
                inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to("cuda")
                labs = option_labels(q.n)
                txt = f"Question: {q.text}\nOptions:\n" + "".join(f"{l}. {o.label()}\n" for l, o in zip(labs, q.options)) + "Decision:"
                e = tok.encode(txt, add_special_tokens=False)
                ids = torch.cat([inp["input_ids"], torch.tensor([e], device="cuda")], 1)
                kw = {k: v for k, v in inp.items() if k not in ("input_ids", "attention_mask", "mm_token_type_ids")}
                ex = {"mm_token_type_ids": torch.cat([inp["mm_token_type_ids"], torch.zeros(1, len(e), dtype=torch.long, device="cuda")], 1)} if "mm_token_type_ids" in inp else {}
                with torch.no_grad():
                    mm(input_ids=ids, attention_mask=torch.ones_like(ids), **ex, **kw)

    lat = timed(run)
    torch.cuda.synchronize()
    res = {"label": a.label, "kind": a.kind, "params_M": round(params / 1e6, 1),
           "weights_GB": round(params * 2 / 2**30, 2),
           "torch_peak_GB": round(torch.cuda.max_memory_allocated() / 2**30, 2),
           "process_vram_GB": round(nvsmi_mb() / 1024, 2),
           "latency_4q_ms": round(lat, 1)}
    print(json.dumps(res, ensure_ascii=False))
    out = ROOT / "experiments" / "exp022_vram.jsonl"
    with open(out, "a") as f:
        f.write(json.dumps(res, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
