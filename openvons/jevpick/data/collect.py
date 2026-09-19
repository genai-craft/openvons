"""Phase 1: target LLM の greedy continuation を記録する (§13 Phase 1 / §10.2)。

出力: {DATA}/traces_<model_tag>.jsonl
  {sample_id, domain, prompt_ids, output_ids, finished}
hidden state はこの段階では保存しない (oracle study に不要)。
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
from openvons.jevpick.paths import DATA  # noqa: E402

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--prompts", default=f"{DATA}/prompts.jsonl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--domain", default="", help="この domain だけ収集")
    ap.add_argument("--quant", default="", help="fp8: FineGrainedFP8 で on-the-fly 量子化")
    ap.add_argument("--split", default="", help="この split だけ収集")
    ap.add_argument("--no-think", action="store_true", help="Qwen3 の enable_thinking=False")
    args = ap.parse_args()

    tag = args.model.split("/")[-1]
    out = Path(args.out or f"{DATA}/traces_{tag}.jsonl")
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if args.quant == "fp8":
        from transformers import FineGrainedFP8Config
        qc = FineGrainedFP8Config(modules_to_not_convert=["in_proj_a", "in_proj_b", "lm_head", "conv1d", "norm"])
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa", quantization_config=qc, device_map=args.device).eval()
    elif args.quant == "nf4":
        from transformers import BitsAndBytesConfig
        qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
                                llm_int8_skip_modules=["lm_head", "in_proj_a", "in_proj_b"])
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa", quantization_config=qc, device_map=args.device).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(args.device).eval()

    rows = [json.loads(l) for l in open(args.prompts)]
    if args.domain:
        rows = [r for r in rows if r["domain"] in args.domain.split(",")]
    if args.split:
        rows = [r for r in rows if r.get("split") == args.split]
    if args.limit:
        rows = rows[: args.limit]
    # 長さでソートして padding を減らす
    texts = []
    for r in rows:
        kw = {"tools": r["tools"]} if r["tools"] else {}
        if args.no_think:
            kw["enable_thinking"] = False
        texts.append(tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True, **kw))
    order = sorted(range(len(rows)), key=lambda i: len(texts[i]))

    done = set()
    if out.exists():
        done = {json.loads(l)["sample_id"] for l in open(out)}
    f = out.open("a")
    t0 = time.time()
    n = 0
    for s in range(0, len(order), args.batch):
        idx = [i for i in order[s : s + args.batch] if rows[i]["sample_id"] not in done]
        if not idx:
            continue
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(args.device)
        with torch.no_grad():
            gen = model.generate(
                **enc, max_new_tokens=args.max_new, do_sample=False, temperature=None, top_p=None, top_k=None,
                pad_token_id=tok.pad_token_id,
            )
        plen = enc["input_ids"].shape[1]
        for j, i in enumerate(idx):
            prompt_ids = enc["input_ids"][j][enc["attention_mask"][j].bool()].tolist()
            out_ids = gen[j, plen:].tolist()
            # pad / eos 以降を落とす (eos 自体は残す)
            cut = len(out_ids)
            for k, t in enumerate(out_ids):
                if t == tok.pad_token_id or t in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")):
                    cut = k + 1 if t != tok.pad_token_id else k
                    break
            out_ids = out_ids[:cut]
            f.write(json.dumps({"sample_id": rows[i]["sample_id"], "domain": rows[i]["domain"],
                                "prompt_ids": prompt_ids, "output_ids": out_ids,
                                "finished": cut < args.max_new}) + "\n")
            n += 1
        f.flush()
        print(f"{n}/{len(rows)}  {time.time()-t0:.0f}s", flush=True)
    print("peak VRAM %.1f GB" % (torch.cuda.max_memory_allocated() / 1e9))
    print("->", out)


if __name__ == "__main__":
    main()
