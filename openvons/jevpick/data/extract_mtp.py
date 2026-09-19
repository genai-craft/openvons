"""Qwen3.5 系 MTP head の draft 候補を各 decode 位置で保存する (extract.py の dflash と同じ layout)。
出力 dir: mtp_tok.npy (total, K, 4), mtp_lp.npy, index.jsonl, mtp_stats.json"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from openvons.jevpick.candidates.mtp_qwen35 import Qwen35MTP  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--traces", required=True)
    ap.add_argument("--prompts", default="/data/openvons/jevpick/prompts_v2.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--K", type=int, default=7)
    ap.add_argument("--quant", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="0/1")
    args = ap.parse_args()
    dev = "cuda:0"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    mdir = snapshot_download(args.model, allow_patterns=["*.json"])
    if args.quant == "fp8":
        from transformers import FineGrainedFP8Config
        qc = FineGrainedFP8Config(modules_to_not_convert=["in_proj_a", "in_proj_b", "lm_head", "conv1d", "norm"])
        target = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa", quantization_config=qc, device_map=dev).eval()
    else:
        target = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
    mtp = Qwen35MTP(target, snapshot_download(args.model, allow_patterns=["*.json"])).to(dev).eval()
    print("MTP loaded, params %.2fB" % (sum(p.numel() for n, p in mtp.named_parameters() if not n.startswith(("embed", "lm_head"))) / 1e9), flush=True)
    split_of = {json.loads(l)["sample_id"]: json.loads(l).get("split", "test") for l in open(args.prompts)}
    traces = [json.loads(l) for l in open(args.traces)]
    if args.limit:
        traces = traces[: args.limit]
    si_, sn_ = map(int, args.shard.split("/"))
    traces = traces[si_::sn_]
    total = sum(len(t["output_ids"]) for t in traces)
    K = args.K
    mtok = np.lib.format.open_memmap(out / "mtp_tok.npy", mode="w+", dtype=np.int32, shape=(total, K, 4))
    mlp = np.lib.format.open_memmap(out / "mtp_lp.npy", mode="w+", dtype=np.float16, shape=(total, K, 4))
    idx_f = (out / "index.jsonl").open("w")
    off = 0
    t0 = time.time()
    hit1 = hitchain = npos = 0
    with torch.inference_mode():
        for si, s in enumerate(traces):
            prompt, outp = s["prompt_ids"], s["output_ids"]
            P, N = len(prompt), len(outp)
            full = prompt + outp
            o = target(input_ids=torch.tensor([full], device=dev), output_hidden_states=True, use_cache=False)
            hid = o.hidden_states[-1]  # (1, T, H) post-norm
            cache = mtp.new_cache()
            mtp.prefill(hid, full, P - 2, cache)  # index 1..P-2
            for t in range(N):
                s_ = P + t - 1
                toks, lps = mtp.draft(hid[:, s_ - 1 : s_], full[s_], s_, K, cache)
                mtok[off + t] = toks.cpu().numpy()
                mlp[off + t] = lps.to(torch.float16).cpu().numpy()
                chain = toks[:, 0].tolist(); truth = outp[t : t + K]
                m = 0
                for a, b in zip(chain, truth):
                    if a != b:
                        break
                    m += 1
                hit1 += int(m >= 1); hitchain += m; npos += 1
            idx_f.write(json.dumps({"sample_id": s["sample_id"], "domain": s["domain"], "split": split_of.get(s["sample_id"], "test"), "offset": off, "N": N, "P": P}) + "\n")
            off += N
            if (si + 1) % 50 == 0:
                print(f"{si+1}/{len(traces)} pos={off} {time.time()-t0:.0f}s  step1 acc {hit1/npos:.3f} mean chain match {hitchain/npos:.2f}", flush=True)
    mtok.flush(); mlp.flush(); idx_f.close()
    json.dump({"K": K, "total": total, "step1_acc": hit1 / max(npos, 1), "mean_chain_match": hitchain / max(npos, 1)}, open(out / "mtp_stats.json", "w"))
    print("->", out, "step1 acc %.3f mean chain match %.2f" % (hit1 / max(npos, 1), hitchain / max(npos, 1)))


if __name__ == "__main__":
    main()
