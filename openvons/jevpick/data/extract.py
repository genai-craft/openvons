"""trace を teacher-forcing で再実行し、scorer 用 hidden state と DFlash draft 候補を各 decode 位置で保存する。

位置 t (t=0..N-1) の定義は oracle study と同じ: 文脈 = full[:P+t], 正解 = out[t:]。
その時点で使える状態は「x = full[P+t-1] が確定済みだが未 forward」なので、
  hidden[t]  = target hidden at index P+t-2 (x を生んだ位置)   ← scorer 入力
  dflash[t]  = anchor x と hidden[:P+t-1] から draft した block  ← Source F
出力 ({DATA}/<tag>/):
  hidden.f16      memmap (total_pos, n_layers, H)
  dflash_tok.i32  memmap (total_pos, 15, 4)   位置別 top-4 token
  dflash_lp.f16   memmap (total_pos, 15, 4)   その log prob
  index.jsonl     {sample_id, domain, split, offset, N, P}
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

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DynamicCache

LAYERS = None  # 実行時に 25/50/75/100% の層を決める (hidden_states のインデックス, 0=embedding)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--draft", default="z-lab/Qwen3-4B-DFlash-b16")
    ap.add_argument("--traces", required=True)
    ap.add_argument("--prompts", default=f"{DATA}/prompts_v2.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-dflash", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="0/1", help="i/n: traces を n 分割した i 番目だけ処理")
    ap.add_argument("--quant", default="", help="fp8: FineGrainedFP8 で on-the-fly 量子化して hidden を取る")
    args = ap.parse_args()
    dev = "cuda:0"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    if args.quant == "fp8":
        from transformers import FineGrainedFP8Config
        qc = FineGrainedFP8Config(modules_to_not_convert=["in_proj_a", "in_proj_b", "lm_head", "conv1d", "norm"])
        target = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa", quantization_config=qc, device_map=dev).eval()
    elif args.quant == "nf4":
        from transformers import BitsAndBytesConfig
        qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
                                llm_int8_skip_modules=["lm_head", "in_proj_a", "in_proj_b"])
        target = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa", quantization_config=qc, device_map=dev).eval()
    else:
        target = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
    tcfg = getattr(target.config, "text_config", None) or target.config
    Hd = tcfg.hidden_size
    nL = tcfg.num_hidden_layers
    global LAYERS
    LAYERS = tuple(round(nL * f) for f in (0.25, 0.5, 0.75, 1.0))
    print("layers", nL, "hidden", Hd, "saving hidden_states idx", LAYERS, flush=True)
    draft = None
    if not args.no_dflash:
        from dflash.model import DFlash2DraftModel, DFlashDraftModel, extract_context_feature
        dcfg = AutoConfig.from_pretrained(args.draft)
        dcls = DFlash2DraftModel if "DFlash2DraftModel" in (dcfg.architectures or []) else DFlashDraftModel
        draft = dcls.from_pretrained(args.draft, attn_implementation="sdpa", dtype=torch.bfloat16).to(dev).eval()
        print("draft", dcls.__name__, "block", draft.block_size, flush=True)
        B = draft.block_size
        mask_id = draft.mask_token_id
        emb_w = target.get_input_embeddings().weight
        lm_head = target.lm_head
    split_of = {json.loads(l)["sample_id"]: json.loads(l).get("split", "test") for l in open(args.prompts)}
    traces = [json.loads(l) for l in open(args.traces)]
    if args.limit:
        traces = traces[: args.limit]
    si_, sn_ = map(int, args.shard.split("/"))
    traces = traces[si_::sn_]
    total = sum(len(t["output_ids"]) for t in traces)
    hid = np.lib.format.open_memmap(out / "hidden.npy", mode="w+", dtype=np.float16, shape=(total, len(LAYERS), Hd))
    if draft is not None:
        dtok = np.lib.format.open_memmap(out / "dflash_tok.npy", mode="w+", dtype=np.int32, shape=(total, B - 1, 4))
        dlp = np.lib.format.open_memmap(out / "dflash_lp.npy", mode="w+", dtype=np.float16, shape=(total, B - 1, 4))
    idx_f = (out / "index.jsonl").open("w")
    off = 0
    t0 = time.time()
    with torch.inference_mode():
        for si, s in enumerate(traces):
            prompt, outp = s["prompt_ids"], s["output_ids"]
            P, N = len(prompt), len(outp)
            full = torch.tensor([prompt + outp], device=dev)
            o = target(input_ids=full, output_hidden_states=True, use_cache=False)
            hs = o.hidden_states  # tuple(37) of (1, T, H)
            sel = torch.stack([hs[l][0, P - 2 : P - 2 + N] for l in LAYERS], dim=1)  # (N, L, H)
            hid[off : off + N] = sel.to(torch.float16).cpu().numpy()
            if draft is not None:
                th = extract_context_feature(hs, draft.target_layer_ids)  # (1, T, 5H)
                cache = DynamicCache(config=draft.config)
                if hasattr(cache, "activate_past_recording"):
                    cache.activate_past_recording()
                prev_end = 0
                toks = np.zeros((N, B - 1, 4), np.int32)
                lps = np.zeros((N, B - 1, 4), np.float16)
                pos_all = torch.arange(P + N + B, device=dev)[None]
                for t in range(N):
                    start = P + t - 1
                    anchor = full[:, start : start + 1]
                    block = torch.full((1, B), mask_id, dtype=torch.long, device=dev)
                    block[:, 0] = anchor
                    noise = F.embedding(block, emb_w)
                    dh = draft(target_hidden=th[:, prev_end:start], noise_embedding=noise,
                               position_ids=pos_all[:, prev_end : start + B], past_key_values=cache, use_cache=True)[:, 1:, :]
                    cache.crop(-(B))  # block 分を落とし、context 分 (prev_end..start) は残す
                    logits = draft.compute_logits(dh, lm_head).float()  # (1, B-1, V)
                    lp = torch.log_softmax(logits, -1)
                    top = torch.topk(lp, 4, dim=-1)
                    ti, tv = top.indices[0], top.values[0]
                    if hasattr(draft, "propose"):
                        # DFlash2: candidate selector が選ぶ経路を chain0 にする (col 0)。col 1..3 は位置別 top-2..4
                        path = draft.candidate_selector.select(dh, logits, anchor[:, 0], 0.0)[0][0]  # (B-1,)
                        ti = torch.cat([path[:, None], ti[:, 1:]], 1)
                        tv = torch.cat([lp[0].gather(-1, path[:, None]), tv[:, 1:]], 1)
                    toks[t] = ti.cpu().numpy()
                    lps[t] = tv.to(torch.float16).cpu().numpy()
                    prev_end = start
                dtok[off : off + N] = toks
                dlp[off : off + N] = lps
            idx_f.write(json.dumps({"sample_id": s["sample_id"], "domain": s["domain"], "split": split_of.get(s["sample_id"], "test"),
                                    "offset": off, "N": N, "P": P}) + "\n")
            off += N
            if (si + 1) % 50 == 0:
                print(f"{si+1}/{len(traces)} pos={off} {time.time()-t0:.0f}s", flush=True)
    hid.flush()
    if draft is not None:
        dtok.flush(); dlp.flush()
    idx_f.close()
    json.dump({"layers": LAYERS, "total": total, "hidden": Hd}, open(out / "meta.json", "w"))
    print("->", out)


if __name__ == "__main__":
    main()
