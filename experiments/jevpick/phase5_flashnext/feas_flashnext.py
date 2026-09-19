"""Qwen3.8-Flash-Next-FP8 (qwen4_exp, MoE 512 expert) で ChoiceSpec が適用できるかの確認:
  1. HF で 2 GPU に載るか (FP8 checkpoint, device_map=auto)
  2. greedy tool call 出力の形式
  3. hidden state 取得と KV 巻き戻し (負の crop) が動くか
  4. 1 token / 9 token forward の時間 (HF eager, 参考)
  5. MTP weight の有無 (mtp.*)
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
from openvons.jevpick.paths import DATA  # noqa: E402

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DynamicCache

mid = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.8-Flash-Next-FP8"
cfg = AutoConfig.from_pretrained(mid)
tc = getattr(cfg, "text_config", None) or cfg
print("model_type", cfg.model_type, "layers", tc.num_hidden_layers, "hidden", tc.hidden_size, "experts", getattr(tc, "num_experts", None), "quant", getattr(cfg, "quantization_config", {}).get("quant_method") if isinstance(getattr(cfg, "quantization_config", None), dict) else getattr(cfg, "quantization_config", None), flush=True)
from huggingface_hub import snapshot_download
d = Path(snapshot_download(mid, allow_patterns=["*.json"]))
idx = json.load(open(d / "model.safetensors.index.json"))["weight_map"]
print("mtp keys", len([k for k in idx if k.startswith("mtp")]), "total tensors", len(idx), flush=True)
tok = AutoTokenizer.from_pretrained(mid)
t0 = time.time()
m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto").eval()
print("loaded %.0fs" % (time.time() - t0), type(m).__name__, {i: round(torch.cuda.memory_allocated(i) / 1e9, 1) for i in range(torch.cuda.device_count())}, flush=True)
rows = [json.loads(l) for l in open(f"{DATA}/prompts_v2.jsonl")]
tests = [x for x in rows if x["domain"] == "toolcall" and x["split"] == "test"][:3]
dev = m.get_input_embeddings().weight.device
with torch.inference_mode():
    for r in tests:
        text = tok.apply_chat_template(r["messages"], tools=r["tools"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
        t0 = time.time()
        g = m.generate(ids, max_new_tokens=48, do_sample=False)
        out = g[0, ids.shape[1]:].tolist()
        print("greedy (%.1fs, %d tok):" % (time.time() - t0, len(out)), repr(tok.decode(out))[:200], flush=True)
    cache = DynamicCache(config=m.config)
    if hasattr(cache, "activate_past_recording"):
        cache.activate_past_recording()
    o = m(input_ids=ids[:, :-1], past_key_values=cache, use_cache=True, output_hidden_states=True)
    print("hidden_states", len(o.hidden_states), tuple(o.hidden_states[-1].shape), flush=True)
    x = ids[0, -1].item(); base = cache.get_seq_length()
    draft = out[:8]
    toks = torch.tensor([[x] + draft], device=dev)
    o2 = m(input_ids=toks, past_key_values=cache, use_cache=True, position_ids=torch.arange(base, base + 9, device=dev)[None])
    y = o2.logits[0].argmax(-1).tolist(); a = 0
    while a < len(draft) and draft[a] == y[a]:
        a += 1
    print("verify: accepted", a, "of 8 (expect 8 if bf16-consistent)", flush=True)
    cache.crop(-3)
    print("negative crop ok, len", cache.get_seq_length(), flush=True)
    for k in (1, 9):
        new = torch.randint(1000, 100000, (1, k), device=dev); pos = torch.arange(base, base + k, device=dev)[None]
        ts = []
        for i in range(8):
            if i:
                cache.crop(-k)
            torch.cuda.synchronize(); t = time.perf_counter(); m(input_ids=new, past_key_values=cache, use_cache=True, position_ids=pos); torch.cuda.synchronize(); ts.append((time.perf_counter() - t) * 1e3)
        cache.crop(-k)
        print("k=%d median %.1f ms" % (k, np.median(ts[3:])), flush=True)
print("FEAS_OK")
