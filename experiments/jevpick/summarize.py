"""全 runtime / vLLM / llama.cpp / scorer 結果を 1 枚の表にまとめる (何もしない = 1.00x 基準)。"""
from __future__ import annotations

import glob
import json
from pathlib import Path

rows = []  # (model, quant, runtime, ctx, task, method, speedup, tok_s, note)


def add(model, quant, rt, ctx, task, method, sp, tps, note=""):
    rows.append((model, quant, rt, ctx, task, method, sp, tps, note))


# --- HF runtime_v2 (decode 専用があれば使う) ---
def hf_runtime(path, model, quant, ctx, task, label_map):
    d = json.load(open(path))
    s = d["summary"]
    b = s["baseline"]
    bdec = b.get("decode_tok_s", b["tok_s"] / (1 - b["time_frac"].get("prefill", 0)))
    for m, r in s.items():
        dec = r.get("decode_tok_s", r["tok_s"] / (1 - r["time_frac"].get("prefill", 0)))
        add(model, quant, "HF eager", ctx, task, label_map.get(m, m), dec / bdec, dec,
            f"受理 {r['mean_accepted']:.2f}/step, 一致 {r['exact_match']:.0%}")


LM4 = {"baseline": "なし", "prior": "有限候補 (prior top-1)", "scorer": "ChoiceSpec: 有限候補+scorer", "dflash": "DFlash (draft のみ)",
       "union": "ChoiceSpec: DFlash+有限候補+scorer", "hybrid": "ChoiceSpec hybrid (scorer が draft 呼び出しを制御)"}
LM27 = {**LM4, "dflash": "DFlash2 (draft のみ)", "union": "ChoiceSpec: DFlash2+有限候補+scorer", "mtp": "MTP head (draft のみ, HF 実装)",
        "union_all": "ChoiceSpec: MTP+DFlash2+有限候補+scorer", "hybrid_all": "ChoiceSpec hybrid_all (scorer が MTP/DFlash2 呼び出しを制御)"}
for f, args in [
    ("experiments/jevpick/phase4_runtime/runtime_v2_toolcall_union_L8_hybrid.json", ("Qwen3-4B", "bf16", "300", "Tool Call", LM4)),
    ("experiments/jevpick/phase4_runtime/runtime_v2_toolcall_union_L16.json", ("Qwen3-4B", "bf16", "300 (L=16)", "Tool Call", LM4)),
    ("experiments/jevpick/phase4_runtime/runtime_v2_python_union_L8.json", ("Qwen3-4B", "bf16", "300", "Python (repo)", LM4)),
    ("experiments/jevpick/phase4_runtime/runtime_v2_toolcall_union_L8_ctx16k.json", ("Qwen3-4B", "bf16", "16k", "Tool Call", LM4)),
    ("experiments/jevpick/phase4_runtime/runtime_v2_python_union_L8_ctx16k.json", ("Qwen3-4B", "bf16", "16k", "Python (repo)", LM4)),
    ("experiments/jevpick/phase4_runtime/runtime_v2_27b_toolcall_union_L8_bf16.json", ("Qwen3.8-27B", "bf16", "300", "Tool Call", LM27)),
    ("experiments/jevpick/phase4_runtime/runtime_v2_27b_toolcall_union_L8_fp8.json", ("Qwen3.8-27B", "FP8", "300", "Tool Call", LM27)),
    ("experiments/jevpick/phase4_runtime/runtime_v2_27b_toolcall_all_L8_nf4.json", ("Qwen3.8-27B", "NF4 (Q4, bitsandbytes)", "300", "Tool Call", LM27)),
    ("experiments/jevpick/phase4_runtime/runtime_v2_27b_toolcall_all_L8_bf16.json", ("Qwen3.8-27B", "bf16 (+MTP)", "300", "Tool Call", LM27)),
]:
    if Path(f).exists():
        hf_runtime(f, *args)

# --- vLLM ---
VL = {"none": "なし", "ngram": "vLLM ngram (prompt lookup)", "ngram16": "vLLM ngram k=16", "dflash": "vLLM DFlash", "mtp": "vLLM MTP k=3", "mtp1": "vLLM MTP k=1", "mtp7": "vLLM MTP k=7", "mtp15": "vLLM MTP k=15"}
for f, model, quant, dfl in [
    ("experiments/jevpick/phase4_runtime/vllm_Qwen3-4B.json", "Qwen3-4B", "bf16", "vLLM DFlash k=15"),
    ("experiments/jevpick/phase4_runtime/vllm_Qwen3.8-27B-bf16.json", "Qwen3.8-27B", "bf16", "vLLM DFlash2 k=7"),
    ("experiments/jevpick/phase4_runtime/vllm_Qwen3.8-27B-bf16-mtp.json", "Qwen3.8-27B", "bf16", "vLLM DFlash2 k=7"),
    ("experiments/jevpick/phase4_runtime/vllm_Qwen3.8-27B-bf16-mtp7.json", "Qwen3.8-27B", "bf16", "vLLM DFlash2 k=15"),
    ("experiments/jevpick/phase4_runtime/vllm_Qwen3.8-27B-AWQ-INT4.json", "Qwen3.8-27B", "AWQ INT4 (Q4)", "vLLM DFlash2 k=7"),
    ("experiments/jevpick/phase4_runtime/vllm_Qwen3.8-27B-NVFP4.json", "Qwen3.8-27B", "NVFP4 (Q4)", "vLLM DFlash2 k=7"),
    ("experiments/jevpick/phase4_runtime/vllm_Qwen3.8-Flash-Next-FP8.json", "Qwen3.8-Flash-Next (MoE)", "FP8 TP=4", "-"),
    ("experiments/jevpick/phase4_runtime/vllm_Qwen3.8-Flash-Next-FP8-mtp.json", "Qwen3.8-Flash-Next (MoE)", "FP8 TP=4", "-"),
]:
    if not Path(f).exists():
        continue
    d = json.load(open(f))
    base_f = f.replace("-mtp7.json", ".json").replace("-mtp.json", ".json")
    base = json.load(open(base_f)).get("none") if Path(base_f).exists() else d.get("none")
    for cfg, r in d.items():
        if "error" in r or cfg == "none" and f != base_f:
            continue
        for dom in ("toolcall_short", "python_short", "toolcall_16k", "python_16k"):
            if dom in r and base and dom in base:
                task = ("Tool Call" if dom.startswith("toolcall") else "Python (repo)")
                ctx = "16k" if dom.endswith("16k") else "300"
                sp = r[dom]["decode_tok_s"] / base[dom]["decode_tok_s"]
                lab = dfl if cfg == "dflash" else VL.get(cfg, cfg)
                add(model, quant, "vLLM 0.29", ctx, task, lab, sp, r[dom]["decode_tok_s"], f"一致 {r[dom].get('exact_match_vs_none', 1.0):.0%}" if cfg != "none" else "")

# --- llama.cpp ---
f = "experiments/jevpick/phase4_runtime/llamacpp_Qwen3.8-27B-Q4_K_M.json"
if Path(f).exists():
    d = json.load(open(f))
    if "none" in d and "error" not in d["none"]:
        b = d["none"]["decode_tok_s"]
        LL = {"none": "なし", "ngram": "llama.cpp ngram-simple", "mtp": "llama.cpp MTP k=3", "mtp7": "llama.cpp MTP k=7", "dflash": "llama.cpp DFlash2 k=7"}
        for cfg, r in d.items():
            if "error" in r:
                continue
            add("Qwen3.8-27B", "GGUF Q4_K_M", "llama.cpp", "300", "Tool Call", LL.get(cfg, cfg), r["decode_tok_s"] / b, r["decode_tok_s"])

# --- 出力 ---
lines = ["# 手法 × タスク × モデル: 何もしない (1.00x) 比の decode 高速化", "",
         "decode 専用 tok/s の比 (prefill 除外)、batch=1、greedy。HF eager は kernel 未最適化 (特に 27B の linear attention) なので比のみ参考。", "",
         "| Model | 量子化 | Runtime | ctx | Task | 手法 | speedup | tok/s | 備考 |", "|---|---|---|---|---|---|---:|---:|---|"]
for r in rows:
    lines.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | **{r[6]:.2f}x** | {r[7]:.1f} | {r[8]} |")
# scorer offline (G1) 要約
lines += ["", "## Scorer (offline, L=8): 候補選択 top-1 正解率 / 平均受理 token (prior → scorer, oracle)", "", "| Model | 評価 hidden | 候補 | prior | scorer | 受理 prior→scorer (oracle) |", "|---|---|---|---:|---:|---|"]
for f, model, hid in [
    ("experiments/jevpick/phase2_scorer/result_toolcall_finite_L8_layer2_pool.json", "Qwen3-4B", "bf16"),
    ("experiments/jevpick/phase2_scorer/result_toolcall_union_L8_layer2_pool.json", "Qwen3-4B", "bf16"),
    ("experiments/jevpick/phase2_scorer/result_python_finite_L8_layer2_pool.json", "Qwen3-4B (Python)", "bf16"),
    ("experiments/jevpick/phase2_scorer/result_python_union_L8_layer2_pool.json", "Qwen3-4B (Python)", "bf16"),
    ("experiments/jevpick/phase2_scorer/result_27b_toolcall_finite_L8_layer3_pool.json", "Qwen3.8-27B", "bf16 (layer 64)"),
    ("experiments/jevpick/phase2_scorer/result_27b_toolcall_finite_L8_layer2_pool.json", "Qwen3.8-27B", "bf16 (layer 48)"),
    ("experiments/jevpick/phase2_scorer/result_27b_toolcall_finite_L8_layer2_pool_FP8test.json", "Qwen3.8-27B", "**FP8** (bf16 学習を転移)"),
    ("experiments/jevpick/phase2_scorer/result_27b_toolcall_finite_L8_layer2_pool_NF4test.json", "Qwen3.8-27B", "**NF4 (Q4)** (bf16 学習を転移)"),
    ("experiments/jevpick/phase2_scorer/result_27b_toolcall_union_L8_layer2_pool.json", "Qwen3.8-27B", "bf16"),
    ("experiments/jevpick/phase2_scorer/result_27b_toolcall_union_L8_layer2_pool_FP8test.json", "Qwen3.8-27B", "**FP8** (転移)"),
    ("experiments/jevpick/phase2_scorer/result_27b_toolcall_union_L8_layer2_pool_NF4test.json", "Qwen3.8-27B", "**NF4** (転移)"),
    ("experiments/jevpick/phase2_scorer/result_27b_toolcall_union_mtp_L8_layer3_pool.json", "Qwen3.8-27B", "bf16"),
    ("experiments/jevpick/phase2_scorer/result_27b_toolcall_union_all_L8_layer3_pool.json", "Qwen3.8-27B", "bf16"),
]:
    if not Path(f).exists():
        continue
    d = json.load(open(f)); a = d["all"]; c = d["config"]
    lines.append(f"| {model} | {hid} | {c['source']} | {a['prior_top1_correct']:.1%} | **{a['scorer_top1_correct']:.1%}** | {a['prior_mean_acc']:.2f} → {a['scorer_mean_acc']:.2f} ({a['oracle_mean_acc']:.2f}) |")
out = Path("docs/jevpick/summary_table.md")
out.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
