# 手法 × タスク × モデル: 何もしない (1.00x) 比の decode 高速化

decode 専用 tok/s の比 (prefill 除外)、batch=1、greedy。transformers (Hugging Face の推論ライブラリ、eager 実行) は kernel 未最適化 (特に 27B の linear attention) なので比のみ参考。

| Model | 量子化 | Runtime | ctx | Task | 手法 | speedup | tok/s | 備考 |
|---|---|---|---|---|---|---:|---:|---|
| Qwen3-4B | bf16 | transformers (eager) | 300 | Tool Call | なし | **1.00x** | 78.1 | 受理 0.00/step, 一致 100% |
| Qwen3-4B | bf16 | transformers (eager) | 300 | Tool Call | ChoiceSpec: 有限候補+scorer | **3.77x** | 293.9 | 受理 4.10/step, 一致 100% |
| Qwen3-4B | bf16 | transformers (eager) | 300 | Tool Call | DFlash (draft のみ) | **2.85x** | 222.6 | 受理 3.34/step, 一致 100% |
| Qwen3-4B | bf16 | transformers (eager) | 300 | Tool Call | ChoiceSpec: DFlash+有限候補+scorer | **3.55x** | 277.2 | 受理 4.44/step, 一致 100% |
| Qwen3-4B | bf16 | transformers (eager) | 300 | Tool Call | ChoiceSpec hybrid (scorer が draft 呼び出しを制御) | **3.96x** | 308.9 | 受理 4.44/step, 一致 100% |
| Qwen3-4B | bf16 | transformers (eager) | 300 (L=16) | Tool Call | なし | **1.00x** | 78.2 | 受理 0.00/step, 一致 100% |
| Qwen3-4B | bf16 | transformers (eager) | 300 (L=16) | Tool Call | ChoiceSpec: 有限候補+scorer | **4.71x** | 368.5 | 受理 5.07/step, 一致 100% |
| Qwen3-4B | bf16 | transformers (eager) | 300 (L=16) | Tool Call | DFlash (draft のみ) | **2.85x** | 222.8 | 受理 3.65/step, 一致 100% |
| Qwen3-4B | bf16 | transformers (eager) | 300 (L=16) | Tool Call | ChoiceSpec: DFlash+有限候補+scorer | **4.42x** | 345.4 | 受理 5.83/step, 一致 100% |
| Qwen3-4B | bf16 | transformers (eager) | 300 (L=16) | Tool Call | ChoiceSpec hybrid (scorer が draft 呼び出しを制御) | **4.80x** | 375.5 | 受理 5.72/step, 一致 100% |
| Qwen3-4B | bf16 | transformers (eager) | 300 | Python (repo) | なし | **1.00x** | 74.2 | 受理 0.00/step, 一致 100% |
| Qwen3-4B | bf16 | transformers (eager) | 300 | Python (repo) | 有限候補 (prior top-1) | **1.19x** | 88.4 | 受理 0.65/step, 一致 72% |
| Qwen3-4B | bf16 | transformers (eager) | 300 | Python (repo) | ChoiceSpec: 有限候補+scorer | **1.30x** | 96.3 | 受理 0.67/step, 一致 79% |
| Qwen3-4B | bf16 | transformers (eager) | 300 | Python (repo) | DFlash (draft のみ) | **2.05x** | 152.4 | 受理 2.21/step, 一致 76% |
| Qwen3-4B | bf16 | transformers (eager) | 300 | Python (repo) | ChoiceSpec: DFlash+有限候補+scorer | **1.90x** | 141.3 | 受理 2.16/step, 一致 79% |
| Qwen3-4B | bf16 | transformers (eager) | 16k | Tool Call | なし | **1.00x** | 51.1 | 受理 0.00/step, 一致 100% |
| Qwen3-4B | bf16 | transformers (eager) | 16k | Tool Call | 有限候補 (prior top-1) | **0.78x** | 40.0 | 受理 1.62/step, 一致 100% |
| Qwen3-4B | bf16 | transformers (eager) | 16k | Tool Call | ChoiceSpec: 有限候補+scorer | **0.98x** | 49.9 | 受理 2.04/step, 一致 100% |
| Qwen3-4B | bf16 | transformers (eager) | 16k | Tool Call | DFlash (draft のみ) | **0.63x** | 32.3 | 受理 1.53/step, 一致 100% |
| Qwen3-4B | bf16 | transformers (eager) | 16k | Tool Call | ChoiceSpec: DFlash+有限候補+scorer | **0.95x** | 48.3 | 受理 2.52/step, 一致 100% |
| Qwen3-4B | bf16 | transformers (eager) | 16k | Tool Call | ChoiceSpec hybrid (scorer が draft 呼び出しを制御) | **1.00x** | 51.3 | 受理 2.52/step, 一致 100% |
| Qwen3-4B | bf16 | transformers (eager) | 16k | Python (repo) | なし | **1.00x** | 59.8 | 受理 0.00/step, 一致 100% |
| Qwen3-4B | bf16 | transformers (eager) | 16k | Python (repo) | 有限候補 (prior top-1) | **0.48x** | 28.9 | 受理 0.73/step, 一致 62% |
| Qwen3-4B | bf16 | transformers (eager) | 16k | Python (repo) | ChoiceSpec: 有限候補+scorer | **0.63x** | 37.7 | 受理 0.73/step, 一致 62% |
| Qwen3-4B | bf16 | transformers (eager) | 16k | Python (repo) | DFlash (draft のみ) | **0.69x** | 41.1 | 受理 1.60/step, 一致 54% |
| Qwen3-4B | bf16 | transformers (eager) | 16k | Python (repo) | ChoiceSpec: DFlash+有限候補+scorer | **0.73x** | 43.4 | 受理 1.46/step, 一致 67% |
| Qwen3-4B | bf16 | transformers (eager) | 16k | Python (repo) | ChoiceSpec hybrid (scorer が draft 呼び出しを制御) | **0.70x** | 41.9 | 受理 1.46/step, 一致 67% |
| Qwen3.8-27B | bf16 | transformers (eager) | 300 | Tool Call | なし | **1.00x** | 18.2 | 受理 0.00/step, 一致 100% |
| Qwen3.8-27B | bf16 | transformers (eager) | 300 | Tool Call | 有限候補 (prior top-1) | **2.85x** | 51.8 | 受理 2.43/step, 一致 38% |
| Qwen3.8-27B | bf16 | transformers (eager) | 300 | Tool Call | ChoiceSpec: 有限候補+scorer | **3.81x** | 69.3 | 受理 3.42/step, 一致 55% |
| Qwen3.8-27B | bf16 | transformers (eager) | 300 | Tool Call | DFlash2 (draft のみ) | **4.79x** | 87.2 | 受理 5.73/step, 一致 86% |
| Qwen3.8-27B | bf16 | transformers (eager) | 300 | Tool Call | ChoiceSpec: DFlash2+有限候補+scorer | **4.70x** | 85.6 | 受理 5.78/step, 一致 83% |
| Qwen3.8-27B | bf16 | transformers (eager) | 300 | Tool Call | ChoiceSpec hybrid (scorer が draft 呼び出しを制御) | **4.63x** | 84.3 | 受理 5.01/step, 一致 59% |
| Qwen3.8-27B | FP8 | transformers (eager) | 300 | Tool Call | なし | **1.00x** | 15.8 | 受理 0.00/step, 一致 100% |
| Qwen3.8-27B | FP8 | transformers (eager) | 300 | Tool Call | 有限候補 (prior top-1) | **2.65x** | 41.8 | 受理 2.39/step, 一致 38% |
| Qwen3.8-27B | FP8 | transformers (eager) | 300 | Tool Call | ChoiceSpec: 有限候補+scorer | **3.51x** | 55.4 | 受理 3.40/step, 一致 59% |
| Qwen3.8-27B | FP8 | transformers (eager) | 300 | Tool Call | DFlash2 (draft のみ) | **4.72x** | 74.5 | 受理 5.73/step, 一致 90% |
| Qwen3.8-27B | FP8 | transformers (eager) | 300 | Tool Call | ChoiceSpec: DFlash2+有限候補+scorer | **4.75x** | 75.0 | 受理 5.94/step, 一致 90% |
| Qwen3.8-27B | FP8 | transformers (eager) | 300 | Tool Call | ChoiceSpec hybrid (scorer が draft 呼び出しを制御) | **4.51x** | 71.3 | 受理 5.18/step, 一致 66% |
| Qwen3.8-27B | NF4 (Q4, bitsandbytes) | transformers (eager) | 300 | Tool Call | なし | **1.00x** | 23.3 | 受理 0.00/step, 一致 100% |
| Qwen3.8-27B | NF4 (Q4, bitsandbytes) | transformers (eager) | 300 | Tool Call | 有限候補 (prior top-1) | **2.31x** | 53.9 | 受理 2.35/step, 一致 31% |
| Qwen3.8-27B | NF4 (Q4, bitsandbytes) | transformers (eager) | 300 | Tool Call | ChoiceSpec: 有限候補+scorer | **3.16x** | 73.6 | 受理 3.48/step, 一致 55% |
| Qwen3.8-27B | NF4 (Q4, bitsandbytes) | transformers (eager) | 300 | Tool Call | DFlash2 (draft のみ) | **3.89x** | 90.5 | 受理 5.70/step, 一致 90% |
| Qwen3.8-27B | NF4 (Q4, bitsandbytes) | transformers (eager) | 300 | Tool Call | MTP head (draft のみ, HF 実装) | **3.30x** | 76.7 | 受理 6.97/step, 一致 86% |
| Qwen3.8-27B | NF4 (Q4, bitsandbytes) | transformers (eager) | 300 | Tool Call | ChoiceSpec: DFlash2+有限候補+scorer | **3.91x** | 91.0 | 受理 5.84/step, 一致 86% |
| Qwen3.8-27B | NF4 (Q4, bitsandbytes) | transformers (eager) | 300 | Tool Call | ChoiceSpec: MTP+DFlash2+有限候補+scorer | **2.93x** | 68.3 | 受理 6.28/step, 一致 86% |
| Qwen3.8-27B | NF4 (Q4, bitsandbytes) | transformers (eager) | 300 | Tool Call | ChoiceSpec hybrid_all (scorer が MTP/DFlash2 呼び出しを制御) | **3.26x** | 75.9 | 受理 5.33/step, 一致 69% |
| Qwen3.8-27B | bf16 (+MTP) | transformers (eager) | 300 | Tool Call | なし | **1.00x** | 18.3 | 受理 0.00/step, 一致 100% |
| Qwen3.8-27B | bf16 (+MTP) | transformers (eager) | 300 | Tool Call | ChoiceSpec: 有限候補+scorer | **3.88x** | 71.1 | 受理 3.50/step, 一致 59% |
| Qwen3.8-27B | bf16 (+MTP) | transformers (eager) | 300 | Tool Call | DFlash2 (draft のみ) | **4.87x** | 89.3 | 受理 5.82/step, 一致 90% |
| Qwen3.8-27B | bf16 (+MTP) | transformers (eager) | 300 | Tool Call | MTP head (draft のみ, HF 実装) | **4.28x** | 78.4 | 受理 7.02/step, 一致 93% |
| Qwen3.8-27B | bf16 (+MTP) | transformers (eager) | 300 | Tool Call | ChoiceSpec: MTP+DFlash2+有限候補+scorer | **3.70x** | 67.8 | 受理 6.31/step, 一致 93% |
| Qwen3.8-27B | bf16 (+MTP) | transformers (eager) | 300 | Tool Call | ChoiceSpec hybrid_all (scorer が MTP/DFlash2 呼び出しを制御) | **3.92x** | 71.9 | 受理 5.15/step, 一致 69% |
| Qwen3-4B | bf16 | vLLM 0.29 | 300 | Tool Call | なし | **1.00x** | 125.1 |  |
| Qwen3-4B | bf16 | vLLM 0.29 | 300 | Python (repo) | なし | **1.00x** | 134.1 |  |
| Qwen3-4B | bf16 | vLLM 0.29 | 16k | Tool Call | なし | **1.00x** | 121.3 |  |
| Qwen3-4B | bf16 | vLLM 0.29 | 16k | Python (repo) | なし | **1.00x** | 119.1 |  |
| Qwen3-4B | bf16 | vLLM 0.29 | 300 | Tool Call | vLLM ngram (prompt lookup) | **1.94x** | 242.6 | 一致 100% |
| Qwen3-4B | bf16 | vLLM 0.29 | 300 | Python (repo) | vLLM ngram (prompt lookup) | **1.28x** | 171.7 | 一致 50% |
| Qwen3-4B | bf16 | vLLM 0.29 | 16k | Tool Call | vLLM ngram (prompt lookup) | **0.76x** | 92.5 | 一致 94% |
| Qwen3-4B | bf16 | vLLM 0.29 | 16k | Python (repo) | vLLM ngram (prompt lookup) | **0.68x** | 81.1 | 一致 62% |
| Qwen3-4B | bf16 | vLLM 0.29 | 300 | Tool Call | vLLM DFlash k=15 | **5.38x** | 673.6 | 一致 100% |
| Qwen3-4B | bf16 | vLLM 0.29 | 300 | Python (repo) | vLLM DFlash k=15 | **2.38x** | 319.4 | 一致 31% |
| Qwen3-4B | bf16 | vLLM 0.29 | 16k | Tool Call | vLLM DFlash k=15 | **0.99x** | 119.6 | 一致 94% |
| Qwen3-4B | bf16 | vLLM 0.29 | 16k | Python (repo) | vLLM DFlash k=15 | **0.92x** | 110.0 | 一致 56% |
| Qwen3-4B | bf16 | vLLM 0.29 | 300 | Tool Call | vLLM ngram k=16 | **1.87x** | 234.4 | 一致 100% |
| Qwen3-4B | bf16 | vLLM 0.29 | 300 | Python (repo) | vLLM ngram k=16 | **1.22x** | 163.5 | 一致 69% |
| Qwen3-4B | bf16 | vLLM 0.29 | 16k | Tool Call | vLLM ngram k=16 | **0.75x** | 91.0 | 一致 94% |
| Qwen3-4B | bf16 | vLLM 0.29 | 16k | Python (repo) | vLLM ngram k=16 | **0.71x** | 84.5 | 一致 62% |
| Qwen3.8-27B | bf16 | vLLM 0.29 | 300 | Tool Call | なし | **1.00x** | 27.1 |  |
| Qwen3.8-27B | bf16 | vLLM 0.29 | 300 | Tool Call | vLLM ngram (prompt lookup) | **1.85x** | 50.2 | 一致 31% |
| Qwen3.8-27B | bf16 | vLLM 0.29 | 300 | Tool Call | vLLM DFlash2 k=7 | **6.92x** | 187.3 | 一致 100% |
| Qwen3.8-27B | bf16 | vLLM 0.29 | 300 | Tool Call | vLLM MTP k=3 | **3.37x** | 91.2 | 一致 100% |
| Qwen3.8-27B | bf16 | vLLM 0.29 | 300 | Tool Call | vLLM MTP k=1 | **1.84x** | 49.9 | 一致 100% |
| Qwen3.8-27B | bf16 | vLLM 0.29 | 300 | Tool Call | vLLM MTP k=7 | **5.74x** | 155.5 | 一致 100% |
| Qwen3.8-27B | bf16 | vLLM 0.29 | 300 | Tool Call | vLLM MTP k=15 | **7.09x** | 192.0 | 一致 100% |
| Qwen3.8-27B | bf16 | vLLM 0.29 | 300 | Tool Call | vLLM DFlash2 k=15 | **8.30x** | 224.7 | 一致 100% |
| Qwen3.8-27B | AWQ INT4 (Q4) | vLLM 0.29 | 300 | Tool Call | なし | **1.00x** | 62.9 |  |
| Qwen3.8-27B | AWQ INT4 (Q4) | vLLM 0.29 | 300 | Tool Call | vLLM ngram (prompt lookup) | **1.22x** | 76.9 | 一致 38% |
| Qwen3.8-27B | AWQ INT4 (Q4) | vLLM 0.29 | 300 | Tool Call | vLLM DFlash2 k=7 | **6.22x** | 391.2 | 一致 100% |
| Qwen3.8-27B | AWQ INT4 (Q4) | vLLM 0.29 | 300 | Tool Call | vLLM MTP k=7 | **4.26x** | 267.8 | 一致 100% |
| Qwen3.8-27B | NVFP4 (Q4) | vLLM 0.29 | 300 | Tool Call | なし | **1.00x** | 49.6 |  |
| Qwen3.8-27B | NVFP4 (Q4) | vLLM 0.29 | 300 | Tool Call | vLLM ngram (prompt lookup) | **0.81x** | 40.3 | 一致 31% |
| Qwen3.8-27B | NVFP4 (Q4) | vLLM 0.29 | 300 | Tool Call | vLLM DFlash2 k=7 | **4.61x** | 228.8 | 一致 94% |
| Qwen3.8-27B | NVFP4 (Q4) | vLLM 0.29 | 300 | Tool Call | vLLM MTP k=7 | **5.95x** | 295.2 | 一致 88% |
| Qwen3.8-Flash-Next (MoE) | FP8 TP=4 | vLLM 0.29 | 300 | Tool Call | なし | **1.00x** | 102.3 |  |
| Qwen3.8-Flash-Next (MoE) | FP8 TP=4 | vLLM 0.29 | 300 | Tool Call | vLLM MTP k=7 | **5.77x** | 590.3 | 一致 100% |
| Qwen3.8-27B | GGUF Q4_K_M | llama.cpp | 300 | Tool Call | なし | **1.00x** | 39.1 |  |
| Qwen3.8-27B | GGUF Q4_K_M | llama.cpp | 300 | Tool Call | llama.cpp ngram-simple | **1.00x** | 38.9 |  |
| Qwen3.8-27B | GGUF Q4_K_M | llama.cpp | 300 | Tool Call | llama.cpp MTP k=3 | **1.87x** | 73.2 |  |
| Qwen3.8-27B | GGUF Q4_K_M | llama.cpp | 300 | Tool Call | llama.cpp MTP k=7 | **2.03x** | 79.3 |  |
| Qwen3.8-27B | GGUF Q4_K_M | llama.cpp | 300 | Tool Call | llama.cpp DFlash2 k=7 | **3.75x** | 146.7 |  |

## Scorer (offline, L=8): 候補選択 top-1 正解率 / 平均受理 token (prior → scorer, oracle)

| Model | 評価 hidden | 候補 | prior | scorer | 受理 prior→scorer (oracle) |
|---|---|---|---:|---:|---|
| Qwen3-4B | bf16 | finite | 60.3% | **80.3%** | 3.84 → 4.73 (5.48) |
| Qwen3-4B | bf16 | union | 60.1% | **84.9%** | 5.12 → 5.71 (6.21) |
| Qwen3-4B (Python) | bf16 | finite | 63.0% | **70.8%** | 1.60 → 1.88 (2.48) |
| Qwen3-4B (Python) | bf16 | union | 75.2% | **73.4%** | 3.40 → 3.43 (4.03) |
| Qwen3.8-27B | bf16 (layer 64) | finite | 64.1% | **88.0%** | 4.41 → 5.56 (5.97) |
| Qwen3.8-27B | bf16 (layer 48) | finite | 64.1% | **84.3%** | 4.41 → 5.38 (5.97) |
| Qwen3.8-27B | **FP8** (bf16 学習を転移) | finite | 64.0% | **83.8%** | 4.44 → 5.39 (6.00) |
| Qwen3.8-27B | **NF4 (Q4)** (bf16 学習を転移) | finite | 64.2% | **81.1%** | 4.34 → 5.20 (5.89) |
| Qwen3.8-27B | bf16 | union | 46.0% | **80.8%** | 6.11 → 6.30 (6.83) |
| Qwen3.8-27B | **FP8** (転移) | union | 45.5% | **79.8%** | 6.12 → 6.28 (6.85) |
| Qwen3.8-27B | **NF4** (転移) | union | 46.2% | **78.9%** | 6.04 → 6.20 (6.77) |
| Qwen3.8-27B | bf16 | union_mtp | 47.9% | **85.3%** | 6.32 → 6.58 (6.93) |
| Qwen3.8-27B | bf16 | union_all | 42.8% | **84.0%** | 6.11 → 6.57 (6.95) |
