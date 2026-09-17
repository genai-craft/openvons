# Architecture

## 入力レイアウト (models/encoding.py)

```
State:\n{state}\n\n                          ← 質問間で共有 (kv_shared / block_diag)
Question: {text}\nOptions:\n
A. {id}: {description}\n                     ← 各 option 行の最終 token の hidden state = option vector
B. ...\n
Decision:[<|decision|>]                      ← query vector (pooling=last は ':' の位置、decision は追加 special token)
```

セグメントごとに tokenize して連結するので位置は常に正確。option vector は causal attention で state を
既に見ているため、「この選択肢が state に合うか」の判断が transformer 内部で行われる。

## Head (models/heads.py)

| head | 式 | option 数 | 未知の選択肢 |
|---|---|---|---|
| linear | `Linear(H, 2)` (Noul) / `Linear(H, 255)` + mask | 固定位置 | × (位置 = クラス) |
| embed | `<Wq h_dec, Wk h_opt_i> / sqrt(d)` | 可変 | ○ |
| mlp | `MLP(Wq h_dec + Wk h_opt_i)` | 可変 | ○ |

## Multi-question (backends/model_backend.py)

| mode | 説明 |
|---|---|
| naive | 質問ごとに 1 forward |
| batched | 質問を batch 次元に並べて 1 forward (padding) |
| kv_shared | state を 1 回 prefill → KV cache を B 倍に複製 → 質問を batch で forward |
| block_diag | `[state | Q1 | Q2 | ...]` を 1 系列にし、4D の block-diagonal attention mask + 質問ごとに position を state 直後から再開 |

fp32 で 4 方式のロジットは 1e-4 以内で一致することを `scripts/test_model.py` で確認済み。

## 学習 (training/train.py)

- `head`: backbone 凍結。pooled / option vectors を一度抽出して `/data/decision_model/features` にキャッシュし、head だけを数秒で学習 → 損失・head の ablation が安価
- `lora`: peft LoRA (q,k,v,o,gate,up,down) + head
- `topN` / `full`: 上位 N 層 / 全層を fp32 で学習
- 全モードで valid の予測から temperature を推定し、test の ECE を calibration 前後で記録

## Calibration (calibration/)

Temperature scaling (必須)、vector scaling、isotonic (top-label)、Platt (Noul)。
指標は ECE (15 bin, top-label)、Brier (多クラス和)、NLL、reliability table。
