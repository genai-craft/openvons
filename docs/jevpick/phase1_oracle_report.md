# ChoiceSpec 検証レポート: Phase 1 Oracle Study + verifier 先行確認

**日付:** 2026-09-19
**根拠:** `~/OpenVons_ChoiceSpec_研究実験仕様書_v1.0.md` §22「最初に実行する最小実験」、§2.2 G0、§13 Phase 4 実装順 1〜2

## 1. 条件

| 項目 | 値 |
|---|---|
| target | Qwen/Qwen3-4B-Instruct-2507, bf16 (仕様書は 7B〜35B を想定。ローカルにあった 4B で先行) |
| GPU | RTX PRO 6000 Blackwell 1 枚 (GPU4/5) |
| 推論 | HF transformers 5.17 eager, DynamicCache, batch=1 |
| decoding | greedy, max 256 token (trace 収集は batch 48 / left padding) |
| tasks | Python: MBPP 500 (関数記述 + テスト → コード) / Tool Call: glaive-function-calling-v2 500 (最初の応答が tool call のもの、tools は Qwen chat template 経由) / Chat: ultrachat_200k 250 + databricks-dolly-15k-ja 250 |
| candidate source | `prompt_ngram` / `output_ngram` / `ngram` (両者) / `grammar` (固定 template を suffix 一致で引く簡易版) / `corpus` (同ドメイン他サンプルの greedy 出力を 2-fold で suffix 一致検索。§6.2 の「外部 corpus」の最小版、自分自身は含めない) / `mixed` (全部) |
| n-gram | suffix 長 n = 6→1、prior = (n, 出現回数, 直近性)、raw 128 → K=16 |
| block | L = 2 / 4 / 8 / 16 |
| scorer | なし。oracle (K 候補中の最長一致) と prior top-1 (§12 baseline 3) を比較 |

コード: `openvons/jevpick/` (candidates / metrics / runtime / data)、`experiments/jevpick/phase1_oracle/`、`experiments/jevpick/phase4_runtime/`。
生データ: `/data/openvons/jevpick/` (prompts.jsonl, traces_*.jsonl, oracle_*.pkl, bench_verify.json)。

## 2. §22 出力表 (source=mixed, K=16)

| Domain | Oracle recall@16 L=4 | Mean max match (L=4) | Zero-hit率 (L=4) | 推定最大 speedup (oracle, 実測コスト, best L) | prior top-1 の推定 speedup | 平均出力長 |
|---|---:|---:|---:|---:|---:|---:|
| Python | 19.2% | 1.54 | 33.5% | 1.71x (L=8) | 1.25x (L=8) | 83 |
| Tool Call | **75.9%** | **3.21** | 7.3% | 4.47x (L=16) | 3.34x (L=16) | 31 |
| Chat (対照) | 7.9% | 0.82 | 58.2% | 1.25x (L=8) | 1.03x (L=4) | 243 (256 上限で切れ) |

推定 speedup は replay (順に辿って verification 回数を数える) に、下記 §5 の実測 forward 時間比を掛け、候補が無い位置は通常 decode に fallback したもの。

## 3. G0 判定 (oracle recall@16, block=4 ≥ 50%)

| Domain | prompt_ngram | output_ngram | ngram | grammar | corpus | mixed | 判定 |
|---|---:|---:|---:|---:|---:|---:|---|
| Python | 0.8% | 4.1% | 4.9% | 0.0% | 16.3% | 19.2% | **No-Go** (< 30%: §22「候補 source を先に改善」) |
| Tool Call | 20.8% | 0.4% | 21.2% | 10.6% | **74.6%** | **75.9%** | **Go** (mean max match 3.21 ≥ 3: scorer 学習へ) |
| Chat | 3.4% | 3.8% | 7.2% | 0.0% | 2.1% | 7.9% | 対照群。想定通り低い |

全 source × L の表は `experiments/jevpick/phase1_oracle/results_Qwen3-4B-Instruct-2507.md`。

### Oracle と prior top-1 の差 (= scorer の価値, §22 判断 3〜4)

| Domain | L | recall@1 (prior top-1 が正解) | recall@16 (oracle) | mean top-1 match | mean max match@16 | 推定 speedup top-1 → oracle |
|---|---:|---:|---:|---:|---:|---|
| Python | 4 | 9.7% | 19.2% | 0.84 | 1.54 | 1.24x → 1.68x |
| Tool Call | 4 | 55.8% | 75.9% | 2.59 | 3.21 | 2.32x → 2.85x |
| Tool Call | 8 | 36.7% | 61.7% | 3.97 | 5.36 | 3.08x → 3.84x |
| Chat | 4 | 5.2% | 7.9% | 0.49 | 0.82 | 1.03x → 1.24x |

Python・Tool Call とも oracle と prior の差が大きく、「候補はあるのに選べていない」位置が多い。仕様書の基準では OpenVons scorer の価値が大きい側。

## 4. verifier 先行確認 (Phase 4 実装順 1〜2, scorer なし・prior top-1)

`openvons/jevpick/runtime/verifier.py`: [x_t, d_1..d_L] を一括 forward → argmax と順に比較 → 一致 prefix + 訂正 token を採用 → KV crop。候補は ngram + corpus + grammar の prior top-1。各ドメイン先頭サンプルで通常 decode (同じコード経路、L=0) と全 token 比較。

| dtype | Domain | n | baseline tok/s | L=4: 一致 / speedup / tok/step / 平均受理 / zero-accept | L=8: 一致 / speedup / tok/step / 平均受理 / zero-accept |
|---|---|---:|---:|---|---|
| bf16 | Python | 20 | 79.8 | 90% / 1.45x / 1.73 / 0.76 / 60% | 90% / 1.43x / 1.72 / 0.75 / 65% |
| bf16 | Tool Call | 20 | 76.6 | 100% / 2.24x / 2.87 / 1.95 / 46% | **100% / 2.90x** / 3.92 / 3.12 / 41% |
| bf16 | Chat | 20 | 80.2 | 10% / 1.13x / 1.33 / 0.34 / 78% | 15% / 1.13x / 1.34 / 0.34 / 78% |
| fp32 | Python | 10 | 59.2 | **100%** / 1.64x / 2.05 / 1.08 / 47% | 100% / 1.50x / 2.07 / 1.11 / 54% |
| fp32 | Tool Call | 10 | 52.1 | **100%** / 1.87x / 2.63 / 1.75 / 50% | 100% / 2.14x / 3.58 / 2.78 / 44% |
| fp32 | Chat | 10 | 61.7 | **100%** / 1.09x / 1.31 / 0.32 / 78% | 100% / 0.95x / 1.30 / 0.30 / 79% |

- **fp32 で全ドメイン 100% 一致** → verifier のアルゴリズム (H5 lossless) は成立。
- bf16 の不一致は「1 token forward と (1+L) token forward で matmul 形状が変わり、近接 logit の argmax が反転する」数値非決定性。open-ended な chat ほど近接位置が多く、一度反転すると以後が全部ずれるので一致率が低く見える。§11.4 の「floating-point 非決定性を除いた」比較では一致。実運用では決定的 kernel の採用か許容の方針決定が必要 (既存の speculative decoding 実装も同じ性質)。
- 候補生成 (Python 実装, CPU) の overhead は総時間の 0.2〜0.7% (§2.1 の 20% 未満を満たす。baseline が vLLM 級に速くなると比率は上がる)。
- Tool Call は prior top-1 だけで G4 (1.2x, 完全一致) を超える。Python も 1.45〜1.64x。ただし baseline が HF eager (80 tok/s) である点は §6 参照。

## 5. verification コスト実測 (Phase 3 入力)

ctx=1024, batch=1, HF eager, warm-up 10 + 100 回, 中央値:

| forward token 数 k | 1 | 2 | 3 | 5 | 9 | 17 |
|---|---:|---:|---:|---:|---:|---:|
| wall ms | 13.41 | 17.06 | 17.14 | 17.32 | 17.46 | 18.08 |
| 比 (k=1 基準) | 1.00 | 1.27 | 1.28 | 1.29 | 1.30 | 1.35 |

k=1 → k=2 で +27% の段差があり (1 token decode 専用経路と多 token 経路の差)、それ以降はほぼ平坦。この段差が speedup の損益分岐を決めている: 受理長の期待値が 0.3 token 未満なら投機は損 (chat L=8 fp32 が 0.95x になった理由)。vLLM / SGLang では段差が小さいはずなので Phase 3 で再測定が必要。

## 6. 注意点 (結果の解釈, §23)

1. **target が 4B** で仕様書の 7B〜35B より小さい。受理長は target の「予測しやすさ」より candidate の質に依存するので傾向は変わらないと予想するが、未確認。
2. **baseline が HF eager 80 tok/s** と遅い。speedup 比は forward コスト比 (§5) で決まるので、高速 runtime では比が変わる (段差が消えれば有利、候補生成の CPU 比率が上がれば不利)。
3. **Python は MBPP** = repo 文脈なしの短い関数 (83 token)。prompt n-gram がほぼ効かない (0.8%) のはこのため。仕様書の主戦場である repo-level 補完 (RepoBench / CrossCodeEval 相当) なら prompt/repo retrieval が効くはずで、ここでの Python の値は下限側。
4. **Tool Call の glaive は定型が強い**。corpus source は同一データセットの他サンプルから作っているので「既知 schema」条件 (§10.3)。未知 schema・複数 tool・長い引数での再評価が必要。
5. chat の trace は 256 token で打ち切り。
6. `mixed` の prior は source 間で比較不能 (recency が別系列) なので top-1 は雑。これも scorer の出番。

## 7. 判断と次の一手 (§2.2 / §22)

| Domain | 判断 | 次 |
|---|---|---|
| Tool Call | G0 Go、mean max match ≥ 3、prior だけで G4 相当 | Week 2 へ: hidden state 収集 (§10.2)、pooling scorer、G1 目標は top-1 正解率 55.8% → 65.8% 以上 (L=4)。未知 schema split を追加 |
| Python | G0 No-Go (19.2%) だが oracle/prior 差は大きい | candidate 改善を先に: repo retrieval (Source B)、macro+copy (Source E)、データを repo-level 補完へ。その後 scorer |
| Chat | 対照群として想定通り | 現時点で投資しない (H2 を支持) |
| Runtime | H5 (lossless) 成立 (fp32) | bf16 決定性の方針決定、vLLM/SGLang 上で §5 を再測定、L の動的選択 (Tool Call は L=8、Python は L=4 が最良) |

## 8. 再現

```bash
cd ~/dev/openvons
.venv/bin/python openvons/jevpick/data/build_prompts.py 500          # prompts.jsonl
CUDA_VISIBLE_DEVICES=4 .venv/bin/python openvons/jevpick/data/collect.py --batch 48   # traces (6 分)
CUDA_VISIBLE_DEVICES=5 .venv/bin/python experiments/jevpick/phase1_oracle/bench_verify.py         # forward コスト
.venv/bin/python experiments/jevpick/phase1_oracle/run.py --workers 48            # oracle (数分)
.venv/bin/python experiments/jevpick/phase1_oracle/report.py                      # results_*.md
CUDA_VISIBLE_DEVICES=4 .venv/bin/python experiments/jevpick/phase4_runtime/golden_test.py [--dtype float32 --per-domain 10]
```
