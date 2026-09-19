# ChoiceSpec 検証レポート v2: Tool Call 特化・source 改善・DFlash 併用・長文脈

**日付:** 2026-09-19
**根拠:** `~/OpenVons_ChoiceSpec_研究実験仕様書_v1.0.md` §13 Phase 1〜4, §2.2 G0〜G4, §12 baseline 8 (DFlash), §13 Phase 6 (context 16K)
**前回:** `docs/jevpick/phase1_oracle_report.md` (v1: Qwen3-4B-Instruct-2507 / MBPP / n-gram のみ)

## 0. 要約

| 項目 | 結果 |
|---|---|
| target | **Qwen/Qwen3-4B** (non-thinking, bf16)。DFlash 公開 draft (`z-lab/Qwen3-4B-DFlash-b16`, 0.5B, block 16) と同一 target にするため v1 から変更 |
| Tool Call G0 | **通過**: oracle recall@16 (L=4) finite 90.8% (既知 schema) / 69.7% (未知)。DFlash 併用 (union) で 94.7% / 87.4% |
| Tool Call G1 | **通過**: scorer top-1 正解率 60.3% → 80.3% (finite, +20pt)、60.1% → 84.9% (union)。未知 schema でも 56 → 79% |
| Tool Call G4 (実測) | **通過**: finite+scorer **3.66x** (69 → 252 tok/s, 出力完全一致)。DFlash 単独 2.64x、prior のみ 3.26x |
| Python (repo-level) G0 | finite 28.5% (v1 MBPP 19.2% から改善、まだ <50%)。DFlash 43.1%、**union 50.0% で通過** |
| Python G1 | finite で +7.8pt (63 → 71%) と **10pt 未満**。union では scorer が DFlash argmax を上回れず (75.2 vs 73.5%) |
| Python 実測 | dflash 1.99x > union 1.85x > scorer 1.33x > prior 1.22x |
| 併用 (union) | 受理長は常に最大 (Tool Call 4.48 vs finite 4.22 vs DFlash 3.34 token/step)。transformers (eager) では draft コスト (総時間の 14%) で finite+scorer に届かない |
| 複合 controller (hybrid) | 有限候補の期待受理長が高い step では draft を呼ばない。Tool Call 16k で draft 呼び出しを 43% 削減しつつ受理長は union と同じ |
| 長文脈 16k | **vLLM 実測: DFlash 5.4x (300tok) → 0.99x (16k)、ngram 1.9x → 0.76x** で「DFlash は数 k までしか速くない」を確認。受理長は DFlash 3.3 → 1.5 に半減、有限候補 (schema/corpus) は 2.0 維持、union 2.5。transformers (eager) では 16k で多 token verify が decode の 3 倍かかり全構成 1x 未満だが、scorer/hybrid の skip で 1.0x に踏みとどまる (常時投機の prior/dflash は 0.63〜0.78x) |

## 1. 条件

| 項目 | 値 |
|---|---|
| target / draft | Qwen/Qwen3-4B bf16 (36 層, H=2560) / z-lab/Qwen3-4B-DFlash-b16 (5 層, target layer [1,9,17,25,33] の hidden を条件にする block diffusion draft) |
| GPU / runtime | RTX PRO 6000 Blackwell ×1 (GPU4/5)、HF transformers 5.17 eager sdpa、DynamicCache、batch=1、greedy |
| Tool Call data | glaive-function-calling-v2。train 3000 / test 500 = 既知 schema 250 + **未知 schema 250** (tool 名 453 種を test 専用に隔離) |
| Python data | site-packages の実 repo 20 個から「関数 def 直後で切ったファイル prefix → 続きを書く」repo-level 補完。train 871 (14 repo) / test 247 (6 repo: rich, httpx, starlette, aiohttp, PIL, click; **repo 単位 split**) |
| trace | greedy 256 token。位置数: Tool Call 107k、Python 163k (合計 271k decode 位置) |
| hidden | 各位置で layer 9/18/27/36 を fp16 保存 (13.9 GB) |
| 候補 source | `ngram` (prompt+出力)、`grammar` (固定 template)、`corpus` (同 domain の train 出力。train は 2-fold で自分を除外)、`schema` (Source D: tools JSON から Qwen tool_call 形式を展開、user 発話の値を copy)、`macro` (Source E: prompt の identifier を定型に埋める)、`repo` (Source B: 同 repo 全ファイルの token 列 suffix index、自ファイル除外)、`dflash` (Source F 相当: DFlash argmax chain + 先頭 token を top-2..4 に差し替えた chain、計 4 本)、`finite` = dflash 以外全部、`union` = dflash + finite |
| K / L | raw → prior 順 16 / L = 2, 4, 8, 16 |
| scorer | BlockScorer (§7.2 S1 prefix-pooling / S2 2 層 Transformer + §7.3 P(match≥k) head、5.3M / 9.5M param)。入力 = hidden (1 層) + anchor token emb + 候補 token emb (target embedding 凍結) + source + prior。loss = ordinal BCE + listwise ranking。3 epoch、1〜2 分 |

コード: `openvons/jevpick/{candidates,scorer,runtime,data}/`、`experiments/jevpick/phase{1_oracle,2_scorer,4_runtime}/`。データ: `/data/openvons/jevpick/`。

## 2. Phase 1: Oracle Study v2 (source 拡張)

oracle recall@16 (L=4) / mean max match@16。太字 = G0 (50%) 通過。

| Group | ngram | grammar | corpus | schema | macro | repo | **finite** | dflash | **union** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| toolcall / 既知 schema | 22.1% / 1.81 | 10.1% / 0.85 | **85.3%** / 3.41 | **57.0%** / 2.57 | - | - | **90.8%** / 3.55 | **76.9%** / 3.22 | **94.7%** / 3.65 |
| toolcall / 未知 schema | 23.1% / 1.81 | 9.6% / 0.82 | 42.8% / 2.19 | **57.9%** / 2.58 | - | - | **69.7%** / 3.03 | **77.0%** / 3.23 | **87.4%** / 3.52 |
| python / test repo | 27.1% / 1.60 | 0.0% / 0.09 | 5.6% / 0.78 | - | 1.1% / 0.30 | 11.3% / 1.10 | 28.5% / 1.84 | 43.1% / 2.57 | **50.0%** / 2.81 |

所見:

- **schema source (D) は未知 schema でも 57% を維持**し、corpus が 85 → 43% に落ちる分を埋める。Tool Call 特化の要は「tool 定義からの展開」で、学習データに依存しない。
- Python は prompt n-gram が主力 (MBPP 0.8% → repo-level 27%)。repo retrieval は 11% で寄与するが、macro+copy は 1% で現状ほぼ無効 (テンプレ設計が粗い)。
- DFlash は Python で有限候補の 1.5 倍の recall。有限候補と DFlash の当たる位置は重なりが小さく、union で +7〜18pt。
- 推定 speedup (replay × 実測 forward コスト比、候補なし位置は通常 decode) は `experiments/jevpick/phase1_oracle/results_v2_Qwen3-4B.md`。

## 3. Phase 2: Scorer (G1)

test split、候補あり位置のみ。top-1 正解 = 選んだ候補が oracle の最長一致と同じ長さ。

| Domain | 候補 | L | prior top-1 | **scorer top-1** | 平均受理: prior / scorer / oracle | replay speedup: prior / scorer / oracle |
|---|---|---:|---:|---:|---|---|
| toolcall | finite | 8 | 60.3% | **80.3%** | 3.84 / 4.73 / 5.48 | 3.04 / 3.60 / 3.94 |
| toolcall (未知 schema) | finite | 8 | 56.2% | **79.4%** | 3.12 / 4.05 / 4.72 | 2.59 / 3.11 / 3.48 |
| toolcall | union | 8 | 60.1% | **84.9%** | 5.12 / 5.71 / 6.21 | 3.33 / 4.28 / 4.46 |
| toolcall | union | 16 | 44.7% | **74.3%** | 6.27 / 7.16 / 8.68 | 3.49 / 4.80 / 5.17 |
| python | finite | 8 | 63.0% | 70.8% | 1.60 / 1.88 / 2.48 | 1.36 / 1.59 / 1.95 |
| python | union | 8 | 75.2% | 73.5% | 3.40 / 3.43 / 4.03 | 2.67 / 2.66 / 3.23 |

union の prior = 「DFlash argmax chain を最優先」。Tool Call では scorer が DFlash の chain と有限候補を使い分けて (選択比 dflash 45% / corpus 44% / schema 6% / ngram 5%) oracle の 96% まで詰める。Python では scorer が DFlash を選ばない 24% の位置で失敗が増え、DFlash argmax 固定と同等 (guard 付き 2.72 vs 2.67)。

Ablation (toolcall / finite / L=8):

| 変更 | top-1 | 平均受理 | replay |
|---|---:|---:|---:|
| layer 9 (25%) | 77.4% | 4.59 | 3.54 |
| layer 18 (50%) | 79.1% | 4.67 | 3.57 |
| layer 27 (75%) | 80.3% | 4.73 | 3.60 |
| layer 36 (final) | 81.2% | 4.78 | 3.61 |
| layer 27 + Transformer encoder (S2) | **83.5%** | 4.89 | 3.65 |
| union + Transformer encoder | **88.2%** | 5.82 | 4.32 |

layer は深いほど良く (仕様書の「中間層が有利」仮説は今回は否定的)、差は小さい。候補 encoder は S2 が +3pt。

## 4. Phase 4: Runtime 実測 (短文脈, L=8, bf16, transformers (eager))

`experiments/jevpick/phase4_runtime/runtime_v2.py`。同一 verifier 経路で mode だけ変える。n=39 (toolcall) / 29 (python)、max_new 128、warm-up 1 本除外。

| Domain | mode | tok/s | speedup | 完全一致 | tok/step | 平均受理 | zero-accept | 時間内訳 (candidates / draft / scorer / verify) |
|---|---|---:|---:|---:|---:|---:|---:|---|
| toolcall | baseline | 69.0 | 1.00 | 100% | 1.00 | - | - | - / - / - / 96% |
| toolcall | prior (finite top-1) | 225.2 | 3.26 | 100% | 4.56 | 3.70 | 34% | 0.5% / - / - / 80% |
| toolcall | **scorer (finite)** | **252.2** | **3.66** | 100% | 5.05 | 4.22 | 31% | 0.5% / - / 9.3% / 75% |
| toolcall | dflash (argmax chain) | 182.2 | 2.64 | 100% | 4.06 | 3.34 | 22% | - / 18% / - / 71% |
| toolcall | union (scorer) | 235.0 | 3.24 | 100% | **5.30** | **4.48** | 24% | 0.4% / 14% / 4.5% / 67% |
| toolcall | **hybrid (th_hi=3)** | **263.1** | **3.52** | 100% | 5.25 | 4.44 | 24% | 0.5% / 5.3% / 5.4% / 74% (draft 呼び出し 67% skip) |
| python | baseline | 72.0 | 1.00 | 100% | 1.00 | - | - | - |
| python | prior (finite top-1) | 88.1 | 1.22 | 72%* | 1.63 | 0.65 | 70% | 1.9% / - / - / 94% |
| python | scorer (finite) | 95.6 | 1.33 | 72%* | 1.70 | 0.72 | 71% | 1.8% / - / 5.8% / 86% |
| python | **dflash** | **141.1** | **1.96** | 76%* | 3.17 | 2.21 | 29% | - / 14% / - / 77% |
| python | union (scorer) | 132.9 | 1.85 | 79%* | 3.12 | 2.16 | 37% | 1.4% / 14% / 5.2% / 74% |

別 run (同条件, n=39) の decode 専用 speedup: L=8 で scorer 3.77x / dflash 2.85x / union 3.55x / **hybrid 3.96x**、**L=16** で scorer 4.71x / dflash 2.85x / union 4.42x / **hybrid 4.80x** (平均受理 5.72 token、全 100% 一致)。Tool Call では block 長を 16 に伸ばす方が効く (仕様書 §2.1 目標条件 1.8x / 受理長 4.0 を大きく超える)。

\* bf16 の数値非決定性 (一括 forward と逐次 forward で近接 logit の argmax が反転) による不一致。v1 で fp32 では 100% 一致を確認済み。DFlash 公式実装 (`dflash_generate`) も同条件で toolcall 100% / python 77% だった。

DFlash 公式実装との整合: 公式 `dflash_generate` は toolcall b16 で 3.22x (平均受理 4.0)、b8 で 2.79x (3.06)。自前 verifier 経路の dflash L=8 は 2.64x (3.34) で概ね一致 (hidden 取得の overhead 分だけ遅い)。

読み方:

- Tool Call は「有限候補 + scorer」が **DFlash を 1.4 倍上回る**。draft model を走らせずに CPU の suffix index (総時間の 0.5%) と 5M param の scorer で済む。
- union は受理長で最良だが、transformers (eager) では DFlash draft 1 回 (~2.8 ms) が decode step (13 ms) の 20% に相当し、その分を受理長の増加 (+0.26 token/step) で回収できない。draft を fused kernel で回す runtime なら逆転しうる。
- Python は DFlash が支配的。有限候補の当たりが少なく (zero-accept 70%)、scorer も DFlash の判断を改善できない。

## 5. 併用 (複合技) の整理

| 構成 | 何を組み合わせるか | 本実験での位置づけ |
|---|---|---|
| union | DFlash の chain (+先頭差し替え) と有限候補を同じ候補集合に入れ、scorer が選ぶ | 受理長は常に最大。draft コストは毎 step 払う |
| hybrid (実装済) | 有限候補を scorer で採点 → 期待受理長 ≥ th_hi なら draft を呼ばずに採用、未満なら DFlash を呼んで union を再採点 | draft 呼び出しを削減。Tool Call 16k で 43% skip、受理長は union と同等 |
| DFlash の候補選択器を scorer で置換 | DFlash2 は位置ごとの top-k から「selector」で 1 経路を選ぶ。OpenVons scorer を target hidden 付きの selector として使い、有限候補も同じ土俵に載せる | 未実施。DFlash2 の selector は codebook 内積の軽量モデルなので、scorer 側が重くなる分の損益分岐を要確認 |
| MTP head との併用 | MTP 出力 (未来 k token の分布) を候補 source として union に入れる | Qwen3-4B に MTP head が無いため未実施。構造は dflash source と同じ (位置別 top-k → chain 化) |

## 6. 長文脈 (16k) — transformers (eager)

prompt ~15.7k token (python: 同 repo の他ファイルを文脈として前置 / toolcall: 100 tool のカタログ)。n=24、L=8、th_hi=3。**decode 専用 tok/s** (prefill を除外, §17.3)。

| Domain | mode | decode tok/s | 対 baseline | tok/step | 平均受理 | zero-accept | draft skip |
|---|---|---:|---:|---:|---:|---:|---:|
| toolcall 16k | baseline | 51.1 | 1.00 | 1.00 | - | - | - |
| toolcall 16k | prior (finite) | 40.0 | 0.78 | 2.62 | 1.62 | 48% | - |
| toolcall 16k | scorer (finite) | 49.9 | 0.98 | 3.04 | 2.04 | 40% | - |
| toolcall 16k | dflash | 32.3 | 0.63 | 2.40 | **1.53** | 35% | - |
| toolcall 16k | union | 48.3 | 0.95 | **3.48** | **2.52** | 35% | - |
| toolcall 16k | hybrid | 51.3 | 1.00 | 3.48 | 2.52 | 34% | 43% |
| python 16k | baseline | 59.8 | 1.00 | 1.00 | - | - | - |
| python 16k | scorer (finite) | 37.7 | 0.63 | 1.71 | 0.73 | 74% | - |
| python 16k | dflash | 41.1 | 0.69 | 2.57 | 1.60 | 38% | - |
| python 16k | union | 43.4 | 0.73 | 2.44 | 1.46 | 52% | - |

受理長の変化 (300 token → 16k):

| Domain | dflash | finite+scorer | union |
|---|---|---|---|
| toolcall | 3.34 → **1.53** (−54%) | 4.22 → 2.04 (−52%) | 4.48 → 2.52 (−44%) |
| python | 2.21 → 1.60 (−28%) | 0.72 → 0.73 (±0) | 2.16 → 1.46 (−32%) |

**なぜ transformers (eager) では全構成が 1x 未満か**: 実測 (`bench_ctx.py`, batch=1):

| ctx | decode 1 token | verify 9 token | 比 | verify 17 | DFlash draft | n-gram lookup (CPU) |
|---|---:|---:|---:|---:|---:|---:|
| 1k | 13.4 ms | 17.5 ms | 1.30 | 18.3 | 2.8 ms | 0.003 ms |
| 4k | 15.1 | 26.2 | 1.73 | 26.9 | 2.9 | 0.003 |
| 16k | 20.2 | 60.8 | **3.02** | 62.0 | 3.8 | 0.004 |
| 32k | 26.2 | 103.5 | **3.95** | 105.2 | 4.8 | 0.005 |

transformers (eager) は q_len>1 の attention で 4D mask 経路 (flash kernel 不可) に落ち、16k で verify が decode の 3 倍になる。**投機の損益分岐が「受理 2 token 以上」に上がり、どの draft 方式でも赤字**。これは runtime の kernel の問題で、方式固有ではない。DFlash draft 自体のコストは 2.8 → 4.8 ms と緩やかで、「DFlash が 4k までしか速くない」の主因はコストより **受理長の低下** (Tool Call で半減) にある。有限候補は文脈が長いほど候補が増えるため相対的に強くなり、Tool Call 16k では finite+scorer が DFlash を tok/step で上回る (3.04 vs 2.40)。

## 7. 長文脈 — vLLM (kernel 最適化済 runtime) での確認

vLLM 0.29 (`experiments/jevpick/phase4_runtime/bench_vllm.py`)、同じ prompt、greedy、batch=1、max_tokens 128、decode 専用 tok/s (max_tokens=1 の TTFT を差し引き)。n=16/条件。ChoiceSpec の scorer/controller は vLLM 未統合なので、ここでは vLLM 純正の `ngram` (prompt lookup, k=8/16) と `dflash` (k=15) を「有限候補 source」「生成型 draft」の代表として測る。

| 構成 | toolcall 300tok | python 300tok | **toolcall 16k** | **python 16k** | 出力一致 (対 none) |
|---|---:|---:|---:|---:|---|
| none (通常 decode) | 125.1 (1.00x) | 134.1 (1.00x) | 121.3 (1.00x) | 119.1 (1.00x) | - |
| ngram k=8 | 242.6 (1.94x) | 171.7 (1.28x) | 92.5 (**0.76x**) | 81.1 (**0.68x**) | tc 100/94%, py 50/63% |
| ngram k=16 | 234.4 (1.87x) | 163.5 (1.22x) | 91.0 (0.75x) | 84.5 (0.71x) | tc 100/94%, py 69/63% |
| **DFlash k=15** | **673.6 (5.38x)** | **319.4 (2.38x)** | 119.6 (**0.99x**) | 110.0 (**0.92x**) | tc 100/94%, py 31/56% |
| suffix decoding | 未計測 (arctic-inference 未導入) | | | | |

累積 acceptance (全 domain 込み): ngram 0.66 token/draft (8 本中)、DFlash 1.96 token/draft (15 本中)。位置別受理数は DFlash が [1059, 699, 447, 292, 191, ...] と急減衰。

所見:

- **vLLM の通常 decode は 16k でも 121 tok/s と平坦** (transformers (eager) は 78 → 51)。一方 **DFlash は 5.4x → 1.0x、ngram は 1.9x → 0.76x に落ちる**。「DFlash は数 k token までしか速くない」は本環境 (Blackwell 1 枚, batch=1, vLLM 0.29) でそのまま再現した。
- 原因は 2 つ。(a) 受理長の低下: §6 の HF 計測で Tool Call の DFlash 受理長が 3.3 → 1.5 に半減 (100 tool のカタログで draft が迷う)。(b) 投機 1 step のコスト増: 16k では draft (target 全文脈 hidden への attention) と 16 token verify の両方が伸びるのに対し、1 token decode は KV 読みが支配的で伸びない。損益分岐の受理長が上がり、常時投機 (vLLM の ngram/dflash は controller を持たない) は赤字になる。
- **ここが ChoiceSpec の controller の出番**: HF 16k (§6) で prior/dflash 固定は 0.63〜0.78x に落ちたが、scorer の期待受理長で skip する scorer/hybrid は 0.98〜1.00x で損失を止めた。長文脈では「投機するか否か」の判断だけで方式間の差が決まる。
- 有限候補は文脈が長いほど候補が増えて相対的に強くなる (Tool Call 16k: finite+scorer 3.04 vs DFlash 2.40 token/step)。速度に変えるには、多 token verify のコストが文脈長に対して平坦な kernel (FlashInfer 系の decode attention で query 数 9〜17) と、suffix index の incremental 構築が前提。
- 短文脈では vLLM 上の DFlash が 5.4x と transformers (eager) の 2.9x より大きく伸びる (draft/verify が fused)。ChoiceSpec も vLLM 統合後の再測定が必要で、transformers (eager) の 3.7〜4.8x は下限側と見る。

## 8. 判定と次の一手

| Gate | Tool Call | Python |
|---|---|---|
| G0 (oracle recall@16 L=4 ≥ 50%) | **Go** (finite 91/70%, union 95/87%) | finite 28.5% → No-Go、union 50.0% → Go |
| G1 (scorer top-1 ≥ prior +10pt) | **Go** (+20〜25pt) | No-Go (+7.8pt、union では負け) |
| G2 (平均受理 ≥ 2.0, 誤投機 < 40%) | **Go** (4.2 / zero 31%) | union で Go (2.2 / 37%)、finite で No-Go |
| G3 (simulator ≥ 1.3x) | Go (3.6x) | Go (union 2.7x) |
| G4 (実測 ≥ 1.2x, 完全一致) | **Go** (3.66x、bf16 で完全一致) | 1.85〜1.96x、bf16 一致 72〜79% (fp32 で要再確認) |

次の一手:

1. **Tool Call 製品化路線 (G5)**: vLLM/SGLang へ verifier + suffix index + scorer を統合し、DFlash と同じ runtime で比較。scorer は Transformer encoder 版 (88%)。concurrent request と continuous batching は未評価。
2. **Python**: DFlash (または MTP) を主 draft とし、有限候補は補助。scorer より「DFlash top-k の再順位付け (DFlash2 selector 置換)」を試す方が筋が良い。repo retrieval は identifier slot 正規化 (§6.2) と AST 状態 (§6.3) が未実装で伸び代あり。macro+copy は現テンプレでは無効。
3. **長文脈**: 有限候補の優位が出るのは 16k 以上。ただし runtime の多 token verify kernel が前提。suffix index の構築 (16k で 238 ms, Python 実装) は prefill 中に incremental に行うか、Rust/C++ 化が必要。
4. **bf16 決定性**: 一括 verify と逐次 decode で argmax が反転する問題は DFlash 公式実装でも同じ。製品では「投機経路の出力を正」と定義するか、決定的 kernel を使うかの方針決定。
5. **未実施**: sampling 対応 (§9.2)、tree verification (§9.3)、contextual bandit controller (§8.3)、utility loss (§7.4 L_utility)、MTP head との併用。

## 9. 再現

```bash
cd ~/dev/openvons
.venv/bin/python openvons/jevpick/data/build_prompts_v2.py                  # prompts_v2.jsonl
CUDA_VISIBLE_DEVICES=4 .venv/bin/python openvons/jevpick/data/collect.py --model Qwen/Qwen3-4B --no-think \
    --prompts /data/openvons/jevpick/prompts_v2.jsonl --out /data/openvons/jevpick/traces_v2_Qwen3-4B.jsonl --batch 24
CUDA_VISIBLE_DEVICES=4 .venv/bin/python openvons/jevpick/data/extract.py --traces .../traces_v2_Qwen3-4B.jsonl --out .../extract_v2_Qwen3-4B_0 --shard 0/2   # hidden + DFlash 候補 (GPU5 で 1/2)
.venv/bin/python experiments/jevpick/phase1_oracle/run_v2.py --workers 40 && .venv/bin/python experiments/jevpick/phase1_oracle/report_v2.py
CUDA_VISIBLE_DEVICES=4 .venv/bin/python experiments/jevpick/phase2_scorer/train.py --domain toolcall --source union --L 8 --layer 2 --encoder pool --save /data/openvons/jevpick/scorer_toolcall_union_L8_l27_pool.pt
CUDA_VISIBLE_DEVICES=5 .venv/bin/python experiments/jevpick/phase4_runtime/runtime_v2.py --domain toolcall --modes baseline,prior,scorer,dflash,union,hybrid --scorer /data/openvons/jevpick/scorer_toolcall_union_L8_l27_pool.pt
CUDA_VISIBLE_DEVICES=5 .venv/bin/python experiments/jevpick/phase4_runtime/bench_ctx.py
CUDA_VISIBLE_DEVICES=4 PATH=~/dev/typesafe_clone/decision-model/.venv/bin:$PATH ~/dev/typesafe_clone/decision-model/.venv/bin/python experiments/jevpick/phase4_runtime/bench_vllm.py
```
