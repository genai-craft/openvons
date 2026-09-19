# ChoiceSpec 検証レポート v3: Qwen3.8-27B（bf16 / FP8 / NVFP4）と MTP・DFlash2 併用

**日付:** 2026-09-19
**問い:** 27B 級（量子化前提）で ChoiceSpec の予測機は成り立つか。MTP / DFlash2 と比べて・組み合わせてどうか。
**前回:** `phase2_report.md`（Qwen3-4B）

## 0. 結論

| 問い | 答え |
|---|---|
| 27B で scorer は成り立つか | **成り立つ**。Tool Call: prior top-1 64.1% → **scorer 84.3%**（+20pt、4B と同じ改善幅）。平均受理 4.41 → 5.38 token（oracle 5.97, L=8） |
| 量子化しても成り立つか | **成り立つ**。bf16 で学習した scorer を **FP8 モデルの hidden state にそのまま適用して 83.8%**（−0.5pt）。候補の当たり方も bf16/FP8 でほぼ同じ（oracle recall 90.1 vs 90.8%）。FP8 の greedy 出力は bf16 と 96.6% 一致 |
| MTP（checkpoint 同梱 head）は | Tool Call で非常に強い: 1 層 0.42B の head で step-1 正解率 99%、7 token chain の平均一致 6.27 token（K=7）。vLLM 実測では bf16 27B で MTP k=15 が 7.1x、NVFP4 で k=7 が 6.0x。Tool Call では MTP がほぼ飽和しており、scorer の上積み余地は小さい |
| DFlash2（2B draft）との併用は | DFlash2 単独で平均受理 6.11/8 と強く、有限候補を足した union は 6.30（oracle 6.83）。**27B では「有限候補 + scorer」の単独価値より、生成型 draft（MTP/DFlash2）の候補を選ぶ/skip する controller としての価値が主** |
| Flash-Next（qwen4_exp MoE, FP8 186GB） | transformers (Hugging Face の推論ライブラリ) では load 不可（FP8 MoE 重みの変換失敗）→ scorer 用 hidden tap は vLLM 側に要実装。vLLM TP=4 で通常 102 tok/s、**MTP k=7 で 590 tok/s（5.77x）**、ngram は PLE 衝突で不可（§6） |
| 実測速度 | vLLM（scorer 未統合）: 4-bit 27B で DFlash2 4.6〜6.2x、MTP 4.3〜6.0x。transformers 上の runtime（scorer 込み）: 有限候補+scorer 3.5〜3.8x、DFlash2 併用 4.7x（bf16/FP8）、NF4 は §5.2 |

## 1. 条件

| 項目 | 値 |
|---|---|
| target | Qwen/Qwen3.8-27B（Qwen3_5ForCausalLM、64 層 = linear attention 48 + full attention 16、hidden 5120）、non-thinking、greedy |
| 量子化 | HF: FineGrainedFP8（on-the-fly、weight 128×128 block、linear_attn の in_proj_a/b・lm_head・norm は除外）29.5 GB。vLLM: unsloth/Qwen3.8-27B-NVFP4（速度のみ） |
| draft | incoai/Qwen3.8-27B-DFlash2（2B、block 8、candidate selector 付き）／ checkpoint 同梱 MTP head（1 層、fc + full-attention layer、0.42B）を transformers 上で自前実装（`candidates/mtp_qwen35.py`、vLLM の qwen3_5_mtp.py と同じ式） |
| data | Tool Call のみ。train 3000 / test 500（既知 250 + 未知 schema 250）。bf16 trace 150k 位置、FP8 trace は test 500 件 |
| tool_call 形式 | Qwen3.5 系は `<tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>...` 形式。schema source（Source D）はこの形式で展開（chat template から自動判定） |
| scorer | BlockScorer pooling、layer 48（75%）、L=8、3 epoch |
| transformers 上の runtime 注意 | linear attention の fused kernel（flash-linear-attention / causal_conv1d）が無く reference 実装で動く。1 token decode 60 ms（FP8）と遅いので、transformers 上の絶対 tok/s は参考値。速度は vLLM で測る |

## 2. Oracle（G0）— bf16 と FP8

oracle recall@16（L=4）/ mean max match。test split。

| Group | ngram | corpus | schema | **finite** | DFlash2 | **union** |
|---|---:|---:|---:|---:|---:|---:|
| bf16 / 既知 schema | 32.4% / 1.95 | 88.9% / 3.60 | 78.4% / 3.16 | **90.1%** / 3.64 | 95.8% / 3.79 | **99.1%** / 3.85 |
| bf16 / 未知 schema | 31.8% / 1.93 | 59.2% / 2.75 | 79.8% / 3.21 | **79.3%** / 3.37 | 95.1% / 3.78 | **97.7%** / 3.82 |
| FP8 / 既知 schema | 32.8% / 1.97 | 89.5% / 3.63 | 79.1% / 3.18 | **90.8%** / 3.66 | 96.3% / 3.80 | **99.5%** / 3.86 |
| FP8 / 未知 schema | 32.1% / 1.95 | 59.5% / 2.76 | 80.5% / 3.24 | **79.7%** / 3.38 | 95.4% / 3.79 | **98.0%** / 3.83 |

- schema source が 4B の 57% → 27B（XML 形式）で **79%**。形式が行区切りで冗長なため、tool 定義からの展開がそのまま当たる。未知 schema でも変わらない。
- FP8 と bf16 で候補の当たり方は同じ（差 <1pt）。greedy 出力自体は 96.6% のサンプルで完全一致、共通 prefix は平均 97%。
- DFlash2（selector 経路 + 先頭差し替え 3 本）は単独で 95%。

## 3. Scorer（G1）— bf16 学習 → bf16 / FP8 評価

L=8、候補あり位置。

| 候補 | 評価 hidden | prior top-1 | **scorer top-1** | 平均受理 prior / scorer / oracle | replay speedup prior / scorer / oracle |
|---|---|---:|---:|---|---|
| finite | bf16 | 64.1% | **84.3%** | 4.41 / 5.38 / 5.97 | 3.69 / 4.53 / 5.11 |
| finite | **FP8**（転移） | 64.0% | **83.8%** | 4.44 / 5.39 / 6.00 | 3.75 / 4.62 / 5.20 |
| finite / 未知 schema | bf16 | 60.8% | 81.0% | 3.80 / 4.77 / 5.46 | 3.31 / 4.03 / 4.74 |
| union (DFlash2 + finite) | bf16 | 46.0%* | 80.8% | 6.11 / 6.30 / 6.83 | 5.22 / 5.26 / 5.77 |
| union | **FP8**（転移） | 45.5%* | 79.8% | 6.12 / 6.28 / 6.85 | 5.26 / 5.31 / 5.83 |
| finite, **layer 64（最終層）** | bf16 | 64.1% | **88.0%** | 4.41 / 5.56 / 5.97 | 3.69 / 4.65 / 5.11 |
| union, Transformer encoder | bf16 | 46.0%* | 85.3% | 6.11 / 6.39 / 6.83 | 5.22 / 5.39 / 5.77 |

\* union の prior = DFlash2 selector 経路を最優先。top-1 正解率は低いが平均受理は 6.11 と高い（DFlash2 が外すときは有限候補も外す位置が多い）。

読み方:

- **G1 は 27B でも通過**（+20pt）。hidden の層は 48（75%）。
- **FP8 転移は無損失に近い**（−0.5pt）。量子化で hidden state の分布がわずかに動いても、scorer が見ている構造（候補 token 列と文脈の整合）は保たれる。製品では量子化モデルの hidden で学習し直す必要はほぼ無い。
- union では scorer の上積みが小さい（6.11 → 6.30）。DFlash2 が 27B 用に 2B で学習された強い draft のため、有限候補が勝てる位置が少ない。scorer が有限候補を選ぶのは 51〜65%（corpus）で、それが受理を伸ばすのは既知 schema のみ。

## 4. MTP head

Qwen3.8-27B checkpoint 同梱の MTP head（`mtp.*` 15 tensor、fc + full-attention 1 層 + norm、0.42B）を transformers 上で実装し（`openvons/jevpick/candidates/mtp_qwen35.py`、vLLM `qwen3_5_mtp.py` と同式）、各 decode 位置で 7 token を chain で draft した（150k 位置）。

| 指標 | 値 |
|---|---|
| step-1（次 token）正解率 | **99.7%** |
| 7 token chain の平均一致長 | **6.27 / 7** |
| HF NF4 runtime での平均受理（L=8, 4 prompt smoke） | 7.58 / 8 |

vLLM（bf16 27B, Tool Call, decode tok/s）: MTP k=1 1.84x → k=3 3.37x → k=7 5.74x → **k=15 7.09x**（受理 759/1035 draft token）。DFlash2 は k=7 6.92x、k=15 **8.29x**。
llama.cpp（GGUF Q4_K_M, GPU7 を MTP 抽出と共用）: MTP k=3 1.87x、k=7 2.03x、DFlash2 k=7 **3.75x**（none 39.1 tok/s）。llama.cpp の MTP は 1 step の draft コストが大きく、受理率 97% でも伸びない。

**Tool Call では MTP head 単体でほぼ飽和**する（次 token 正解 99.7%）。この領域で OpenVons scorer が MTP の上に乗せられる価値は「MTP が外す 3% の位置で有限候補に切り替える」「MTP の top-k 経路の選択」に限られ、後述 §5 の union_all で定量化した。

## 5. 速度 — vLLM（bf16 / NVFP4）と HF

### 5.1 vLLM 0.29（kernel 最適化済 runtime、batch=1、Tool Call 300 token、decode 専用 tok/s）

| 量子化 | none | ngram (prompt lookup) | DFlash2 k=7 | DFlash2 k=15 | MTP k=7 | MTP k=15 |
|---|---:|---:|---:|---:|---:|---:|
| bf16 (55 GB) | 27.1 (1.00x) | 50.2 (1.85x) | 187.3 (6.92x) | **224.7 (8.29x)** | 155.5 (5.74x) | 192.0 (7.09x) |
| **NVFP4** (unsloth, ~16 GB) | 49.6 (1.00x) | 40.3 (0.81x) | 228.8 (4.61x) | - | **295.2 (5.95x)** | - |
| **AWQ INT4** (cyankiwi W4A16) | 62.9 (1.00x) | 76.9 (1.22x) | **391.2 (6.22x)** | - | 267.8 (4.26x) | - |
| **GGUF Q4_K_M** (llama.cpp) | 39.1 (1.00x) | 38.9 (1.00x) | 146.7 (3.75x) | - | 79.3 (2.03x) | - |

- 4-bit でも DFlash2 / MTP は 4〜6x。draft の受理率は量子化でほぼ変わらない（DFlash2 6.0/7、MTP 6.8/7 token per draft）。
- vLLM/llama.cpp の ngram は Tool Call でほぼ効かない（受理 1.2〜1.3 token/draft、min n-gram=1 で外れ draft を出し続ける）。ChoiceSpec の有限候補（schema 展開 + corpus、oracle 90%）とは別物。
- NVFP4 は vLLM が FlashInfer を要求（`nvcc` を PATH に置いて JIT）。AWQ は Marlin kernel で追加依存なし。

### 5.2 ChoiceSpec との組み合わせ（transformers 上の runtime、同一 verifier 経路、L=8、n=29）

scorer は vLLM 未統合のため、ここは transformers (eager)（27B の linear attention は reference 実装で 1 token decode が 60 ms 前後と遅い）。**比のみ**を見る。

| 量子化 | mode | decode tok/s | 対 baseline | 平均受理/step | 一致 |
|---|---|---:|---:|---:|---:|
| bf16 | baseline | 18.2 | 1.00x | - | 100% |
| bf16 | 有限候補 prior top-1 | 51.8 | 2.85x | 2.43 | 38%* |
| bf16 | **有限候補 + scorer** | 69.3 | **3.81x** | 3.42 | 55%* |
| bf16 | DFlash2 のみ | 87.2 | 4.79x | 5.73 | 86%* |
| bf16 | DFlash2 + 有限候補 + scorer (union) | 85.6 | 4.70x | 5.78 | 83%* |
| bf16 | hybrid (scorer が draft 呼び出しを 60% skip) | 84.3 | 4.63x | 5.01 | 59%* |
| FP8 | baseline | 15.8 | 1.00x | - | 100% |
| FP8 | 有限候補 + scorer | 55.4 | **3.51x** | 3.40 | 59%* |
| FP8 | DFlash2 のみ | 74.5 | 4.72x | 5.73 | 90%* |
| FP8 | DFlash2 + 有限候補 + scorer | 75.0 | 4.75x | 5.94 | 90%* |
| **NF4 (Q4)** | baseline | 23.3 | 1.00x | - | 100% |
| NF4 | 有限候補 + scorer | 73.6 | **3.16x** | 3.48 | 55%* |
| NF4 | DFlash2 のみ | 90.5 | 3.89x | 5.70 | 90%* |
| NF4 | MTP head のみ（transformers 上の自前実装、draft 時間 20%） | 76.7 | 3.30x | **6.97** | 86%* |
| NF4 | DFlash2 + 有限候補 + scorer (union) | 91.0 | **3.91x** | 5.84 | 86%* |
| NF4 | MTP + DFlash2 + 有限候補 + scorer (union_all) | 68.3 | 2.93x | 6.28 | 86%* |
| NF4 | hybrid_all（draft 呼び出し 68% skip） | 75.9 | 3.26x | 5.33 | 69%* |
| bf16 (+MTP run) | baseline | 18.3 | 1.00x | - | 100% |
| bf16 | 有限候補 + scorer | 71.1 | 3.88x | 3.50 | 59%* |
| bf16 | DFlash2 のみ | 89.3 | **4.87x** | 5.82 | 90%* |
| bf16 | MTP head のみ（transformers 上の自前実装、draft 時間 22%） | 78.4 | 4.28x | **7.02** | 93%* |
| bf16 | MTP + DFlash2 + 有限候補 + scorer (union_all) | 67.8 | 3.70x | 6.31 | 93%* |
| bf16 | hybrid_all（draft 呼び出し 66% skip） | 71.9 | 3.92x | 5.15 | 69%* |

\* bf16/FP8 の数値非決定性（一括 verify と逐次 decode で近接 logit の argmax が反転。27B は linear attention の chunk/recurrent 実装差も加わる）。4B の fp32 では 100% 一致を確認済み。DFlash2 公式実装も同条件で 86〜94%。



### 5.3 Jev 相当技術（scorer）の上積みはどこにあるか — 27B Tool Call、offline replay（L=8、位置単位）

| 候補集合 | 単純選択 (prior) | **scorer** | oracle | scorer の上積み |
|---|---:|---:|---:|---|
| 有限候補のみ | 4.41 token | **5.56** | 5.97 | **+1.15 token/step（+26%）**、top-1 64 → 88% |
| DFlash2 + 有限 | 6.11 | 6.30〜6.39 | 6.83 | +0.2〜0.3（+4%） |
| MTP + 有限 | 6.32 | 6.58 | 6.93 | +0.26（+4%）、未知 schema では +0.04 |
| MTP + DFlash2 + 有限 | 6.11 | 6.57 | 6.95 | +0.46（+8%） |

- **draft model が無い条件では scorer が主役**（+26%、oracle の 93% まで到達）。学習 1〜2 分、5M param、CPU の suffix index で済み、量子化版にも転移する。
- **学習済み draft（MTP / DFlash2）がある条件では上積み 4〜8%**。Tool Call は MTP の次 token 正解率が 99.7% と飽和しているため。online（transformers 上の runtime）では scorer の選択が MTP 単独を下回る場面もあり（union_all 6.28 vs mtp 6.97）、draft と有限候補の混在時の学習（位置分布・候補長の不一致）が課題。
- scorer の**もう一つの価値は controller**: 期待受理長で draft 呼び出しを 60〜68% skip（hybrid）、長文脈で投機を止めて損失を防ぐ（v2 §6）。

## 6. Qwen3.8-Flash-Next への適用可否

**モデル**: `Qwen/Qwen3.8-Flash-Next-FP8`（`qwen4_exp`、Qwen4 の先行アーキ）。48 層、hidden 2560、MoE 512 expert 中 10 + shared 1 活性、hybrid（Gated DeltaNet 3 : Qwen Sparse Attention 1）、**N-gram PLE embedding**（生の input_ids から n-gram 特徴を層ごとに埋め込む）、MTP head 同梱。FP8 で 186 GB → 1 GPU に乗らず、TP=2 でも 93 GB/GPU で不足、TP=4（4 GPU）で動く。

| 経路 | 結果 |
|---|---|
| Hugging Face の推論ライブラリ transformers 5.17（hidden state 取得 → JevPick） | **不可**。`qwen4_exp` のクラスはあるが、FP8 checkpoint の fused expert 重み（`mlp.experts.gate_up_proj` / `down_proj`）の自動変換が失敗。bf16 版は 370 GB でこの機材に乗らない |
| vLLM 0.29、TP=4、block_size=320（QSA ring capacity 20 の倍数が必要）、通常 decode | **可**: Tool Call 300 token で **102.3 tok/s**（27B bf16 の 27 tok/s、NVFP4 の 50 tok/s より速い: 活性 ~4B の MoE） |
| vLLM ngram（prompt lookup） | **不可**: draft token に対して PLE 入力が用意されず `PLE inputs were not prepared`。n-gram 埋め込みを持つアーキと外部 n-gram draft の相性問題 |
| vLLM MTP k=7 | **可**: **590.3 tok/s（5.77x）**、受理 752/833 draft token（6.3/7 per draft）。同梱 MTP は Flash-Next でも Tool Call でほぼ飽和 |

**ChoiceSpec 適用の見立て**:

1. **候補生成（有限候補）は適用可**。tool schema 展開・corpus・prompt n-gram は target の tokenizer と chat template だけに依存し、Flash-Next の tool_call 形式は 27B と同じ XML 形式（同じ chat template 系）。oracle 評価は vLLM の greedy trace だけで回せる。
2. **scorer は hidden state のタップが前提**。transformers で読めないため、vLLM 内部（`Qwen4ExpModel.forward` の最終 hidden）にフックを入れるか、transformers の qwen4_exp FP8 変換が直るのを待つ必要がある。scorer 自体は「hidden 1 本 + 候補 token 列」しか見ないので、タップさえ取れれば 27B と同じ手順（1〜2 分の学習）で載る。
3. **verifier（KV 巻き戻し）**は Gated DeltaNet + QSA の両方で recurrent/sparse 状態の巻き戻しが要る。vLLM の MTP が動くなら runtime 側には巻き戻し機構が既にあり、そこに候補を差し込む形になる。
4. **MTP head が同梱**なので、Tool Call では 27B と同様に MTP 単独でほぼ飽和すると予想。ChoiceSpec の役割は controller（投機の可否・長さ）と未知 schema の補完に寄る。
5. **PLE（n-gram 埋め込み）が draft と衝突**する点は要注意。vLLM の ngram が壊れたのと同じ理由で、外部候補を verify する際に draft token 分の PLE 入力を作る実装が必要。

## 7. まとめ: 27B で ChoiceSpec をどう使うか

1. **予測機として**: 27B・FP8 で成り立つ。scorer は target ごとに 1〜2 分の学習、量子化版への転移可。
2. **Tool Call 単独の高速化器として**: 有限候補 + scorer で replay 4.5〜4.6x（L=8）。draft model 不要・VRAM 増ほぼ 0（scorer 5M param + CPU index）。
3. **MTP / DFlash2 がある場合**: 生成型 draft の方が受理長で勝つ（MTP 6.3/7、DFlash2 6.1/8 vs 有限 5.4/8）。ChoiceSpec の役割は「候補の選択」より「投機の可否・長さの判断（controller）」と「未知 schema での補完」に寄る。
4. **量子化の実運用**: transformers の FP8 は kernel 未整備で遅い。NVFP4/FP8 は vLLM/SGLang で動かし、scorer は最終層近くの hidden を 1 本抜くだけなので統合コストは小さい（DFlash も同じ hidden tap を使う）。
