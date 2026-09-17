# Benchmark 結果と考察 (2026-09-16, 第一マイルストーン)

GPU: RTX PRO 6000 Blackwell 96GB。データは MASSIVE scenario (18 クラス, en/ja) を主戦場に、Noul (tweet offensive)、
Score (Yelp 5 段階)、可変 choice (glaive tool routing)、60 クラス intent も実施。数値は全て test split。
全表は [experiments.md](experiments.md)。

## 1. EXP-001 生成 LLM baseline vs EXP-002 Decision Model (massive_scenario_en)

| system | 学習 | acc | ECE | ECE (温度補正後) | Brier | p50 ms (1 質問) |
|---|---|---|---|---|---|---|
| Qwen3-4B-Instruct vLLM, guided choice + logprob (zero-shot) | なし | 0.823 | 0.170 | 0.040 | 0.345 | 14.5 |
| Qwen3.8-27B vLLM, 同上 (zero-shot) | なし | 0.875 | 0.068 | 0.017 | 0.194 | 73.6 |
| Qwen3-0.6B 凍結 + embed head | head のみ 43 s | 0.881 | 0.016 | 0.020 | 0.172 | 10.4 |
| Qwen3-1.7B 凍結 + embed head | head のみ | 0.893 | 0.039 | 0.021 | — | — |
| Qwen3-4B 凍結 + embed head (last) | head のみ 160 s | 0.906 | 0.048 | 0.019 | 0.146 | 16.4 |
| Qwen3-4B 凍結 + embed head (mean pooling) | head のみ | **0.916** | 0.021 | 0.019 | 0.125 | 16.4 |
| Qwen3-4B + LoRA r16 + embed head | 2 epoch ≈ 20 min | **0.933** | 0.030 | **0.012** | **0.101** | 26.0 (adapter 未 merge) |

ja でも同じ傾向 (LLM 4B 0.795 / 27B 0.862 / 4B head 0.898 / 0.6B head 0.867)。

- **精度**: 1 万件の hard label で head だけ学習した 0.6B が、zero-shot の 27B を上回る。v1/v2 の成功基準 (baseline − 3pt 以内 / baseline 以上) は達成。
- **Calibration**: 学習済み head は補正前でも ECE 0.02〜0.05、温度補正で 0.012〜0.02 (v3 基準 ECE ≤ 0.05 達成)。zero-shot LLM は ECE 0.17 と過信。
- **速度 (単問)**: vLLM は guided choice + `max_tokens=1` にすると 4B でも 14.5 ms。これは事実上「1 forward + LM head」なので、
  HF 実装の Decision Model (14.9〜16 ms) と同等で、**LM head を外すだけでは速くならない** (H1 の速度面は不成立)。
  速度差が出るのは (a) モデル縮小 (0.6B: 10 ms)、(b) 下記の multi-question 共有、(c) JSON 生成をやめること。

## 2. EXP-007/008 質問数と latency (p50 ms, 短い state ≈ 90 token)

| system | q=1 | q=2 | q=4 | q=8 | q=16 |
|---|---|---|---|---|---|
| LLM 4B, JSON schema 構造化出力 (1 リクエスト) | 64.6 | 114 | 254 | 508 | 895 |
| LLM 4B, 質問ごと guided choice を並列 | 10.6 | 18.8 | 20.6 | 27.6 | 40.2 |
| LLM 27B, JSON schema | 359 | 622 | 1164 | 3099 | 6199 |
| Decision 4B naive | 14.9 | 29.7 | 59.4 | 119 | 239 |
| Decision 4B batched (v3) | 14.9 | 16.8 | 23.5 | 38.7 | 69.7 |
| Decision 4B kv_shared (v2) | 29.3 | 29.4 | 30.5 | 37.7 | 48.4 |
| Decision 4B **block_diag (v4)** | 15.2 | 16.2 | 17.3 | **22.6** | 32.2 |
| Decision 0.6B block_diag | — | — | — | 12.2 | 12.6 |

長い state (≈ 700 token) では 4B block_diag が 37.5 → 58.5 ms (q=1→16)、JSON baseline は 65 → 882 ms。

- v4 基準「8 質問で単問の 2 倍未満」: block_diag 22.6 ms < 15.2 × 2 で **達成**。
- 仕様の想定する「JSON 構造化出力」との比較では 8 質問で 22 倍、16 質問で 28 倍速い。
- ただし vLLM で質問を独立リクエストとして並列に投げると 4B で 40 ms (q=16) まで詰まる。prefix caching が state を共有するため。
  Decision Model の優位は「1 forward で全質問」「小モデルで十分」「確率が校正済み」に置くべき。
- kv_shared は q=1 で 2 回 forward するため単問では損。block_diag が全域で最良。

## 3. EXP-003 head と pooling (4B, hard, massive_scenario_en)

| 条件 | acc | ECE | 備考 |
|---|---|---|---|
| linear (Linear(H,255) + mask) | 0.906 | 0.027 | 0.6B では 0.838 と embed に 4pt 負ける |
| embed (option token × decision token, bilinear) | 0.906 | 0.048 | 可変 choice に対応 |
| pooling: last / decision token / mean / attention | 0.906 / 0.904 / **0.916** / **0.917** | 0.048 / 0.035 / 0.021 / 0.029 | mean が安価で最良 |

glaive tool routing (選択肢がサンプル毎に違う): embed 0.952 / linear 0.979。train に無いツールだけのサンプル 313 件でも embed 0.943。
ただしこのデータは「ツール 1 個 vs none」が大半で、linear が「位置」で解けてしまう。可変 choice の真の試験には
6 タスク共有 head (§5) の方が適している。

## 4. EXP-004/005 soft label と損失 (4B head-only, massive_scenario_en)

| target | loss | acc | ECE(T) | KL to 27B teacher |
|---|---|---|---|---|
| gold hard | CE | **0.906** | 0.019 | 0.455 |
| 27B soft (teacher acc 0.874) | CE ≡ KL | 0.861 | 0.023 | **0.112** |
| 27B soft | KL+Brier / CE+Brier | 0.861 | 0.024 | 0.121 |
| 27B soft | Focal | 0.856 | 0.026 | 0.128 |
| 0.5 gold + 0.5 27B | CE | 0.894 | 0.017 | — |
| 4B+27B ensemble soft (acc 0.833) | KL | 0.847 | 0.020 | — |
| 0.5 gold + 0.5 ensemble | CE | 0.885 | 0.019 | — |
| gold hard | Brier 単独 (lr 1e-3 / 1e-2 / 5e-2) | 0.09〜0.14 | — | 学習が進まない |

- **H3 (soft > hard) はこの設定では不成立**。teacher (zero-shot 27B, 87.4%) が gold より弱いので、蒸留すると teacher の誤りを継承する
  (KL は下がるが acc は 4.5pt 落ちる)。soft label が効くのは teacher が gold 以上の精度を持つ場面 (人手ラベルがなく LLM だけが答えを持つ業務) に限られる。
  4B と 27B の分布を平均した ensemble はさらに悪化 (4B が 82% と弱い)。
- Brier 単独は softmax が飽和した初期化で勾配が消えて学習しない。CE/KL との複合なら問題なし。
- 校正 (ECE) は loss 間でほぼ差がなく、温度補正でどれも 0.02 前後に揃う → **temperature scaling がほぼ全て**。

## 5. 6 タスク共有 head (exp011, 4B 凍結, hard, embed / mlp)

| task | 単独 head | 共有 embed head | 共有 mlp head |
|---|---|---|---|
| massive_scenario_en (Choice 18) | 0.906 | 0.911 | 0.893 |
| massive_scenario_ja (Choice 18) | 0.898 | 0.900 | 0.877 |
| massive_intent_en (Choice 60) | 0.868 | 0.866 | 0.825 |
| tweet_offensive (Noul) | 0.816 | 0.806 | 0.816 |
| yelp_score (Score 5) | 0.673 | 0.654 | 0.656 |
| glaive_tools (可変 Choice) | 0.952 | 0.944 | 0.954 |

3 種類の primitive・言語 2 つ・可変選択肢を **1 つの 2.6M パラメータ head** で単独学習とほぼ同精度に処理できる。
一方、学習に無い質問 (API デモ: 「緊急対応が必要か」「怒っているか」「スタッフに通知すべきか」) では Noul の答えが
LLM と食い違う (department / fall 判定など Choice は一致)。**汎用 Decision Model にするには質問の多様性を持つ教師データ (Phase 2) が必須**で、
1 万件 × 6 タスクでは足りない。

## 6. §11 fallback strategy の実測 (exp012, massive_scenario_en)

Decision Model (4B mean pooling, 温度補正後) が答え、max probability がしきい値未満の質問だけ 27B に回す構成。

| しきい値 | LLM 呼び出し率 | 最終 accuracy | 平均 latency (試算) |
|---|---|---|---|
| 0 (DM 単独) | 0% | **0.9163** | 16.4 ms |
| 0.6 | 8.2% | 0.9156 | 22.5 ms |
| 0.8 | 15.1% | 0.9099 | 27.5 ms |
| 0.9 | 20.2% | 0.9062 | 31.3 ms |
| 1.0 (LLM 単独) | 100% | 0.8749 | 90.0 ms |

**fallback するほど accuracy が下がる。** 専用学習した Decision Model の方が zero-shot 27B より強いため、
難しいサンプルを弱い方へ送っていることになる。仕様書 §11/§19 の「System 1 が速く安く、System 2 が賢い」という前提は、
専用学習を入れた時点で成立しなくなる。fallback が有効なのは fallback 先が実際に強い場合 (学習データにないドメイン・質問) に限る。

代わりに同じ confidence を「自動実行の可否」に使う運用は明確に有効:

| | カバー率 | カバー範囲の accuracy |
|---|---|---|
| Decision Model conf ≥ 0.9 | 79.8% | **0.9819** |
| 27B LLM conf ≥ 0.9 | 84.2% | 0.9541 |
| Decision Model conf ≥ 0.95 | 74.8% | **0.9870** |
| 27B LLM conf ≥ 0.95 | 80.9% | 0.9643 |

確率の質の差が、そのまま「何割を自動処理できるか」の差になる。再現は `scripts/cascade_analysis.py`。

## 7. Go / No-Go (§26) 判定

| 基準 | 結果 |
|---|---|
| accuracy: LLM baseline との差 ≤ 3pt | ○ 4B head 0.906〜0.933 vs 27B zero-shot 0.875 (上回る) |
| latency: 3 倍以上高速 | ○ JSON 構造化出力比 4〜28 倍。× vLLM 単 token logprob 比では同等 (小モデル化・共有 forward で差をつける必要) |
| calibration: ECE ≤ 0.08 | ○ 補正前 0.02〜0.05、補正後 0.012〜0.02 |
| VRAM: baseline 以下 | ○ 4B bf16 ≈ 8GB (+ features)、vLLM は KV cache 分を確保 |

**Go**。ただし設計判断として:
1. 「LM head 除去」ではなく「小モデル + block-diagonal 共有 forward + 校正済み確率」を売りにする。
2. 蒸留は teacher が gold を超える領域だけで使う。gold があるなら hard label + LoRA が最良 (0.933)。
3. 未知の質問への汎化が次の課題。多様な質問を含む教師データ生成 (Phase 2 本格化) と、Noul 用の多様な二値質問が必要。
4. fallback は「confidence が低いから LLM へ」ではなく「confidence が高いものだけ自動実行、残りは人間へ」に設計し直す (§6)。
5. 推論実装を HF eager から CUDA Graph / torch.compile / FP8 (Phase 7) に移せば単問 5 ms 以下が見込める (vLLM が 4B で 14.5 ms を出している)。

## 未実施 / 制限

- top-layers / full fine-tuning (Step 3/4)、FP8・int8 (EXP-010)、TensorRT/vLLM custom model は未実施。
- LoRA の latency は adapter 未 merge の値 (merge すれば head-only と同じ 16 ms)。
- isotonic / Platt / vector scaling は実装済みだが表には temperature scaling のみ。
- Method B (repeated sampling) の teacher は実装済み・未実行 (Method A logprob で分布が直接取れるため)。
- glaive の train/test ツール分離は不完全 (先頭ツール名のハッシュで分割、重複 164 ツール)。

## 8. §17 VLM Decision Model の feasibility (exp014, Qwen3-VL-4B-Instruct)

テキスト版の構造 (選択肢行末ベクトル × Decision 位置ベクトル) は VLM にそのまま載る。
画像は言語モデルと同じ系列に image token として並ぶだけなので、head も損失も学習コードも変更不要。実測:

- **zero-shot で選択肢の確率が出る**: 合成画像で shape 0.997 / color 0.996 と正しく判定 (LM head 経由、選択肢トークンの softmax)
- **1280x720 の 1 フレーム = 880 image token**。テキスト state (90 token) の約 10 倍で、state 共有の価値が跳ね上がる

1 フレームに N 個の判断を出すまでの端から端まで (画像前処理・vision encoder 込み、bf16、HF 実装):

| 質問数 | naive (毎回画像を再エンコード) | 画像エンコードを共有 | 倍率 |
|---|---|---|---|
| 1 | 61 ms | 82 ms | 0.7x |
| 2 | 119 ms | 86 ms | 1.4x |
| 4 | 233 ms | 84 ms | 2.8x |
| 8 | 461 ms | 89 ms | **5.2x** |
| 16 | 915 ms | 104 ms | **8.8x** |

質問 1 個だけなら共有はむしろ損 (prefill を 2 回に分けるため)。2 個以上で逆転し、
**16 個の判断を 1 フレームあたり 104 ms** で出せる。テキストの場合 (8 質問で 1.5 倍) より画像の方が共有の効きが大きい。

未検証: 実画像データでの head 学習・精度、vision encoder を凍結したままの転移、
mean pooling が画像トークンに引きずられる問題 (画像 880 / テキスト 19 token なので `last` か `decision` pooling が必要)、
block-diagonal attention の VLM 版 (Qwen3-VL は M-RoPE なので position_ids を 3 次元で組み直す必要がある)。
再現は `scripts/vlm_probe.py`。

## 9. 動画クリップでの検証 (exp015, Qwen3-VL-4B-Instruct)

### 9.1 フレームサンプリングの落とし穴 (最重要)

`transformers` の既定の video 前処理 (video_metadata を渡さない場合) は、**何枚渡しても video token 440 に固定**される。
16 フレームのクリップ中、大きな赤い四角が 1 枚だけ映る合成動画で「赤い四角があるか」を聞いた結果:

| 投げ方 | video token | 異常が 1 枚だけ | 全枚に異常 |
|---|---|---|---|
| 4 枚を video として | 440 | **P=0.947** | 0.998 |
| 8 枚を video として | 440 | **P=0.001** | 0.998 |
| 16 枚を video として | 440 | **P=0.001** | 0.998 |
| 32 枚を video として | 440 | **P=0.001** | 0.998 |
| 16 枚を image の連番として | 3591 | P=0.905 | — |

8 枚以上を 1 つの video として投げると、間引かれて**一瞬の事象が完全に消える**。しかも出力は「わからない」ではなく
**P=0.001 で自信を持って「異常なし」**。異常検知として最悪の壊れ方をする。
4 枚中 1 枚 (クリップの 25%) なら検出できるので、原因は「短い事象が時間方向に薄まる」ではなく「フレームが落ちている」。

対策: `video_metadata` で fps / duration を明示する、サンプリングを自前で行う、重要区間は image の連番で投げる。
ただし image 連番はトークンが 7 倍 (440 → 3591) になるのでコストと精度のトレードオフになる。
**なお本結果は transformers 実装の既定動作であり、vLLM / qwen-vl-utils や他モデル (Qwen3.8-27B 等) では挙動が異なりうる。
既存パイプラインは「渡した枚数ぶんのトークンが実際に増えているか」を必ず確認すること。**

### 9.2 1 クリップに複数の判断を出す (16 フレーム video, 端から端まで, p50)

| 質問数 | naive (毎回クリップを再エンコード) | クリップのエンコードを共有 | 倍率 |
|---|---|---|---|
| 1 | 59 ms | 75 ms | 0.8x |
| 4 | 165 ms | 77 ms | 2.2x |
| 8 | 307 ms | 78 ms | 3.9x |
| 16 | 592 ms | 89 ms | **6.7x** |

「危険行為があるか」「どの行為か」「誰が」「どの程度か」「通報すべきか」…と 16 項目を聞いても
1 クリップ 89 ms。**判断の数を増やすコストがほぼゼロ**になるのが動画での最大の利点。
逆に質問が 1 個だけならクリップ共有は損 (prefill が 2 回に分かれるため)。

未検証: 実映像での行為分類の精度、head を学習した場合の小型化余地、時系列を保った pooling。
再現は `scripts/video_probe.py`。

## 10. 「max_tokens=1 にするだけ」との差 (exp016, Qwen3-4B)

質問 1 問に対し 4 条件を比較 (massive_scenario_en / massive_intent_en の test から抽出)。

| 条件 | 18 択・丁寧なプロンプト | 60 択 (ラベルが 2 桁) | 18 択・素朴なプロンプト |
|---|---|---|---|
| max_tokens=1 のみ | parse 100% / acc 0.835 | parse 98.6% / **acc 0.100** | **parse 0%** / acc 0 |
| max_tokens=4 のみ | — | parse 95.4% / acc 0.734 | parse 0% / acc 0 |
| max_tokens=8 のみ | parse 100% / acc 0.835 | parse 95.0% / acc 0.732 | parse 0% / acc 0 |
| guided choice | parse 100% / acc 0.835 | parse 100% / acc 0.762 | parse 100% / acc 0.476 |
| guided choice + logprobs | 同上 + **確率分布** | 同上 + 確率分布 | 同上 + 確率分布 |

結論:

1. **選択肢が 26 以下 + 「ラベルだけで答えろ」と書いたプロンプトなら、`max_tokens=1` だけでほぼ同じ**。
   Qwen3-4B は制約なしでも 100% 妥当なラベルを返し、accuracy も guided choice と完全に一致する。
2. **崩れるのは 2 ケース**。(a) 選択肢が 27 個以上でラベルが複数トークンになると `max_tokens=1` は答えを途中で切る
   (「21」→「2」) ため accuracy が 0.76 → 0.10 に崩壊する。(b) プロンプトが素朴だと文章を書き始めて parse 率 0%。
3. **本質的に増えるのは確率分布だけ**。`max_tokens=1` で得られるのは argmax の 1 個。
   `logprobs` を付けて初めて「配送 91% / 請求 2% / 返品 5%」が 1 forward で手に入る。
   ただしこの確率は zero-shot では校正されていない (§1 の通り ECE 0.17)。校正には学習か温度補正が要る。

再現は `scripts/ablate_decoding.py`。

## 11. OCR 的タスク (exp017, Qwen3-VL-4B, 合成ナンバープレート)

「選択肢を列挙できるか」で結果が正反対になる。

**(1) 開いた転記を桁ごとの Choice に分解する → 失敗**

| 条件 | 1 桁目 | 2 桁目 | 3 桁目 | 4 桁目 | 正答 |
|---|---|---|---|---|---|
| 鮮明 | 1 (0.74) | 1 (0.45) | 2 (0.27) | 8 (0.32) | 1/4 |
| ぼかし 1.5px | 1 (0.75) | 2 (0.26) | 8 (0.30) | 8 (0.28) | 1/4 |

「左から N 桁目」という位置指定の問いが単一 forward では解けない。ただし確信度は 0.25〜0.45 と低く、
**間違ってはいるが自信は持っていない** (正直な壊れ方)。桁分解による転記は素直には成立しない。

**(2) 閉じた選択 (登録済み 5 台から選ぶ) → 高精度**

5 台すべて正しく識別 (0.994〜0.996)。選択肢に無いプレート (11-11) を見せると:

| 選択肢 | 出力 | 確信度 |
|---|---|---|
| 登録 5 台のみ | 12-34 (誤り) | **0.427** ← 確信度が崩れる |
| 登録 5 台 + "none of these" | **none of these** (正しく棄却) | 0.906 |

**(3) 劣化耐性 (正解 56-78、none あり)**

| 条件 | 出力 |
|---|---|
| 鮮明 / ぼかし 8px / 1/8 縮小 / 1/16 縮小 | 56-78 を正しく識別 (0.99) |
| ぼかし 20px / 1/32 縮小 | none of these (棄却) |

**読めなくなると誤答ではなく棄却に倒れる**。OCR の実運用で一番効く性質。
必ず "none / 該当なし" を選択肢に入れること。入れないと低確信の誤答になる。

設計上の含意: 住所・車番のような「候補が有限」のタスクは Choice に落とせる。
選択肢が 255 を超える場合 (市区町村 ~1900 等) は階層に分解する (都道府県 47 → その県内の市区町村)。
階層化は選択肢がサンプルごとに変わるので embed head が必要 (§3 の glaive と同じ形)。

未検証: 実写プレート・実際の住所表記での精度、フォント/角度/照明の変動、日本語文字種。
再現は `scripts/ocr_probe.py`。

## 12. Qwen3.8-27B をバックボーンにする場合 (exp018)

社内で既に使っている Qwen3.8-27B でこの方式が成立するかを実機で検証した。

### 12.1 成立する

- **マルチモーダル対応済み**: 視覚エンコーダ 461M + 言語モデル 25.62B。`model.visual` / `model.language_model` に分かれており、
  テキスト版と同じく LM head を外して hidden state を取り出せる (1280x720 の state = 944 token, hidden 5120)
- **視覚エンコーダは「大きい方」**: hidden 1152 / depth 27 / intermediate 4304 で、Qwen3-VL-8B/32B と同一仕様。
  Qwen3-VL-2B/4B の小型エンコーダ (hidden 1024 / depth 24) より上位
- zero-shot の選択肢確率も取得できる

### 12.2 小型版が系列内に無い

公開されているのは `Qwen3.8-27B` / `Qwen3.8-2.4T-A95B` (MoE) / `Qwen3.8-Flash-Next` のみ。
0.6B〜4B 級の dense モデルが存在しないため、「小型化して蒸留」は系列内で完結しない。

`Qwen3.8-Flash-Next` は MoE (hidden 2560 / 48 層 / 512 experts のうち 10 active) で、
**視覚エンコーダは 27B と同じ大型 (hidden 1152 / depth 27, out 2560)**。系列内で軽量化を狙うならこれが候補。

### 12.3 共有 forward には 2 箇所の実装上の罠がある (対処済み)

**(1) ハイブリッド構造**: 64 層中 16 層のみ full attention で、残り 48 層は線形アテンション (`Qwen3_5GatedDeltaNet`)。
transformers の `LinearAttentionLayer` には `batch_repeat_interleave` が実装されておらず、
state を 1 回 prefill して複数質問で共有する方式がそのままでは動かない。
線形アテンション層の状態は per-token の KV ではなく `conv_states` / `recurrent_states` という漸化状態なので、
batch 次元に複製すれば同じことができる → `models/hybrid_cache.py` の `repeat_cache_()`。

**また block-diagonal attention は Qwen3.8 では使えない**。線形アテンション層は任意の attention mask を取れないため。
KV 共有方式のみが有効 (テキスト版では block_diag が最速だったので、モデルにより最適手が変わる)。

**(2) M-RoPE の位置圧縮**: 画像トークンは位置番号を共有するため、**944 トークンの state が位置番号 99 で終わる**。
質問トークンの `position_ids` を素朴に `S + arange(L)` (= 944, 945, ...) にすると rope がずれる。
正しくは state 末尾の位置番号 + 1 から始める → `question_position_ids()`。

修正の効果 (naive との確率の最大差、8 質問):

| | 最大差 |
|---|---|
| 位置修正なし | 0.154 |
| 位置修正あり | **0.030** |

残差 0.03 は bf16 と batch 形状による数値差。head を学習する場合は推論時と同じ共有 forward で学習すれば影響しない。

### 12.4 速度 (1280x720 の 1 フレーム, bf16)

| 質問数 | naive | 共有 | 倍率 |
|---|---|---|---|
| 1 | 298 ms | 355 ms | 0.8x |
| 4 | 1164 ms | 370 ms | 3.2x |
| 8 | 2322 ms | 392 ms | 5.9x |
| 16 | 4647 ms | 450 ms | **10.3x** |

**注意: この絶対値は `causal_conv1d` / `flash-linear-attention` 未導入で、線形アテンションが
reference PyTorch 実装にフォールバックした状態の値**。専用カーネル導入で大幅に改善する見込み。倍率の傾向は有効。

## 13. 小型化: 視覚エンコーダ単体版と小型 VLM 版 (exp019/020, FairFace)

実データ (FairFace, 顔画像 224x224) で「年齢 9 区分」「性別」の 2 質問を学習。
train 86,744 (VLM 版は 20,000) / valid 5,477 / test 5,477 (VLM 版とゼロショットは 1,000〜2,000 で評価)。

| 構成 | 年齢 9択 acc | 年齢 ECE (補正後) | 性別 acc | 性別 ECE (補正後) | 1枚あたり | VRAM |
|---|---|---|---|---|---|---|
| Qwen3-VL-2B ゼロショット | **0.034** | 0.370 | 0.943 | 0.055 | 20.7 ms | — |
| Qwen3.8-27B ゼロショット | 0.522 | 0.051 | 0.941 | 0.035 | 97.9 ms | — |
| **視覚エンコーダ単体 + head** | 0.597 | 0.015 → 0.031 | 0.946 | **0.007** → 0.008 | **8.5 ms** | **2.15 GB** |
| **小型VLM (2B 凍結) + embed head** | **0.607** | 0.029 → **0.016** | **0.957** | 0.020 → 0.020 | 32.0 ms (2質問同時) | 4.13 GB |

パラメータ:

| 構成 | 凍結 | 学習 |
|---|---|---|
| 視覚エンコーダ単体 | 407M | **26.6K** (LayerNorm + Linear×2) |
| 小型VLM | 2.13B | 2.10M (embed head) |

学習時間: 視覚単体は特徴抽出 276 秒 + head 学習 **5 秒**。VLM 版は特徴抽出 799 秒 + head 学習。

### 読み取れること

1. **27B ゼロショットを、407M の凍結エンコーダ + 2.7 万パラメータの head が上回る** (年齢 0.522 → 0.597、性別 0.941 → 0.946)。
   しかも 11.5 倍速く、VRAM 2.15GB。テキストで確認した「読解力は足りている、答え方の訓練だけが足りない」が画像でも再現した。
2. **ゼロショットの小型モデルは順序尺度で壊滅する**。Qwen3-VL-2B の年齢はランダム (1/9 = 0.111) を大きく下回る 0.034 で、
   しかも自信 0.35 の予測が 0.7% しか当たらない (ECE 0.370)。**学習なしの小型 VLM は使えない**。
   一方、同じモデルの中間表現を使って head を学習すると 0.607 まで上がる。**差は「見えているか」ではなく「答え方」**。
3. 視覚単体版と小型VLM版の精度差はわずか (年齢 +1.0pt、性別 +1.1pt)。
   VLM 版は自然言語で質問できる代わりに、**パラメータ 5.2 倍・レイテンシ 3.8 倍・VRAM 1.9 倍**。
   質問が固定なら視覚単体版で十分。
4. 校正はどちらも良好 (ECE 0.007〜0.031)。温度補正は VLM 版の年齢でのみ効いた (0.029 → 0.016)。

参考: FairFace 論文の教師あり ResNet34 は年齢 9 区分で約 0.595、性別で約 0.945。
**凍結エンコーダ + 2.7 万パラメータの線形 head が、専用に学習した CNN と同等**。

再現: `training/train_vision.py` / `training/train_vlm.py` / `scripts/vision_zeroshot.py`。
API は `api/vision_server.py` (`POST /v1/vision/decision`、画像を multipart か base64 で受ける)。

## 14. 全身 (歩行者) での年代・性別 (exp021, PA-100K)

顔だけでなく監視カメラの全身切り出しで同じことをやる。PA-100K は実際の監視カメラの歩行者画像
(中央値 74x202 px、縦横比 2.7) 10 万枚。train 80,000 / valid 10,000 / test 10,000。
性別・年代に加え、同じ 1 回の画像エンコードで「向き」「荷物の有無」も同時に判定させた。

数値は accuracy / macro F1:

| 構成 | 性別 | 年代 3区分 | 向き 3択 | 荷物 | 1枚 p50 |
|---|---|---|---|---|---|
| Qwen3-VL-2B ゼロショット | 0.947 / 0.942 | **0.010 / 0.096** | 0.673 / 0.668 | 0.804 / 0.797 | 20.5 ms |
| Qwen3.8-27B ゼロショット | 0.942 / 0.937 | 0.949 / 0.491 | **0.831 / 0.833** | 0.599 / 0.566 | 80.4 ms |
| 視覚単体 + head (通常学習) | 0.935 / 0.932 | 0.967 / 0.432 | 0.820 / 0.824 | 0.832 / 0.831 | **8.9 ms** |
| 視覚単体 + head (クラス重み) | 0.935 / 0.931 | 0.928 / **0.590** | 0.827 / 0.831 | 0.834 / 0.832 | **8.9 ms** |
| 小型VLM (2B) + embed head | **0.956 / 0.952** | 0.983 / 0.508 | 0.812 / 0.816 | **0.853 / 0.851** | 34.7 ms (4質問同時) |

### 年代は accuracy で評価してはいけない

test の年代分布は 18歳未満 92 / 成人 9,650 / 60歳以上 258 と極端に偏る (多数派ベースライン 0.965)。
「全部成人」と答えるだけで accuracy 0.965 になるため、**accuracy はモデルの良否を全く反映しない**。

クラス別再現率:

| 学習 | 18歳未満 (n=92) | 成人 (n=9650) | 60歳以上 (n=258) | macro F1 |
|---|---|---|---|---|
| 通常学習 | 0.098 | 0.999 | 0.081 | 0.432 |
| **クラス重み付き損失** | **0.652** | 0.934 | **0.795** | **0.590** |

通常学習の head は事前分布を覚えて「成人」としか答えない。クラス頻度の逆数で重み付けするだけで
少数クラスの再現率が 0.098 → 0.652、0.081 → 0.795 に上がる。
**全身の外観に年代の信号は存在する。問題はモデルではなくデータの偏りと損失の設計だった。**
(accuracy は 0.967 → 0.928 に下がるが、これは「成人と答えるだけ」を捨てた代償で、実用上は正しい方向)

### 顔 (FairFace) との比較

| | 顔 224x224 | 全身 74x202 |
|---|---|---|
| 性別 | 0.946 | 0.935 |
| 年代 | 9区分 0.597 | 3区分 macro F1 0.590 |

**性別は全身でもほぼ落ちない (-1.1pt)**。年代は顔なら 9 区分できるが、全身では 3 区分でようやく実用域。
監視カメラ用途では「子供 / 成人 / 高齢者」程度の粒度が現実的。

### その他の読み取り

1. **ゼロショットでも十分な項目がある**。性別は Qwen3-VL-2B のゼロショットが 0.947 で、学習した視覚単体版 (0.935) を上回る。
   向きも 27B ゼロショットが最良 (0.833)。**全部を学習させる必要はない**。
2. **学習が決定的に効く項目もある**。荷物の有無は 27B ゼロショットが 0.599/0.566 なのに対し学習版は 0.832〜0.853。
   年代も少数クラスはゼロショットでは全く取れない。
3. **小型 VLM (2B) が 27B ゼロショットを 4 項目中 3 項目で上回る**。性別 +1.4pt、荷物 +25pt、年代 macro F1 +1.7pt。向きのみ -1.7pt。
4. 視覚単体版は 27B の **9 倍速く (8.9ms vs 80.4ms)**、学習パラメータは 2.5 万個。

### 実装上の注意

歩行者画像はサイズがばらばら (50x102 〜 323x714) なので、パッチ数が画像ごとに異なる。
`image_grid_thw` で分割してから画像ごとにプーリングする必要がある (`models/vision_model.py`)。
バッチ化による特徴の差はコサイン 1.0 相当で無視できる。

再現: `scripts/prepare_pa100k.py` → `training/train_vision.py --balanced` / `training/train_vlm.py`。
