# JevPick — 文章生成の先読みに「候補から選ぶ」を使う

openvons は「AI に文章を書かせず、有限の選択肢に確率で答えさせる」層です。JevPick はその考え方を、
LLM の文章生成そのものを速くする **先読み (speculative decoding)** に転用したものです。

## 一言で

- 次に出てきそうな数語の**候補メニュー**を、ツール定義・過去の出力・同梱の MTP head などから作る
- **JevPick** がモデル自身の内部状態を見て、メニューから 1 つ選ぶ (Jev の「状況 → 選択肢 → 選ぶ」と同じ形)
- 選んだ候補を本体モデルに一括で検算させ、当たった分だけ進める。**出力は 1 文字も変わらない**
- 当たりそうな候補が無ければ「先読みしない」と判断する

## 分かったこと (2026-09-19)

| | 結果 |
|---|---|
| ツール呼び出し (function calling) | 候補メニューの中に正解の続きが **90%** 入っている。JevPick は単純選択 64% → **88%** で当てる。予測モデルなしで **3.2〜4.8 倍** (Qwen3-4B / Qwen3.8-27B、4-bit 量子化でも)、出力完全一致 |
| 量子化 | bf16 で学習した JevPick は FP8 の内部状態でそのまま使える (−0.5pt)、4-bit (NF4) でも −3pt |
| 長い文脈 (16k) | 先読み手法 (DFlash・n-gram) は 16k 文脈で 1.0 倍以下に落ちる。JevPick を「先読みするか」の判断に使うと損失を止められる |
| MTP / DFlash2 がある場合 | 同梱 MTP はツール呼び出しで次 token を 99.7% 当て、4-bit 27B で 6 倍、Flash-Next で 5.8 倍。その上に JevPick を乗せた上積みは 4〜8% |
| コード補完 | 候補メニューが 3 割しか当たらず弱い。DFlash などの予測モデルの領分 |

詳細レポート: [phase1 (4B, 最小実験)](phase1_oracle_report.md) → [phase2 (4B, source 拡張・DFlash 併用・16k)](phase2_report.md) → [phase3 (27B, FP8/NF4/NVFP4/AWQ/GGUF, MTP, Flash-Next)](phase3_27b_report.md)。
全手法 × タスク × モデルの一覧: [summary_table.md](summary_table.md) (`experiments/jevpick/summarize.py` で再生成)。

## コードの場所

| 場所 | 中身 |
|---|---|
| `openvons/jevpick/candidates/` | 候補メニュー: `ngram` (prompt / 出力の n-gram 索引)、`tool_schema` (ツール定義の展開、Qwen3 JSON / Qwen3.5 XML 両形式)、`repository` (同 repo のコード)、`grammar` / `macro_copy` (定型)、`mtp_qwen35` (Qwen3.5 系同梱 MTP head の HF 実装) |
| `openvons/jevpick/scorer/model.py` | JevPick 本体 (`BlockScorer`)。hidden state + 候補 token 列 → 各候補の期待受理長。5〜10M パラメータ、学習 1〜2 分 |
| `openvons/jevpick/runtime/verifier.py` | greedy の一括検算と KV cache の巻き戻し |
| `openvons/jevpick/data/` | prompt 作成、greedy trace の収集、hidden state / MTP 候補の抽出 (bf16 / FP8 / NF4) |
| `experiments/jevpick/phase1_oracle/` | 候補メニューの上限値 (oracle) 測定 |
| `experiments/jevpick/phase2_scorer/train.py` | JevPick の学習と評価 (量子化モデルへの転移評価つき) |
| `experiments/jevpick/phase4_runtime/` | 実測: 同一 verifier での比較 (`runtime_v2.py`)、vLLM (`bench_vllm.py`)、llama.cpp (`bench_llamacpp.py`)、DFlash 公式 (`bench_dflash.py`)、文脈長別コスト (`bench_ctx.py`) |

## 動かす (Tool Call、Qwen3-4B の例)

```bash
.venv/bin/python openvons/jevpick/data/build_prompts_v2.py                       # prompt (glaive tool call + repo-level Python)
CUDA_VISIBLE_DEVICES=0 .venv/bin/python openvons/jevpick/data/collect.py --model Qwen/Qwen3-4B --no-think \
    --prompts /data/openvons/choice_spec/prompts_v2.jsonl --out /data/openvons/choice_spec/traces_v2_Qwen3-4B.jsonl
CUDA_VISIBLE_DEVICES=0 .venv/bin/python openvons/jevpick/data/extract.py --traces .../traces_v2_Qwen3-4B.jsonl --out .../extract_v2_Qwen3-4B_0   # hidden + DFlash 候補
.venv/bin/python experiments/jevpick/phase1_oracle/run_v2.py && .venv/bin/python experiments/jevpick/phase1_oracle/report_v2.py
CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/jevpick/phase2_scorer/train.py --domain toolcall --source finite --L 8 --save scorer.pt
CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/jevpick/phase4_runtime/runtime_v2.py --domain toolcall --modes baseline,scorer,dflash,union,hybrid --scorer scorer.pt
```

データは `/data/openvons/choice_spec/` 以下 (歴史的な名前のまま)。

## 用語

| 用語 | 意味 |
|---|---|
| 先読み (speculative decoding) | 数語まとめて仮置きし、本体モデルに一括で検算させる高速化。仮置きが外れても捨てるだけで出力は変わらない |
| 候補メニュー / source | 仮置きの候補と、その出どころ |
| 当たり幅 / 受理長 | 1 回の検算で何語進めたか |
| 上限値 (oracle) | メニューに正解があれば必ず選べたと仮定したときの当たり幅。候補メニューの天井 |
| prior | 学習なしの単純ルールで選ぶ baseline (出現頻度順、MTP / DFlash の第一候補) |
| MTP | モデルに同梱された「次の次」を予測する小さな head (Qwen3.5 系) |
| DFlash / DFlash2 | 数語をまとめて生成する学習済みの予測モデル (z-lab)。ブロック拡散型 |
