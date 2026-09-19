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
| `openvons/jevpick/candidates/` | 候補メニュー: `ngram` (prompt / 出力の n-gram 索引)、`tool_schema` (ツール定義の展開、Qwen3 JSON / Qwen3.5 XML 両形式)、`repository` (同 repo のコード)、`grammar` / `macro_copy` (定型)、`mtp_qwen35` (Qwen3.5 系同梱 MTP head の transformers 実装) |
| `openvons/jevpick/scorer/model.py` | JevPick 本体 (`BlockScorer`)。hidden state + 候補 token 列 → 各候補の期待受理長。5〜10M パラメータ、学習 1〜2 分 |
| `openvons/jevpick/runtime/verifier.py` | greedy の一括検算と KV cache の巻き戻し |
| `openvons/jevpick/data/` | prompt 作成、greedy trace の収集、hidden state / MTP 候補の抽出 (bf16 / FP8 / NF4) |
| `experiments/jevpick/phase1_oracle/` | 候補メニューの上限値 (oracle) 測定 |
| `experiments/jevpick/phase2_scorer/train.py` | JevPick の学習と評価 (量子化モデルへの転移評価つき) |
| `experiments/jevpick/phase4_runtime/` | 実測: 同一 verifier での比較 (`runtime_v2.py`)、vLLM (`bench_vllm.py`)、llama.cpp (`bench_llamacpp.py`)、DFlash 公式 (`bench_dflash.py`)、文脈長別コスト (`bench_ctx.py`) |

## 動かす

### 一通り試す (1 GPU、4-bit が既定、20〜40 分)

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[dev,jevpick]"   # torch は自分の CUDA に合う index から
scripts/jevpick_quickstart.sh 0 300 100      # GPU 0、Qwen3-4B を 4-bit (NF4) で、Tool Call 学習 300 件 / 評価 100 件
JEVPICK_MODEL=Qwen/Qwen3.8-27B JEVPICK_DRAFT=incoai/Qwen3.8-27B-DFlash2 scripts/jevpick_quickstart.sh 0 300 100   # 27B を 4-bit で
```

target を bitsandbytes の NF4 (4-bit) で読み、DFlash の公開 draft と一緒に Hugging Face から取得して、
prompt 作成 → greedy trace → hidden state と DFlash 候補の抽出 → 候補メニューの上限値 → JevPick の学習と評価 → 同一 verifier での速度比較、を順に回す。
bf16 で回すなら `JEVPICK_QUANT=none`。結果は `experiments/jevpick/phase1_oracle/results_quickstart_*.md`、`phase2_scorer/result_quickstart_*.json`、`phase4_runtime/runtime_quickstart_*.json`。

VRAM の実測 (ピーク、この repo の quickstart):

| モデル | 精度 | trace 収集 (batch 24) | hidden 抽出 | runtime 比較 | 動く GPU |
|---|---|---:|---:|---:|---|
| Qwen3-4B | NF4 | 4.8 GB | 4.3 GB | 4.2 GB | 8 GB 級から |
| Qwen3.8-27B | NF4 | 25.6 GB | 23.0 GB | 23.1 GB | **32 GB 級 (RTX 5090 など)**。24 GB なら `collect.py --batch 8` |
| Qwen3.8-27B | bf16 | ~60 GB | ~58 GB | ~58 GB | 80 GB 級 |
| Qwen3.8-Flash-Next | — | transformers では読めない (FP8 MoE の重み変換が未対応)。llama.cpp の GGUF + MoE の CPU オフロード (`--cpu-moe`) で 32 GB 級を狙う → `phase4_runtime/bench_llamacpp.py --extra "--cpu-moe"`。実測は準備中 | | | |

本文の数字は N=3000/500 (Tool Call) で出したもの。N=80/40 の quickstart では選ぶ精度が 52 → 67% (4B)、55 → 64% (27B) と、学習データが少ない分だけ低く出る。

- データ置き場は環境変数 `JEVPICK_DATA` (既定 `state/jevpick`)。hidden state が大半で、N=3000 だと ~15 GB
- 量子化は `collect.py` / `extract.py` / `runtime_v2.py` の `--quant nf4` (bitsandbytes) と `--quant fp8` (FineGrainedFP8、`kernels==0.16.0`)。bf16 で学習した JevPick は FP8 / NF4 の hidden state に転移する (−0.5 / −3pt)
- 同梱 MTP head (Qwen3.5 系): `data/extract_mtp.py` で候補を抽出、`runtime_v2.py --mtp` で draft source に使う
- vLLM / llama.cpp の比較 (`phase4_runtime/bench_vllm.py`, `bench_llamacpp.py`) は別 venv / 別ビルドが前提。スクリプト冒頭に手順

### 個別に

```bash
.venv/bin/python openvons/jevpick/data/build_prompts_v2.py --toolcall-train 3000 --toolcall-test 500        # + repo-level Python (site-packages の実 repo)
CUDA_VISIBLE_DEVICES=0 .venv/bin/python openvons/jevpick/data/collect.py --model Qwen/Qwen3-4B --no-think --prompts $JEVPICK_DATA/prompts_v2.jsonl --out $JEVPICK_DATA/traces_v2_Qwen3-4B.jsonl
CUDA_VISIBLE_DEVICES=0 .venv/bin/python openvons/jevpick/data/extract.py --traces $JEVPICK_DATA/traces_v2_Qwen3-4B.jsonl --out $JEVPICK_DATA/extract_v2_Qwen3-4B_0   # --shard i/n で GPU 分割
.venv/bin/python experiments/jevpick/phase1_oracle/run_v2.py && .venv/bin/python experiments/jevpick/phase1_oracle/report_v2.py
CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/jevpick/phase2_scorer/train.py --domain toolcall --source union --L 8 --layer 3 --save $JEVPICK_DATA/jevpick.pt
CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/jevpick/phase4_runtime/runtime_v2.py --domain toolcall --modes baseline,prior,scorer,dflash,union,hybrid --scorer $JEVPICK_DATA/jevpick.pt
```

## 用語

| 用語 | 意味 |
|---|---|
| 先読み (speculative decoding) | 数語まとめて仮置きし、本体モデルに一括で検算させる高速化。仮置きが外れても捨てるだけで出力は変わらない |
| 候補メニュー / source | 仮置きの候補と、その出どころ |
| 当たり幅 / 受理長 | 1 回の検算で何語進めたか |
| 上限値 (oracle) | メニューに正解があれば必ず選べたと仮定したときの当たり幅。候補メニューの天井 |
| prior | 学習なしの単純ルールで選ぶ baseline (出現頻度順、MTP / DFlash の第一候補) |
| transformers | Hugging Face 社の推論ライブラリ (Python)。本文の速度比較で「簡易実装」と呼ぶもの。モデル置き場の Hugging Face Hub とは別 |
| vLLM / llama.cpp | 製品級の推論エンジン。kernel が最適化されており絶対速度が出る |
| MTP | モデルに同梱された「次の次」を予測する小さな head (Qwen3.5 系) |
| DFlash / DFlash2 | 数語をまとめて生成する学習済みの予測モデル (z-lab)。ブロック拡散型 |
