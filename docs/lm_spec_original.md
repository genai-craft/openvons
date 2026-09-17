# System-One / Jev型 Decision Model 再現実験仕様書

## 0. 文書の目的

本プロジェクトでは、TypeSafe AI の System One / Jev が示している以下の性質を、公開情報から再現可能な範囲で段階的に検証する。

- 自由文章を生成せず、`Bool / Choice / Score` のような有限個の意思決定を直接出力する
- 各選択肢について確率値を返す
- 同一の `state` に対して複数の質問を低レイテンシで処理する
- structured output の構文エラーを原理的に排除する
- 生成型LLMより高速・低コストで判断タスクを処理する
- 出力確率を calibration し、「0.8 と言ったものが概ね80%正しい」に近づける
- 将来的には、文章生成モデルとは異なる「Decision Model」として独立させる

この文書は、単なる調査ではなく、実装エージェントがそのまま開発・実験を開始できる仕様書とする。

---

# 1. ゴール

最終的なゴールは、以下のAPIをローカル環境で提供できるモデル・推論サーバーを構築することである。

```python
result = client.decide(
    state={
        "message": "商品がまだ届いていません",
        "customer": {...},
        "order": {...}
    },
    questions={
        "urgent": Noul(
            "緊急対応が必要か"
        ),

        "department": Choice({
            "shipping": "配送に関する問い合わせ",
            "billing": "請求・支払い",
            "returns": "返品・返金",
            "other": "その他"
        }),

        "severity": Score([
            "問題なし",
            "軽度",
            "重大"
        ])
    }
)
```

想定出力:

```json
{
  "urgent": {
    "probability": 0.12
  },
  "department": {
    "probabilities": {
      "shipping": 0.91,
      "billing": 0.02,
      "returns": 0.05,
      "other": 0.02
    },
    "choice": "shipping"
  },
  "severity": {
    "probabilities": [0.05, 0.80, 0.15],
    "score": 1.10
  }
}
```

---

# 2. 本プロジェクトで再現するもの / 再現しないもの

## 2.1 再現対象

以下は自前実装する。

1. System-One型API
2. `Noul / Choice / Score` primitive
3. constrained decision
4. 確率分布の返却
5. soft-label distillation
6. classification head
7. autoregressive generation を使わない推論
8. state encoding の共有
9. multi-question batch / parallel execution
10. calibration
11. latency / throughput / accuracy ベンチマーク
12. uncertainty に応じた fallback
13. OpenAI互換LLMとの比較
14. 3B〜8Bモデルでの実用性評価

## 2.2 再現対象外

以下は公開情報が足りないため完全再現を目標にしない。

- Jevの非公開モデル重み
- Jev固有の内部アーキテクチャ
- RLCDの厳密な学習アルゴリズム
- TypeSafe内部の学習データ
- TypeSafe独自の評価セット

これらは、同じ「性質」を実現する代替方式を設計する。

---

# 3. 最重要仮説

本プロジェクトでは以下の仮説を検証する。

## H1. 文章生成は意思決定タスクには不要

通常LLM:

```text
state
  ↓
Transformer
  ↓
LM Head
  ↓
50k〜150k vocabulary
  ↓
token生成
  ↓
JSON
  ↓
parse
```

Decision Model:

```text
state + question + choices
  ↓
Transformer
  ↓
Decision Head
  ↓
2〜255 logits
  ↓
softmax
  ↓
probability
```

仮説:

> 有限選択タスクでは、LM Head と autoregressive decoding を削除することで大幅な高速化が可能。

---

## H2. state encoding は複数質問間で共有できる

```text
                 ┌─ Q1
                 ├─ Q2
State Encoder ───┼─ Q3
                 ├─ Q4
                 └─ Q5
```

仮説:

> 質問数が増えても、state の prefill を毎回行わなければレイテンシ増加を抑えられる。

---

## H3. hard label より soft label の方が向いている

例:

```json
{
  "shipping": 0.72,
  "returns": 0.21,
  "billing": 0.05,
  "other": 0.02
}
```

仮説:

> Decision Model は「唯一の正解」より、大型LLM・複数サンプル・人間評価から得た probability distribution を蒸留した方が性能とcalibrationが良い。

---

## H4. calibration はモデル価値の主要部分になる

単なるaccuracyではなく、

```text
confidence = 0.8
```

の予測を大量に集めたとき、

```text
accuracy ≒ 0.8
```

になることを重視する。

---

# 4. 全体アーキテクチャ

最終形の候補:

```text
                      ┌──────────────────────┐
                      │ Application / Agent  │
                      └──────────┬───────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │ Decision API Server   │
                     │                       │
                     │ Noul / Choice / Score │
                     └──────────┬────────────┘
                                │
              ┌─────────────────┼──────────────────┐
              │                 │                  │
              ▼                 ▼                  ▼
       State tokenizer    Question encoder    Choice encoder
              │                 │                  │
              └─────────────────┼──────────────────┘
                                ▼
                     ┌──────────────────────┐
                     │ Transformer Backbone │
                     │ Qwen 3B〜8B          │
                     └──────────┬───────────┘
                                │
                  shared hidden representation
                                │
             ┌──────────────────┼──────────────────┐
             ▼                  ▼                  ▼
          Noul Head         Choice Head        Score Head
             │                  │                  │
         sigmoid             softmax            softmax
             │                  │                  │
             └──────────────────┼──────────────────┘
                                ▼
                         Calibration Layer
                                │
                                ▼
                        Probability Output
```

---

# 5. 開発フェーズ

---

# Phase 0: ベースライン構築

## 目的

「普通のLLMで同じAPI」をまず作る。

これが全実験の比較対象になる。

## 実装

OpenAI-compatible API を裏側に利用する。

候補:

- Qwen系 instruct model
- vLLM
- SGLang
- llama.cpp
- Ollama

### API

```http
POST /v1/decision
```

入力:

```json
{
  "state": {},
  "questions": []
}
```

内部ではLLMに structured output を要求する。

### 実験項目

- JSON Schema constrained decoding
- logprobs取得
- choice token probability
- temperature 0
- temperature > 0 sampling
- 5回 / 20回 sampling で分布推定

## Phase 0 出力

以下を記録する。

```text
accuracy
macro F1
latency p50
latency p95
tokens/sec
GPU memory
input tokens
output tokens
cost
ECE
Brier score
```

---

# Phase 1: Decision API互換層

## primitive

### Noul

Boolean decision。

```python
Noul(
    description="この注文は不正利用の可能性があるか"
)
```

出力:

```json
{
  "true": 0.83,
  "false": 0.17
}
```

---

### Choice

```python
Choice({
    "A": "問い合わせA",
    "B": "問い合わせB",
    "C": "問い合わせC"
})
```

出力:

```json
{
  "A": 0.7,
  "B": 0.2,
  "C": 0.1
}
```

制約:

```text
2 <= choices <= 255
```

---

### Score

```python
Score([
    "問題なし",
    "軽度",
    "中程度",
    "重大"
])
```

出力:

```json
{
  "probabilities": [0.05, 0.20, 0.55, 0.20],
  "expected_score": 1.90
}
```

計算:

```text
score = Σ i * p_i
```

---

# Phase 2: Teacher Dataset 作成

## 目的

生成モデルからDecision Modelへ知識蒸留する。

---

## 2.1 データ形式

JSONL:

```json
{
  "state": "...",
  "question": "...",
  "type": "choice",
  "choices": [
    {
      "id": "shipping",
      "description": "配送問題"
    },
    {
      "id": "billing",
      "description": "請求問題"
    }
  ],
  "target_probs": [
    0.91,
    0.09
  ]
}
```

---

## 2.2 Teacher生成方式

### Method A: 単一LLM logprob

可能なら各choiceのlogprobを直接取得する。

---

### Method B: repeated sampling

同じ質問を20回評価。

例:

```text
shipping 16
billing   3
returns   1
```

↓

```text
shipping 0.80
billing  0.15
returns  0.05
```

---

### Method C: ensemble

複数teacherを使用。

```text
Teacher A
Teacher B
Teacher C
Teacher D
```

最終target:

```text
mean(probabilities)
```

または、

```text
weighted mean
```

---

## 2.3 Human label

重要データだけ人間評価を追加する。

特に、

```text
teacher disagreement > threshold
```

のケースを優先して人間に回す。

Active Learning形式にする。

---

# Phase 3: Generation-less Model v0

## 目的

autoregressive generationを削除する。

---

## 3.1 Backbone

第一候補:

```text
Qwen 3B〜8B
```

条件:

- Hugging Face Transformers対応
- hidden states取得可能
- bf16対応
- FlashAttention対応
- classification fine-tuning容易

---

## 3.2 pooling

比較する。

### A

最後のtoken:

```python
h = hidden[:, -1]
```

### B

専用token:

```text
<DECISION>
```

### C

mean pooling

### D

attention pooling

---

## 3.3 Head

### Noul

```python
Linear(hidden_size, 2)
```

### Choice

固定最大数方式:

```python
Linear(hidden_size, 255)
```

未使用optionはmask。

---

## 3.4 より良いChoice方式

固定255分類より、

```text
state/question embedding
        ×
choice embedding
```

のスコアリング方式を優先的に検証する。

```python
score_i = dot(query_embedding, choice_embedding_i)
```

または

```python
score_i = MLP(
    concat(query_embedding, choice_embedding_i)
)
```

利点:

- 未知のchoiceにも対応できる
- option数可変
- choice description自体を意味的に理解できる

---

# Phase 4: 学習

## Loss

第一候補:

```text
KL divergence
```

teacherのsoft labelを使用。

```python
loss = KL(
    student_probs,
    teacher_probs
)
```

比較:

- Cross Entropy
- KL Divergence
- Brier Loss
- Focal Loss
- CE + Brier
- KL + Brier

---

## 学習段階

### Step 1

head only

backbone freeze

### Step 2

LoRA

### Step 3

top transformer layersのみ学習

### Step 4

full fine-tuning

各段階で性能・計算コスト比較。

---

# Phase 5: Probability Calibration

## 評価指標

### ECE

Expected Calibration Error

### Brier Score

```text
mean((p - y)^2)
```

### NLL

Negative Log Likelihood

### Reliability Diagram

confidence bin:

```text
0.0-0.1
0.1-0.2
...
0.9-1.0
```

について、

```text
平均confidence
実accuracy
```

を比較する。

---

## calibration手法

以下を比較する。

### Temperature Scaling

最初に必須。

### Platt Scaling

binary用。

### Isotonic Regression

### Dirichlet Calibration

### Vector Scaling

---

# Phase 6: Multi-question Optimization

ここがJev型の重要実験。

---

## v1: naive

質問ごとに完全forward。

```text
state + Q1
state + Q2
state + Q3
```

---

## v2: KV Cache共有

state部分のみprefill。

```text
State
 ↓
KV cache
 ↓
 ├ Q1
 ├ Q2
 └ Q3
```

計測:

```text
question_count = 1, 2, 4, 8, 16, 32, 64
```

---

## v3: batch questions

Q1〜Qnをbatch dimensionにまとめる。

---

## v4: Shared-state / Multi-query Attention

最終候補。

```text
State Tokens
    │
    ├─────────┐
    │         │
   Q1        Q2       Q3
    │         │        │
    ▼         ▼        ▼
decision1 decision2 decision3
```

attention mask:

```text
Q1 -> stateを参照可能
Q2 -> stateを参照可能
Q3 -> stateを参照可能

Q1 -> Q2 は参照不可
Q2 -> Q3 は参照不可
```

block-diagonal attention を実装する。

---

# Phase 7: Latency最適化

検証対象:

```text
bf16
fp16
fp8
int8
int4
```

その他:

- FlashAttention
- torch.compile
- CUDA Graph
- TensorRT-LLM
- vLLM custom model
- continuous batching

---

# 8. 最重要Benchmark

比較対象:

1. GPT系API
2. Claude系API
3. Qwen Instruct
4. Qwen structured output
5. Decision Model
6. Decision Model + calibration
7. Decision Model + shared-state parallel

---

## Benchmark A: Intent Classification

例:

```text
customer support
```

10〜50 class。

---

## Benchmark B: Moderation

```text
safe
spam
fraud
abuse
etc.
```

---

## Benchmark C: Ranking / Score

```text
0〜4
```

---

## Benchmark D: Tool Routing

agentが使用するtoolを選択。

```text
search
database
email
calendar
none
```

---

## Benchmark E: IoT / Sensor Decision

例:

```text
camera
LiDAR
temperature
motion
```

から

```text
normal
fall
person_down
intrusion
unknown
```

を判断。

これはローカルAI Appliance用途で重要。

---

# 9. ベンチマークデータ量

最低:

```text
train   50k
valid    5k
test    10k
```

推奨:

```text
100k〜1M
```

ただし最初は

```text
10k
```

程度でも仮説検証する。

---

# 10. 成功基準

## v0

LLMベースAPI互換。

成功:

```text
schema error = 0
```

---

## v1

Generation-less。

成功:

```text
accuracy >= baseline - 3%
latency <= baseline / 3
```

---

## v2

蒸留。

成功:

```text
accuracy >= baseline
```

---

## v3

calibration。

成功:

```text
ECE <= 0.05
```

---

## v4

multi-question。

成功:

8質問で

```text
latency < single_question_latency * 2
```

---

## v5

production candidate。

目標:

```text
p50 < 100 ms
p95 < 250 ms
```

短いstateの場合。

---

# 11. Fallback Strategy

Decision Modelだけですべて処理しない。

confidenceが低い場合:

```text
if max_probability > 0.9:
    execute

elif max_probability > 0.6:
    verify

else:
    fallback_to_llm
```

構成:

```text
                Decision Model
                     │
             confidence high
              │             │
             YES           NO
              │             │
          execute      Generative LLM
```

これにより、

- 平均latency低下
- APIコスト削減
- 難問のみ大型モデル

を実現する。

---

# 12. Repository構成

```text
decision-model/
│
├─ README.md
│
├─ docs/
│   ├─ architecture.md
│   ├─ experiments.md
│   └─ benchmark.md
│
├─ api/
│   ├─ server.py
│   ├─ schemas.py
│   └─ primitives.py
│
├─ models/
│   ├─ backbone.py
│   ├─ decision_model.py
│   ├─ heads.py
│   └─ pooling.py
│
├─ training/
│   ├─ train.py
│   ├─ losses.py
│   ├─ dataset.py
│   └─ distillation.py
│
├─ calibration/
│   ├─ temperature.py
│   ├─ isotonic.py
│   └─ metrics.py
│
├─ teacher/
│   ├─ openai.py
│   ├─ anthropic.py
│   ├─ local_llm.py
│   └─ ensemble.py
│
├─ benchmark/
│   ├─ run.py
│   ├─ latency.py
│   ├─ accuracy.py
│   └─ calibration.py
│
├─ data/
│   ├─ raw/
│   ├─ generated/
│   └─ processed/
│
└─ scripts/
    ├─ generate_teacher_data.py
    ├─ train_head.py
    ├─ train_lora.py
    └─ benchmark_all.py
```

---

# 13. 実験ログ

全実験を以下形式で保存する。

```json
{
  "experiment": "exp_023",
  "model": "qwen-3b",
  "pooling": "decision_token",
  "head": "choice_embedding",
  "loss": "kl+brier",
  "quantization": "bf16",
  "dataset": "support_v2",
  "accuracy": 0.914,
  "ece": 0.032,
  "brier": 0.081,
  "latency_p50_ms": 42,
  "latency_p95_ms": 63,
  "gpu": "RTX 6000 Pro Blackwell"
}
```

Weights & BiasesまたはMLflow使用可。

---

# 14. 最初に実行する実験

優先度順。

## EXP-001

普通のQwen structured output。

目的:

baseline取得。

---

## EXP-002

Qwen hidden state + linear classifier。

目的:

生成除去の効果を見る。

---

## EXP-003

linear head vs choice embedding。

目的:

可変choiceへのgeneralizationを見る。

---

## EXP-004

hard label vs soft label。

---

## EXP-005

CE vs KL vs Brier。

---

## EXP-006

temperature scaling。

---

## EXP-007

1 / 4 / 8 / 16 questions batch。

---

## EXP-008

state KV cache reuse。

---

## EXP-009

LoRA tuning。

---

## EXP-010

FP8。

RTX 6000 Pro Blackwellで測定。

---

# 15. 特に調べるべきポイント

以下は結果次第で設計が大きく変わる。

## A. 本当に3B必要か

比較:

```text
0.5B
1.5B
3B
7B
```

意思決定だけなら小さいモデルで十分な可能性がある。

---

## B. state全体をTransformerに入れる必要があるか

長文stateの場合、

```text
state encoder
↓
compressed latent
↓
questions
```

という構造も検討する。

---

## C. questionを毎回自然言語で入れる必要があるか

productionでは頻出questionにIDを付け、

```text
question_id = 23
```

だけにできる可能性がある。

その場合さらに高速化できる。

---

## D. choice descriptionの事前embedding

choiceが固定なら、

```text
choice embedding
```

を事前計算できる。

---

# 16. 発展案: 固定業務専用Decision Model

汎用モデルではなく、

```text
customer support model
factory monitoring model
healthcare monitoring model
security model
```

のように用途別モデルを作る。

メリット:

- 小型化
- 高速化
- accuracy向上
- calibration容易
- edge GPU対応

---

# 17. 発展案: VLM Decision Model

将来的には

```text
Image
+
Sensor
+
Text
```

↓

```text
Decision
```

を直接出す。

例:

```text
camera image
LiDAR grid
motion history
```

↓

```text
fall probability = 0.91
```

自由文章は生成しない。

見守り、設備監視、交通量、異常検知などでは特に有効。

---

# 18. 発展案: Agent向けSystem-One Layer

通常Agent:

```text
LLM
↓
考える
↓
toolを決める
```

を、

```text
Decision Model
↓
tool routing
↓
confidence低い場合だけLLM
```

に置き換える。

例:

```json
{
  "search_web": 0.05,
  "search_db": 0.88,
  "ask_user": 0.04,
  "do_nothing": 0.03
}
```

ここは非常に大きな商用用途候補。

---

# 19. 研究として最も面白い部分

単なるstructured output cloneにしない。

本当に価値があるのは、

```text
Generative AI
```

から

```text
Decision AI
```

を分離すること。

つまり、

```text
System 1:
高速
安価
probabilistic
bounded output
大量実行可能

System 2:
生成LLM
高コスト
長いreasoning
難問対応
```

の2層構造にする。

```text
             Request
                │
                ▼
        ┌────────────────┐
        │ Decision Model │
        └───────┬────────┘
                │
       confidence low?
           │         │
          NO        YES
           │         │
           ▼         ▼
        Action      LLM
```

---

# 20. 最終成果物

このプロジェクト完了時には以下を作る。

### 1. Python SDK

```python
from decision_ai import Noul, Choice, Score
```

### 2. REST API

```text
POST /v1/decision
```

### 3. Local model

Hugging Face形式。

### 4. Benchmark

```text
accuracy
F1
ECE
Brier
latency
throughput
VRAM
```

### 5. Evaluation dashboard

モデルバージョン比較。

### 6. Teacher generation pipeline

### 7. Calibration pipeline

### 8. Agent fallback integration

---

# 21. 第一マイルストーン

まず以下だけを完成させる。

```text
Qwen 3B
+
Choice / Noul
+
soft-label distillation
+
linear / choice embedding head
+
temperature calibration
```

対象データ:

```text
10,000〜50,000 samples
```

測定:

```text
accuracy
ECE
Brier
latency
```

これで、

> 「文章生成を捨てたDecision Modelは本当にLLMより速く、十分賢いのか」

を判定する。

---

# 22. 第二マイルストーン

第一段階で有望なら、

```text
state shared encoding
+
multi-question batch
+
KV reuse
+
FP8
```

を実装。

ここで、

> 質問数が増えてもlatencyがほぼ増えない

というJev型特性にどこまで近づけるかを見る。

---

# 23. 第三マイルストーン

次に用途特化。

優先候補:

1. Agent tool routing
2. サポート問い合わせ分類
3. IoT / 見守り判断
4. VLM異常判定
5. 業務承認・リスクスコア

この段階から、汎用研究ではなく製品化候補として評価する。

---

# 24. 実装エージェントへの指示

実装時は、最初から巨大な独自アーキテクチャを作らないこと。

順序:

```text
baseline
↓
classifier
↓
distillation
↓
calibration
↓
parallelization
↓
architecture optimization
```

各改善は必ずablation testを行う。

「速くなった」「精度が上がった」という感覚ではなく、

```text
baselineとの差分
```

を数値で記録する。

未知のJev内部構造を想像してコピーすることより、

> 必要な性質を最小構成で再現する

ことを優先する。

---

# 25. 最初の実装タスク

実装エージェントは以下から着手する。

```text
TASK-001
FastAPIで /v1/decision を作成

TASK-002
Noul / Choice / Score schemaを実装

TASK-003
OpenAI-compatible LLM backendを作る

TASK-004
benchmark CLIを作る

TASK-005
Qwen backboneからhidden state取得

TASK-006
binary classification head

TASK-007
choice embedding head

TASK-008
teacher dataset generator

TASK-009
soft-label training

TASK-010
temperature calibration

TASK-011
latency benchmark

TASK-012
4/8/16 questions batch benchmark
```

TASK-012まで完了した時点で、一度アーキテクチャ判断を行う。

それ以前に独自attention kernel等へ進まない。

---

# 26. Go / No-Go 判定

第一フェーズの結果が以下なら継続。

```text
Accuracy:
LLM baselineとの差 <= 3pt

Latency:
3倍以上高速

Calibration:
ECE <= 0.08

VRAM:
baseline以下
```

以下の場合はモデル方式を再検討。

```text
3Bでもaccuracyが大幅低下
choice generalization不可
confidence calibration不能
```

その場合、

```text
small generative model
+
constrained decoding
+
KV reuse
```

の方が商用上有利な可能性も認める。

---

# 27. プロジェクトの判断基準

目標は「TypeSafe AIをコピーした」と言うことではない。

最終的に判断するのは以下。

> **有限個の意思決定を大量・高速・安価・確率付きで処理する専用AI層を、ローカル環境で成立させられるか。**

成立するなら、LLMの前段・後段に置く汎用コンポーネントとして価値がある。

特にローカルGPU、エッジAI、Agent、IoTでは、生成モデルそのものより実用性が高い可能性がある。
