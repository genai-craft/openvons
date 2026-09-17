# TypeSafe Jev (System One API) の公開仕様と open-Jev の対応 (2026-09-17 調査)

出典は docs.typesafe.ai (ログイン不要、`llms.txt` に索引)、typesafe.ai/legal、公開 SDK (MIT)。
console.typesafe.ai (鍵・プレイグラウンド) はログインが要るが、仕様そのものは公開文書で足りる。

## 公開 API の要点

- 唯一の推論エンドポイント `POST https://api.typesafe.ai/v1/systemone`、`Authorization: Bearer $TYPESAFE_API_KEY`。
- request: `{"state": str|object|array, "model": "jev-latest", "questions": {name: {"type": "noul"|"choice"|"score", "instructions": str, "criteria": ...}}}`
  - noul の criteria は `{"true": ..., "false": ...}` (任意)、choice は `{id: description|null}` か配列、score は 2〜10 レベルの配列。
  - few-shot は criteria 内に文字列で埋める。専用アップロード API・fine-tune・feedback API は無い。
- response: `{"model": "jev-1.13.0", "answers": {name: {"type": "noul", "noul": p} | {"type": "choice", "choice": id, "confidence": c, "probabilities": {...}} | {"type": "score", "score": e, "confidence": c, "probabilities": {"0": p0, ...}, "legend": {...}}}, "usage": {"input_tokens": n, "output_tokens": 0}}`
  - confidence は「分布の集中度を 0〜1 に潰した値」とだけ定義 (式は非公開)。noul には無い。
  - 該当なし (abstain) の専用フィールドは無く、「other を選択肢に足せ」というガイドのみ。
- 制限: 1 リクエスト約 32k トークン、choice 2〜255、score 2〜10、250k tok/s・1,200 req/min。価格 $0.042/MTok 入力、出力無料。
- 主張: 70〜500ms E2E、質問を足してもレイテンシはほぼ増えない (state を 1 回だけ読む)、質問は互いに独立。
  これは open-Jev の block-diagonal / KV 共有 forward と同じ構造 (docs/lm_architecture.md)。
- テキスト専用 (画像・音声・動画は非対応、画像は「今後」)。重み非公開、self-host / on-device 無し。

## open-Jev の対応

`jev.lm.api.server` に `POST /v1/systemone` を追加した (`jev/lm/api/systemone.py`)。typesafe-sdk の
`TYPESAFE_BASE_URL` をこのサーバーに向ければ、同じコードで手元の Decision Model が答える。
confidence はここでは「最大確率 − 2 位」(margin) を採用。校正した確率は probabilities に出しているので、
しきい値運用は probabilities 側で行う。

open-Jev が公式に無いものとして持つ機能:
- 該当なし (none) を明示的な選択肢として校正に含める (jev.core.none_calibration)
- 判断ポリシー (実行 / 確認 / 棄却) と危険度 (jev.core.decision)
- 画像 (jev.vision) と音声 (jev.voice)
- 状態依存の選択肢集合、担当範囲、事前学習 (音声)
- 重みと学習コードの公開、on-device (蒸留した小型 kana モデル、進行中)

内部仕様 (docs/lm_spec_original.md) との差分は表記の違いが主: `/v1/decision` → `/v1/systemone`、
`description` → `instructions` + `criteria`、noul の `probability` → `noul`、score の legend/confidence 追加、
state が str|object|array、32k トークン上限、model フィールド。設計思想 (生成なし・state 共有・独立評価・校正) は一致。

## 規約上の注意 (法的助言ではない。原文を確認のこと)

Master Customer Agreement (typesafe.ai/legal/mca、2026-08-27) は API 顧客に対し、
「リバースエンジニアリング」「出力を使った蒸留・模倣モデルの学習」「類似・競合製品の開発」
「**サービスのベンチマークや性能情報の公開**」を禁じている。

open-Jev への含意:
1. Jev の API 出力を教師データ・校正データ・比較対象として**使わない**。open-Jev は公開情報と自前実装のみで作る。
2. Jev との性能比較を公開しない (evals.typesafe.ai の数字を引用するに留める)。
3. 名前: 「Jev」は同社の製品名。商標ポリシーの公開ページは無いが、製品名をそのまま含む「open-Jev」は
   少なくとも「TypeSafe 非公認」の明記が要る。既存の第三者実装は `qwen-rlcd`、`system-one-adapter-python` のような
   名前を使っている。別名 (例: **noul**、**decidr**、**yubisashi/指差し**) への変更を推奨。
4. 互換 API を提供すること自体は「顧客としての利用」ではないが、SDK の名前空間は使わず、自前の名前で公開する。
