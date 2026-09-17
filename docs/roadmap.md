# 公開までの手順と残課題 (2026-09-17 時点)

(プロジェクト名は openvons、旧称 open-Jev。以下の open-Jev はすべて openvons を指す)

## 済んだこと
- 統合 repo (openvons.core / lm / vision / voice / tts)、統合 venv (transformers 5.17)、ライセンス棚卸し (docs/licensing.md)
- デモ 2 本 (駅名で動く路線図 = 一般向け、道路カメラ監視 = 業務例)、共通デモサーバー (--app 差し替え)、PWA manifest
- Jev 互換 API (POST /v1/systemone、docs/jev_api.md)、規約上の注意の整理
- 実音声の収集口 (マイク発話の自動保存)、事前学習の小範囲対策 (声・言い方の自動増量、正則化)
- 小型化パイプライン (ReazonSpeech 100h に kana-whisper で疑似ラベル 62,046 発話 → whisper-small 2 層 decoder を蒸留)

## 公開前にやること (優先順)
1. **名前の決定**。「Jev」は TypeSafe AI の製品名。候補: noul / decidr / yubisashi など。決めたら repo・パッケージ・README を一括改名
   (パッケージ名 `jev` は内部名として残してもよいが、公開名と揃えた方が説明が楽)。
2. **実音声での再校正と実測**。デモに人が話した発話 (state/<app>/utts/) を集め、合成音声の校正値との差を見る。
   docs/voice_evaluation.md に「実音声」の節を足す。数十件 × 数人で十分。
3. **蒸留モデルの評価と配布**。検証 CER と、合成評価 (`scripts/eval_synthetic.py --asr-model <dir>`) で kana-whisper との差を出す。
   実用域なら HF に Apache-2.0 で公開 (ReazonSpeech で学習したモデルの Apache-2.0 配布は kotoba-whisper に先例)。
   モデルカードに疑似ラベルの出自 (kana-whisper, MIT) と学習データ (ReazonSpeech small, 再配布なし) を明記。
4. **ブラウザ内推論 (スマホ)** — 現状: PWA (サーバー推論) は 4 デモとも完了、蒸留 pilot も完了、ONNX 変換以降が未着手。transformers.js v4 (WebGPU、iOS 26 以降; それ以前は WASM) に蒸留モデルを ONNX 変換して載せ、
   候補採点 (decoder forward に強制トークン) を JS で実装。サーバー推論への自動フォールバックを付ける。
5. **英語 README とモデルカード**、CONTRIBUTING、Issue テンプレート。
6. **CI**: GPU 不要のテスト (tests/test_voice_core.py、core の校正・判断) を GitHub Actions で。
7. **テキスト / 画像側の整備**: scripts/lm_* の再現手順を README から辿れるようにし、公開データでの学習ジョブを 1 コマンドに。

## スマホ対応の現状 (2026-09-17)

| 段階 | 状態 |
|---|---|
| https で開ける (マイクの前提) | 完了。openvons.com のトンネル経由 |
| PWA (ホーム画面に追加、サーバー推論) | 完了。4 デモとも manifest + アイコン + テーマ色、狭幅レイアウト (指で押せるボタン、セーフエリア) |
| 端末のマイク入力 | 完了。サンプルレートを毎コールバック読む (機種でルートが変わると壊れる罠の対策) |
| 小型モデルの蒸留 | pilot 完了。whisper-small 2 層 decoder、147M、検証 kana CER 0.23 (docs/voice_small_model.md) |
| 小型モデルの品質 | 課題。自由認識は同等だが候補の尤度採点が弱い (PTZ 0.58)。ReazonSpeech medium (1,000h) + KL 蒸留 + decoder 4 層が次 |
| ONNX 変換 → transformers.js (端末内推論) | 未着手 |
| ネイティブアプリ (Capacitor) | 未着手。PWA で足りる間は不要 |

## 残課題 (技術)
- 短い固有名詞だけの小さな範囲では該当なしとの分離が甘い (山手線 30 駅: 文法外の誤受理 8%)。担体付きの言い方 (〜まで / 〜駅) を
  UI で促す、校正サンプルの増量、短候補への追加ペナルティのいずれかで詰める。
- 同じ読みの実体 (市役所前 ×8、上り/下り) は確率が割れて確認に回る。「どちらですか」と聞き返す曖昧性解消の仮説を state.py に足す。
- 重なった発話 (別の声 −12dB) は kana-whisper が背景側の語を書く。話者分離かマイク指向性で対処 (Aunvox の AuK 分離が使える)。
- 自分宛判定 (addressee) の前段。Aunvox の addressee head を openvons.voice に移植すれば、確認状態以外の誤受理も下がる。

## 運用メモ
- デモ起動: `JEV_APP=examples.stations.app scripts/serve_demo.sh start 2 8601` / `examples.road_cameras.app ... 8600`。
  公開は aunvox トンネル (~/.config/aunvox/cloudflared/config.yml、ingress 変更後は `docker restart aunvox-tunnel-local-1`)。
- LM API: `scripts/serve_lm_api.sh start 8410` (vLLM :8300 の Qwen3-4B を基準バックエンドに)。
- TTS: VOICEVOX (docker `voicevox`, :50021)。校正だけに使い、学習データには使わない。
- プロセス停止は pid ファイル経由。`pkill -f` はコマンド行に起動部分が含まれると自シェルを殺す (このプロジェクトで 4 回)。
