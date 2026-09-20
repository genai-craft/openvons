# openvons (open-Jev) — 有限選択肢に確率で答える判断層

[English README](README.md)

名前は Jevons (限界効用の経済学者) から。open-Jev として始めたプロジェクトで、パッケージ名は `openvons`、`import jev` も互換で動く。

LLM / VLM / ASR に**文章を生成させる代わりに、有限の選択肢へ確率で直接答えさせる**。
「該当なし」も選択肢に入れ、校正した確率で **自動実行 / 確認 / 棄却** を分ける。

同じ主張をテキスト・画像・音声の 3 つの入口で実装している。

| 入口 | 何を選ぶか | 実装 | 実測 (docs/) |
|---|---|---|---|
| `openvons.lm` | 意図分類・ツール選択・スコア (Noul / Choice / Score) | 凍結 LLM + 学習する出力ヘッド | 4B 凍結 + head 0.916 vs 27B ゼロショット 0.875、8 質問 22.6ms ([lm_benchmark](docs/lm_benchmark.md)) |
| `openvons.vision` | 画像の属性 (年齢・性別・向き・荷物…) | 凍結視覚エンコーダ (407M) + 2.5 万パラメータの head | 27B ゼロショットを上回り VRAM 1/34・36 倍速 ([vision_summary](docs/vision_summary.md)) |
| `openvons.voice` | 数万の固有名詞 + 操作コマンド (状態依存) | kana-whisper の候補一括採点 + 該当なし付き校正 | 50ms、校正後 99〜100%、電話応対の棄却 100% ([voice_evaluation](docs/voice_evaluation.md)) |

共通層 `openvons.core`: Noul / Choice / Score の表現 (`Question`)、判断ポリシー (`decide`、確信度ゲート)、
校正 (温度・isotonic・**該当なし付き校正**)、指標 (ECE / Brier / NLL / macro-F1)。

## JevPick — 「有限のメニューから選ぶ」を文章生成の高速化に使う

[docs/jevpick/README.md](docs/jevpick/README.md)。予測モデルの代わりに、ツール定義・過去の出力・同梱 MTP head から**次の数語の候補メニュー**を作り、
target モデルの hidden state を見て JevPick が 1 つ選び、target が一括で検算する。ツール呼び出し (Qwen3-4B / Qwen3.8-27B、bf16 と 4-bit) で、
メニューに正解が 90% 入り、JevPick は 88% で当て (頻度順ルールは 64%)、**出力を 1 文字も変えずに 3.2〜4.8 倍**。bf16 で学習した JevPick は
FP8 / NF4 の hidden state にそのまま転移する。レポート: [phase 1](docs/jevpick/phase1_oracle_report.md) / [phase 2](docs/jevpick/phase2_report.md) /
[phase 3 (27B・量子化・MTP / DFlash2・Flash-Next)](docs/jevpick/phase3_27b_report.md) / [一覧表](docs/jevpick/summary_table.md)。

## 動かす

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[dev]"   # torch は cu130 index から
.venv/bin/python -m pytest -q tests/test_voice_core.py
```

### テキストの判断デモ

```bash
vllm serve Qwen/Qwen3-4B-Instruct-2507 --served-model-name qwen3-4b --port 8300 --logprobs-mode processed_logprobs   # OpenAI 互換なら何でもよい
scripts/serve_text.sh start 8604
```

### 音声コマンドのデモ

```bash
docker run -d --name voicevox -p 127.0.0.1:50021:50021 voicevox/voicevox_engine:cpu-latest   # 事前学習用 TTS
JEV_APP=examples.stations.app     scripts/serve_demo.sh start 2 8601   # 駅名で動く路線図 (一般向け)
JEV_APP=examples.kasen.app        scripts/serve_demo.sh start 2 8603   # 河川ライブカメラ (関東地整、実画像)
JEV_APP=examples.road_cameras.app scripts/serve_demo.sh start 2 8600   # 道路カメラ監視 (合成カタログ、業務例)
```

マイクは https か localhost が必須 (`ssh -L 8601:localhost:8601 <host>`)。スマホはホーム画面に追加すると PWA として開ける。
マイク経由の発話は `JEV_DUMP_UTTS=1` (既定) で `state/<app>/utts/` に保存され、再校正・実測に使える。

| デモ | 公開 URL | 中身 |
|---|---|---|
| テキストの判断 | https://text.openvons.com | 問い合わせメール・点検メモ・打ち合わせメモを貼り、Noul / Choice / Score の質問をまとめて投げて質問ごとの確率を見る。「確率」「JSON 生成」「一択のみ」の比較と、質問を増やしたときの所要時間をその場で実測 (8 問で 40ms 対 900ms) |
| 駅名で動く路線図 | https://eki.openvons.com | 8,987 駅 (station_database, CC BY 4.0)。駅名 → 寄る、次/前の駅、路線切替、お気に入り (要確認) |
| 河川ライブカメラ (kasen) | https://kasen.openvons.com | 関東地方整備局の河川ライブカメラ 199 地点 (利根川・荒川・那珂川・久慈川・渡良瀬川・霞ヶ浦)。地点名 → ライブ画像、上流 / 下流へ、更新、拡大、お気に入り (要確認) |
| 動画の審判 (judge) | https://judge.openvons.com | 15 秒〜1 分の動画を窓に切り、喧嘩 / 万引き / 高所作業のハーネス未着用 / ヘルメット未着用 / ダブルドリブル / 倒れている人 / 立入禁止 / 火・煙 / 運転中スマホ / 放置荷物 の 10 項目を 3 択で判定、時系列で表示。ライブ (カメラ) でロボットの進路 (左 / 右 / 直進 / 停止) もリアルタイム判定 |
| 顔と全身の属性 (画像) | https://kao.openvons.com | Web カメラか画像。顔ごとに年齢 9 区分・性別 (FairFace head)、全身の性別・年代・向き・荷物 (PA-100K head)。校正済み確率と 確定 / 要確認 / 不明 の 3 段。画像は保存しない |

アプリは `examples/<name>/app.py` (意図・状態・実体) と `static/` だけで、認識・校正・事前学習・サーバーは共通。
設計・評価: [docs/voice_design.md](docs/voice_design.md) / [docs/voice_evaluation.md](docs/voice_evaluation.md)。

### 画像デモ

```bash
scripts/serve_vision.sh start 0 8602   # OPENVONS_FACE_CKPT / OPENVONS_BODY_CKPT に学習済み head、YuNet の onnx は state/models/ に
```

### テキスト / 画像

```bash
.venv/bin/python scripts/lm_prepare_datasets.py               # 公開データ → JSONL
.venv/bin/python scripts/lm_train_head.py --task massive_scenario_en --model Qwen/Qwen3-4B-Instruct-2507
DM_LLM_URL=http://127.0.0.1:8300/v1 .venv/bin/python -m openvons.lm.api.server   # POST /v1/decision と TypeSafe Jev 互換の POST /v1/systemone
```

`/v1/systemone` は typesafe-sdk のワイヤフォーマット (state + questions{type, instructions, criteria} → answers) と同じ形で、
SDK の base_url を向けるだけで手元の Decision Model が答える。公開仕様との対応と規約上の注意は [docs/jev_api.md](docs/jev_api.md)。

### スマホの中だけで動くアプリ (Android、公開済み)

Flutter のアプリ (`app/`) が、音声も画像も **端末内で** 処理する: log-mel → 認識 → 候補採点 → 校正 → 判断、画像の埋め込み → 選択肢との照合。
サーバーはモデルと「選べるものの一覧」を配るだけで、音声・画像は端末から出ない。

- **[Android APK をダウンロード](https://github.com/genai-craft/openvons/releases/latest)** (arm64 39 MB、32bit 31 MB、universal 111 MB。Android 8.0 以上、提供元不明のアプリの許可が要る)
- 3 画面: 声で操作 (河川カメラを声で選んで動かす。従来方式との 2 列比較と「従来なら誤作動 n 回」)、映像の状態 (端末カメラか河川ライブカメラを状態として答える)、設定 (モデル配布元、アクセラレータ)
- 初回だけ約 300 MB のモデルを取得 (既定 `https://ondevice.openvons.com`)。自前で配るなら `scripts/serve_ondevice.sh start 0 8606`
- 実測 (Nothing Phone 3、NNAPI、int8): 音声 1 回 2.0–2.2 秒、画像の埋め込み 260–290 ms、9 個の質問に答える 11 ms
- ブラウザ内推論 (WebGPU / WASM) のデモは https://ondevice.openvons.com
- 小型モデル: kana-whisper (809M) の疑似ラベルで whisper-small を蒸留 (decoder 2〜4 層、147M、`openvons/voice/distill/`)。候補採点方式は自由認識の精度に寛容なので小型化と相性がよい。比較と実機の数字は [docs/voice_small_model.md](docs/voice_small_model.md)、実装の罠は [docs/ondevice_app.md](docs/ondevice_app.md)
- iOS は設定済みだが Mac が無いとビルドできないため未配布

ビルド: `cd app && flutter pub get && flutter build apk --release` ([app/README.md](app/README.md))。

## 構成

```
openvons/core/       primitives (Question), formats (Sample), decision (ポリシー), temperature/isotonic/none_calibration (校正), metrics
openvons/lm/         models (backbone/heads/pooling/decision_model/hybrid_cache), training, backends (LLM 基準), teacher, api, benchmark
openvons/vision/     vision_model (視覚単体), vlm_decision_model (小型 VLM), train_vision/train_vlm, server
openvons/voice/      kana, lexicon (実体・担当範囲), grammar (状態別コマンド集合), asr (kana-whisper), engine, state, vad, synth (事前学習)
openvons/tts/        TTS バックエンド (voicevox:// 既定、irodori://、openai://)
examples/       text_decision (テキストの判断)、stations (駅名で動く路線図)、kasen (河川ライブカメラ)、road_cameras (合成カタログの道路カメラ)
openvons/voice/demo_server.py  共通のデモサーバー (--app で差し替え)、openvons/voice/distill/ 小型 kana モデルの蒸留
scripts/        lm_* (テキスト/画像の実験)、build_catalog / build_stations / eval_synthetic / refit_calibration (音声)、serve_demo.sh
docs/           lm_* / vision_* / voice_* の設計・評価・調査、licensing.md
```

## 導入・カスタマイズのご相談

openvons は独立したオープンソースのプロジェクトです。自社の対象物 (機器・拠点・帳票など) や業務の状態遷移に合わせた作り込み、
オンプレミス導入、モデルの追加学習、実音声での評価などのご相談は **https://genai-craft.com** からお願いします。

GitHub の Issue はコード自体の不具合・質問用です。

## ライセンス

コードは Apache-2.0 — 商用利用・改変・再配布は自由。ただし**出典の記名**をお願いします: 再配布や派生物には [LICENSE](LICENSE) と [NOTICE](NOTICE) をそのまま残し (Apache-2.0 第 4 条の義務)、
製品・サービス・論文・記事で使うときは「openvons (open-Jev) — https://github.com/genai-craft/openvons」と記載してください (引用形式は [CITATION.cff](CITATION.cff))。第三者のモデル・データは [docs/licensing.md](docs/licensing.md)。
openvons (open-Jev) は TypeSafe AI 社および同社製品 Jev とは無関係の独立実装で、同社の API 出力は一切使っていない。
