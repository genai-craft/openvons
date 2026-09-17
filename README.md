# open-Jev — 有限選択肢に確率で答える判断層

LLM / VLM / ASR に**文章を生成させる代わりに、有限の選択肢へ確率で直接答えさせる**。
「該当なし」も選択肢に入れ、校正した確率で **自動実行 / 確認 / 棄却** を分ける。

同じ主張をテキスト・画像・音声の 3 つの入口で実装している。

| 入口 | 何を選ぶか | 実装 | 実測 (docs/) |
|---|---|---|---|
| `jev.lm` | 意図分類・ツール選択・スコア (Noul / Choice / Score) | 凍結 LLM + 学習する出力ヘッド | 4B 凍結 + head 0.916 vs 27B ゼロショット 0.875、8 質問 22.6ms ([lm_benchmark](docs/lm_benchmark.md)) |
| `jev.vision` | 画像の属性 (年齢・性別・向き・荷物…) | 凍結視覚エンコーダ (407M) + 2.5 万パラメータの head | 27B ゼロショットを上回り VRAM 1/34・36 倍速 ([vision_summary](docs/vision_summary.md)) |
| `jev.voice` | 数万の固有名詞 + 操作コマンド (状態依存) | kana-whisper の候補一括採点 + 該当なし付き校正 | 50ms、校正後 99〜100%、電話応対の棄却 100% ([voice_evaluation](docs/voice_evaluation.md)) |

共通層 `jev.core`: Noul / Choice / Score の表現 (`Question`)、判断ポリシー (`decide`、確信度ゲート)、
校正 (温度・isotonic・**該当なし付き校正**)、指標 (ECE / Brier / NLL / macro-F1)。

## 動かす

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[dev]"   # torch は cu130 index から
.venv/bin/python -m pytest -q tests/test_voice_core.py
```

### 音声コマンドのデモ (道路カメラ監視)

```bash
docker run -d --name voicevox -p 127.0.0.1:50021:50021 voicevox/voicevox_engine:cpu-latest   # 事前学習用 TTS
scripts/serve_road_cameras.sh start 2 8600          # GPU 2、ポート 8600 → http://localhost:8600 (マイクは https か localhost)
```

デモの操作・設計・評価: [docs/voice_design.md](docs/voice_design.md) / [docs/voice_evaluation.md](docs/voice_evaluation.md)。
稼働中のデモ: https://sashizu.aunvox.com

### テキスト / 画像

```bash
.venv/bin/python scripts/lm_prepare_datasets.py               # 公開データ → JSONL
.venv/bin/python scripts/lm_train_head.py --task massive_scenario_en --model Qwen/Qwen3-4B-Instruct-2507
.venv/bin/python -m jev.lm.api.server                        # POST /v1/decision (Noul / Choice / Score)
```

## 構成

```
jev/core/       primitives (Question), formats (Sample), decision (ポリシー), temperature/isotonic/none_calibration (校正), metrics
jev/lm/         models (backbone/heads/pooling/decision_model/hybrid_cache), training, backends (LLM 基準), teacher, api, benchmark
jev/vision/     vision_model (視覚単体), vlm_decision_model (小型 VLM), train_vision/train_vlm, server
jev/voice/      kana, lexicon (実体・担当範囲), grammar (状態別コマンド集合), asr (kana-whisper), engine, state, vad, synth (事前学習)
jev/tts/        TTS バックエンド (voicevox:// 既定、irodori://、openai://)
examples/       road_cameras (音声デモ)、今後 stations (一般向け)
scripts/        lm_* (テキスト/画像の実験)、build_catalog / eval_synthetic / refit_calibration (音声)、serve_road_cameras.sh
docs/           lm_* / vision_* / voice_* の設計・評価・調査、licensing.md
```

## ライセンス

コード Apache-2.0。第三者のモデル・データは [docs/licensing.md](docs/licensing.md)。
