# 第三者ライセンスの棚卸し (2026-09-17)

コードは Apache-2.0。実行時に取得・利用するモデルとデータ:

| 対象 | ライセンス | 用途 | 備考 |
|---|---|---|---|
| sbintuitions/kana-whisper | MIT | 音声の自由認識・候補採点 | 学習データ非公開。出力 (疑似ラベル) の利用制限なし |
| Silero VAD (silero-vad) | MIT | 発話区間の切り出し | |
| pyopenjtalk + Open JTalk 辞書 (naist-jdic) | MIT / BSD-3 | 読みの生成 (フォールバック) | 初回に辞書を DL |
| rapidfuzz | MIT | 絞り込み | |
| 日本郵便 郵便番号データ (KEN_ALL) | 著作権主張なし・自由利用 | 全国の町域名と読み (合成カタログ) | 都道府県別 zip を取得 |
| VOICEVOX ENGINE + 各キャラクター | ENGINE は LGPL-3.0 (別途利用規約)、音声はキャラごとの利用規約 | 事前学習の合成音声 (既定) | 合成音声を配布する場合はクレジット表記が要るキャラがある。デモは合成音声を保存・配布しない |
| Aratako/Irodori-TTS-v4.1-Small | HF のモデルカードを確認 (docs/voice_research.md) | 事前学習の合成音声 (任意バックエンド) | 既定ではない |
| Qwen3 / Qwen3-VL (Qwen/…) | Apache-2.0 | テキスト・画像の Decision Model のバックボーン | |
| FairFace / PA-100K / MASSIVE / tweet_eval / Yelp / glaive | 各データセットの規約 | 評価・学習 (再配布しない) | docs/lm_benchmark.md |

未確認・要作業: Irodori TTS の重みライセンス、VOICEVOX 各キャラの規約 (デモで使う話者を限定して明記する)。
