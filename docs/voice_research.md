# 技術調査メモ (2026-09-17)

音声コマンドを「有限選択肢への確率付き分類」として解くために調べたこと。結論と根拠だけを残す。
出典 URL は末尾。

## 1. ASR の選定: kana-whisper (sbintuitions, MIT)

- whisper-large-v3-turbo (encoder 32 層 + decoder 4 層、809M) をカタカナ出力に fine-tune したもの。
  出力は読み (例: キョーワイイテンキデスネ)。漢字の表記揺れが消えるので、地名を読みで照合する用途に理想的。
- decoder が 4 層と薄いため、候補文を教師強制で一括採点するコストが小さい。
  実測 (RTX PRO 6000, fp16): encoder 1 回 + 自由認識 (greedy) 30〜45ms、64 候補の採点 40ms。
- 罠: 読点「、」を挟んで出力することがある (フナバシミナミイチ、ノボリ)。候補側は読点なしなので
  採点位置がずれ、尤度が 10 nats 落ちて「該当なし」に倒れた。生成と採点の**両方**で句読点トークンを
  -inf にマスクして解決 (生成側だけ抑制しても、採点側にその位置の確率質量が残る)。
- 罠: transformers の版により generate() の返り値に prefix トークンが含まれたり含まれなかったりする。
  特殊トークンは全部 id >= EOT なので、それで内容トークンを切り出す。
- 学習データ・評価値は非公開。雑音や自然発話への頑健性は保証がないので、合成音声で自前評価した (docs/evaluation.md)。

代替として比較した候補 (未採用):
| モデル | 構成 | ライセンス | 所見 |
|---|---|---|---|
| kotoba-whisper-v2.0 | large-v3 蒸留、decoder 2 層、756M | Apache-2.0 | 漢字出力なので読み照合には正規化が要る。Aunvox の brain で使用中 |
| reazonspeech-nemo-v2 | FastConformer 純 RNNT 619M | Apache-2.0 | NeMo の CTC-WS 語彙バイアスは Hybrid 前提で純 RNNT に使えない。GPU-PB (boosting tree) なら可だが旧 .nemo の互換要確認 |
| ARK-ASR-0.6B | Whisper 風 enc + Qwen2 0.6B dec | Apache-2.0 | trust_remote_code、RTFx 314 |
| kodama-ja-streaming-small | Moonshine 系 140M、ストリーミング可 | Apache-2.0 | 確定遅延 0.9s、雑音に弱い、幻覚率 0.1 |

## 2. 既知語彙への寄せ方 (バイアス) の選択肢

1. **教師強制リスコアリング** (採用): 各候補 T について log p(T | 音声) を decoder の forward 1 回 (バッチ可) で得る。
   専用 API は無いが HF 実装でそのまま書ける。CTC top-k → attention decoder rescoring の二段デコード論文と同じ構造。
   長所: 学習不要・候補集合を実行時に差し替えられる・確率が出る。短所: 候補数に線形 (絞り込みで解決)。
2. prompt_ids (initial prompt) による語彙提示: HF ドキュメントに明記。ただし Aunvox の実験では固有名詞で効かず悪化した
   (brain.py のコメント)。不採用。
3. prefix_allowed_tokens_fn による制約ビームサーチ: 候補の trie に沿って許容トークンを制限。
   文法外の発話でも必ず候補のどれかに落ちる (該当なしが作れない) ので不採用。
4. TCPGen for Whisper / CB-Whisper / fine-tune なし contextual biasing: 学習が要る、または KWS の別モデルが要る。
   選択肢が担当範囲ごとに変わる本件では実行時に差し替えられる 1. が適する。

## 3. 「該当なし」を選択肢に入れる

typesafe (Decision Model) の OCR 実験で得た知見: 登録 N 台からの選択に none を入れると、劣化時に誤答でなく棄却に倒れる
(none 無しだと誤答)。音声でも同じで、**自由認識の対数尤度 s_free から β を引いたものを「該当なし」の logit** として
候補と一緒に softmax する。文法内の発話は候補が s_free に近づき (差 0〜数 nats)、文法外の発話は候補が 20〜50 nats 落ちる。
β と温度 T は合成音声で NLL 最小化により校正する (calibration.py)。

## 4. 状態依存文法と確認ダイアログ (先行技術)

- PTZ 制御の context-sensitive grammar (US 9786276)、ユーザー定義音声コマンドの曖昧性安全解析 (US 8234120)、
  医療機器の安全度別多重確認 (US 12478360)。
- 確認方式は 3 段閾値が標準 (高 → 暗黙確認 = 実行して結果を見せる、中 → 明示確認、低 → 棄却)。
  Sagawa+ Interspeech 2004、Belief confirmation using confidence measures。
- 本実装: 危険度 low は p>=0.85 で実行、medium は p>=0.95、high (プリセット保存等) は常に確認。
  確認状態では受理する意図を「はい/いいえ」の 10 表層形に絞る。

## 5. 読み (G2P) の調達

- pyopenjtalk.g2p(kana=True) は長音を「ー」で書く (トーキョー、センセー)。kana-whisper も同じ流儀。
  ただし naist-jdic 依存で固有名詞を誤読する (谷津 → タニツ)。
- 日本郵便の郵便番号データ (KEN_ALL) は全国 12 万町域の読みを持つ (著作権主張なし)。半角カナで「チュウオウ」型
  なので normalize() で「チューオー」に寄せる。全国一括 zip は 2026-09 時点で移転していて 404、都道府県別 zip
  (kogaki/zip/NNxxxx.zip) を `service/search/zipcode/download/` 配下から取ると取れる。
- pyopenjtalk-plus (SudachiPy 補正) や mecab-ipadic-neologd は固有名詞に強いが辞書が重い。
  本実装は「人手の読み > 事前学習で採れた読み > KEN_ALL > G2P」の優先順で、G2P はフォールバック。
- TTS (Irodori) も表示名の G2P を誤る (「2下り」→ ニグダリ、谷津 → タニズ)。事前学習では**読みカナを TTS に渡す**
  ことで TTS の誤読を切り離し、表示名での合成は「登録読みと食い違う実体の警告」にだけ使う。

## 6. 道路カメラのデータ

- 国交省道路局の全国ライブカメラは各整備局ページへのリンク集で、統一 CSV/JSON/API は無い。
  各局の地図型システム (北海道 info-road、東北 romen、中国 road_mlit2019 等) は HTML/内部 JSON。
  カメラ名の形式は事務所ごとに不統一 (「新三国トンネル群馬県側（新上橋）」型、「国道16号 入間市高倉地区」型)。
  「船橋南1上り」型は首都国道事務所 (国道 14/357 号) 系の情報板・道路情報提供システムの名付けに近い。
- デモでは実在の階層 (10 局 59 事務所、実在の国道番号) に KEN_ALL の町域名を載せて 2,932 台のカタログを合成した
  (scripts/build_catalog.py)。実データが手に入れば lexicon の JSON を差し替えるだけでよい。

## 出典
- kana-whisper https://huggingface.co/sbintuitions/kana-whisper
- Whisper prompt_ids https://huggingface.co/docs/transformers/model_doc/whisper
- 二段デコード (CTC top-k → attention rescoring) https://arxiv.org/abs/2506.12154
- TCPGen for Whisper https://github.com/BriansIDP/WhisperBiasing / https://arxiv.org/abs/2306.01942
- CB-Whisper https://aclanthology.org/2024.lrec-main.262/ 、fine-tune なしバイアス https://arxiv.org/abs/2410.18363
- NeMo word boosting (GPU-PB) https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/asr_customization/word_boosting.html 、https://arxiv.org/abs/2508.07014 、CTC-WS https://arxiv.org/abs/2406.07096
- kotoba-whisper https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0 、reazonspeech-nemo-v2 https://huggingface.co/reazon-research/reazonspeech-nemo-v2 、ARK-ASR https://huggingface.co/Audio8/ARK-ASR-0.6B 、kodama https://huggingface.co/ayousanz/kodama-ja-streaming-small
- pyopenjtalk ユーザー辞書 https://eqol.main.jp/2023/12/24/pyopenjtalk/ 、pyopenjtalk-plus https://github.com/tsukumijima/pyopenjtalk-plus
- 確認ダイアログ Sagawa+ 2004 https://www.isca-archive.org/interspeech_2004/sagawa04_interspeech.pdf
- 郵便番号データ https://www.post.japanpost.jp/service/search/zipcode/download/utf-zip.html (仕様 utf-readme.html)
- 国交省 道路ライブカメラ https://www.mlit.go.jp/road/bosai/LIVEcamera.html 、関東 https://www.ktr.mlit.go.jp/guide/guide00000012.html 、事務所一覧 https://www.ktr.mlit.go.jp/soshiki/office_list.html
- 国道×都道府県 (e-Stat 表26) https://www.e-stat.go.jp/stat-search/files?page=1&bunya_l=08&layout=dataset&toukei=00600610&tstat=000001017725
