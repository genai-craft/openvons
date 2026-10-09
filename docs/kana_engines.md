# kana 入力エンジンの切り替えと、振り分けの WebAssembly 版 (2026-10-09)

音声デモ (指令 shirei・駅 eki・河川 kasen・端末内 ondevice) の「音声 → カナ」の部分を切り替えられるようにし、
「カナ → コマンドへの振り分け」(絞り込み・採点・校正・判断) をブラウザの WebAssembly でも動かせるようにした。

## 何が変わったか

1. **kana 入力を共有 ASR サーバーに 1 つだけ載せる** (`openvons/voice/asr_server.py`、`scripts/serve_asr.sh`、:8630)。
   以前は音声デモ 4 本がそれぞれ kana-whisper (809M) を GPU に読み、計 17 GB を使っていた。今は 1 プロセス 2.5 GB (kana-whisper) +
   komimi 4 本 (CPU) で、各デモは HTTP で問い合わせる (`openvons/voice/remote.py`)。デモのプロセスは GPU を使わない。
2. **kana 入力エンジンを選べる** (`openvons/voice/engines.py`)。kana-whisper に加えて [komimi](https://github.com/genai-craft/komimi)
   (自前の Conformer-CTC、Apache-2.0) の v12 / v12a / v12m / v12s。画面の「カナ入力」で切り替える (セッションごと)。
3. **振り分けを選べる**: 「サーバー (Python)」(従来) と「ブラウザ (WebAssembly)」。後者は komimi の C エンジンと振り分けの C 実装
   (`openvons/voice/wasm/ov_route.c`) を 1 つの WebAssembly (85 KB) にし、音声 → CTC 行列 → 判断までをブラウザで済ませる。
   サーバーに送るのは「どの仮説が何 % か」だけで、音声はブラウザから出ない。サーバーは意味ごとの合算と判断をやり直して状態機械を進める。

```
            ┌───────────── サーバー振り分け (従来) ─────────────┐
マイク ──┬─ PCM ─→ デモ (VAD) ─→ 共有 ASR サーバー :8630 ─→ Recognizer (Python) ─→ 状態機械
         │                       kana-whisper (GPU) / komimi (CPU)
         │
         └─ ブラウザ振り分け: VAD (vad.js) ─→ ワーカー (ovkana.wasm: komimi + ov_route.c) ─→ {仮説 index, 確率} ─→ デモ ─→ 状態機械
```

## なぜ komimi なら振り分けを WebAssembly にできるか

kana-whisper の候補採点は「候補ごとに decoder を教師強制で回す」(log p(候補 + EOT | 音声))。候補の数だけ大きな decoder を回すので、
ブラウザに持ち出すには decoder ごと持ち出すしかない (`examples/ondevice` の kana 蒸留 + onnxruntime-web がそれ)。

komimi は CTC なので、音声側は (T', 1025) の行列を 1 回作るだけで、候補の採点は行列の上の動的計画法 (CTC forward、
blank を挟む全アラインメントの和) で済む。16 候補 + 埋め込み文の採点が WebAssembly で 1 ms 未満。絞り込み (Levenshtein と
部分一致、7,000 仮説) を足しても 3〜6 ms。重いのは komimi の encoder だけで、振り分けそのものは軽い。

CTC は「候補で説明できない音」を blank で埋めるしかないので、短い候補 (「はい」) が長い発話 (「はい、お世話になっております」) に
勝つことがほぼ無い。kana-whisper で γ (説明しない残りへの罰則) が要った理由 (EOT が安い) が構造的に起きない。

## ov_route.c — Python の Recognizer と同じ判断を出す C 実装

`openvons.voice.engine.Recognizer` (+ `grammar.CommandSet.shortlist`、`core.none_calibration`、`core.decision`) を同じ手順・同じ定数で移した:

- 自由認識: greedy (blank と重複を除く) → カナ正規化 (ヲ→オ・長音化・先頭の ー/ッ を落とす) → 反復ハルシネーションの検出
- 絞り込み: rapidfuzz の `Levenshtein.normalized_similarity` 上位 3K + `fuzz.partial_ratio` (cutoff 80) 上位 2K、意味ごとに 2 表層形まで
- 埋め込み採点: 候補を自由認識の最も似た区間に置き換えた文も採点 (`partial_ratio_alignment`)
- トークナイザ: komimi の語彙 (SentencePiece unigram の piece と対数確率、`komimi/models/ja1024_vocab.json`) で Viterbi 分割。SentencePiece と 99.1% 一致
- 採点: CTC forward。校正 (T, β0, β1, γ) → [候補..., 該当なし] の softmax → 意味ごとに合算 → 3 段閾値と危険度

一致の確認 (`scripts/check_route_equivalence.py`): 3 デモ × 全状態 (9 状態) で TTS 発話 306 件 (半分は SNR 10 dB の雑音付き) を
同じ log-softmax 行列から Python と C に通し、**自由認識・絞り込みの集合・最尤の意味・判断がすべて一致** (komimi-v12a と v12s の 2 系統、計 612 件)。
確率の差は最大 3e-4。

落とし穴: rapidfuzz の C++ 版 (実際に Recognizer が使う) と純 Python 版 (`fuzz_py`) は `partial_ratio_alignment` の
**同点の窓の選び方が違う** (乱択 2 万組で 8.5% が別の窓)。C++ 版は同じ長さの窓を二分探索の順で調べ、先に見た窓を採る。
C++ 版の順番を移して乱択 20 万組で窓まで一致させた。C++ 版は cutoff ちょうどの距離も最初の 1 回は受ける (`cutoff_dist + 1`)。

## 使い方

```bash
scripts/serve_asr.sh start 0 8630                    # 共有 ASR サーバー (GPU 0 に kana-whisper、komimi は CPU)
JEV_APP=examples.road_cameras.app scripts/serve_demo.sh start 0 8600   # OPENVONS_ASR_URL (既定 http://127.0.0.1:8630) に問い合わせる
scripts/serve_all.sh start                           # 共有 ASR サーバーを先に、デモ 8 本を後に
openvons/voice/wasm/build.sh                         # WebAssembly を作り直す (komimi の csrc を KOMIMI_HOME から。docker の emscripten)
```

- 画面: マイクボタンの下の「カナ入力」「振り分け」。設定はブラウザに残る (localStorage)。「ブラウザ」を選ぶと komimi のモデル
  (v12s 7 MB / v12m 14 MB / v12a 28 MB / v12 106 MB) を一度だけ取得する。テキスト試験 (TTS) もブラウザ振り分けで通る
  (サーバーが合成した音声をブラウザで認識する)。
- 端末内デモ (ondevice) はモデル一覧に komimi (WebAssembly) が増え、比較用のサーバー側エンジンも選べる。
- 単独で動かすとき (`--asr local`) は、選ばれたエンジンをデモのプロセスに読む (kana-whisper は GPU、komimi は CPU)。
- komimi は `KOMIMI_HOME` (既定 `~/dev/komimi`) の `models/*.kmm` と `csrc/libkomimi.so` (`make -C csrc libkomimi.so`) を使う。
- 校正はエンジンごと (尤度の尺度が違う)。範囲の事前学習 (画面のボタン) は選んでいるエンジンで行い、`profile.calibrations[エンジン]` に入る。
  読みの追加 (TTS 往復で聞こえた読みを辞書に足す) は kana-whisper のときだけ (小さいエンジンの聞き違いを共通の辞書に入れない)。
  事前学習していない範囲は `openvons/voice/engines_calibration.json` (3 デモの既定範囲の校正サンプルをまとめて fit した値) を使う。
  範囲ごとの fit は境界 (T 0.5、β0 12) に張り付くことがあり、パラメータごとの中央値は互いに合わない組み合わせになるので、まとめて 1 つ fit している。

API (デモサーバー):

| | |
|---|---|
| `GET /api/kana/config` | エンジン一覧 (サーバー / ブラウザで使えるか、ブラウザ用モデルの URL)、語彙・WASM・ワーカーの場所 |
| `GET /api/kana/commands?session=&engine=` | いまの状態のコマンド集合 `[カナ, 意味 id, 危険度, フラグ]`、校正値、`cs_key` |
| WS `{"type":"engine","engine":ID}` | このセッションのエンジンを切り替える |
| WS `{"type":"decision", cs_key, cands:[[仮説 index, 確率, スコア]], none_prob, ...}` | ブラウザで振り分けた結果。状態が変わっていたら `stale` が返り、ブラウザが同じ音声で振り分け直す |
| `POST /api/say {recognize:false}` | TTS の音声だけ返す (ブラウザ振り分けの試験用) |

共有 ASR サーバー: `GET /v1/engines`、`POST /v1/encode?engine=` (本文 float32 PCM → handle・自由認識)、`POST /v1/tokenize`、`POST /v1/score`。

## 実測

### 精度 (合成音声、各デモの既定範囲、`scripts/fit_engine_calibration.py`)

デモの「事前学習」と同じ発話 (実体名 × 声 3 種 × 言い方 2 種、SNR 20 / 10 dB の雑音と背景音を混ぜたもの、他状態の命令、文法外の発話)。
「既定」は事前学習なしの既定校正での判断の正解率、「校正後」は同じデータで校正した後の分類精度 (該当なし含む)。

| デモ (既定範囲) | エンジン | 発話 | 範囲で校正後 | 既定値のまま | SNR 20 dB | SNR 10 dB | 文法外を該当なしに | 文法外を実行 | ECE (校正後) | サーバーでの遅延 |
|---|---|---|---|---|---|---|---|---|---|---|
| 指令 (首都国道 100 台) | kana-whisper | 600 | **0.975** | — | 0.965 | 0.984 | 1.00 | 0.00 | 0.015 | 54 ms |
| 指令 (首都国道 100 台) | komimi-v12 | 600 | **0.972** | 0.972 | 0.980 | 0.958 | 1.00 | 0.00 | 0.011 | 206 ms |
| 指令 (首都国道 100 台) | komimi-v12a | 600 | **0.967** | 0.966 | 0.970 | 0.958 | 1.00 | 0.00 | 0.010 | 77 ms |
| 指令 (首都国道 100 台) | komimi-v12m | 600 | **0.971** | 0.967 | 0.970 | 0.964 | 1.00 | 0.00 | 0.010 | 56 ms |
| 指令 (首都国道 100 台) | komimi-v12s | 600 | **0.972** | 0.967 | 0.970 | 0.969 | 1.00 | 0.00 | 0.009 | 30 ms |
| 駅 (山手線 30 駅) | kana-whisper | 360 | **0.990** | — | 1.000 | 0.990 | 0.92 | 0.00 | 0.017 | 44 ms |
| 駅 (山手線 30 駅) | komimi-v12 | 360 | **0.995** | 0.995 | 0.985 | 0.990 | 1.00 | 0.00 | 0.007 | 146 ms |
| 駅 (山手線 30 駅) | komimi-v12a | 360 | **0.993** | 0.990 | 0.993 | 1.000 | 1.00 | 0.00 | 0.012 | 54 ms |
| 駅 (山手線 30 駅) | komimi-v12m | 360 | **0.990** | 0.985 | 0.993 | 0.980 | 1.00 | 0.00 | 0.010 | 41 ms |
| 駅 (山手線 30 駅) | komimi-v12s | 360 | **0.993** | 0.983 | 0.993 | 0.990 | 0.92 | 0.00 | 0.016 | 21 ms |
| 河川 (利根川上流) | kana-whisper | 370 | **0.995** | — | 1.000 | 1.000 | 1.00 | 0.00 | 0.027 | 54 ms |
| 河川 (利根川上流) | komimi-v12 | 370 | **0.993** | 0.988 | 0.993 | 0.982 | 1.00 | 0.00 | 0.009 | 209 ms |
| 河川 (利根川上流) | komimi-v12a | 370 | **0.995** | 0.990 | 0.993 | 0.982 | 1.00 | 0.00 | 0.018 | 78 ms |
| 河川 (利根川上流) | komimi-v12m | 370 | **0.990** | 0.985 | 0.985 | 0.965 | 1.00 | 0.00 | 0.013 | 57 ms |
| 河川 (利根川上流) | komimi-v12s | 370 | **0.993** | 0.983 | 0.977 | 0.974 | 1.00 | 0.00 | 0.013 | 30 ms |

- 「範囲で校正後」は範囲ごとに fit した校正で判断し直した分類精度 (該当なし含む、argmax)。デモの既定範囲にはこの校正値を入れてある
  (`scripts/apply_engine_calibration.py`)。「既定値のまま」は 3 デモのサンプルをまとめて fit した既定値 (`engines_calibration.json`、
  事前学習していない範囲で使う) で判断した精度。kana-whisper の既定値は従来の `Calibration()`。
- **komimi はどの大きさでも kana-whisper (809M) と 1 ポイント以内** (最大の差は指令の v12a で 0.8 ポイント。6.5M の v12s は 3 デモとも 0.3 ポイント以内)。自由認識の CER は 17〜20% と大きく違うが、有限の候補を CTC で
  採点するので選択はほぼ同じ精度になる。文法外の発話を実行してしまったものは全エンジン 0 件 (判断の規則込み)。
  駅の v12m・v12s は既定値のままだと文法外 24 件中 1 件を argmax で候補に寄せる (範囲で校正すれば 0)。
- サーバーでの遅延は共有 ASR サーバー経由 (HTTP 往復込み)。kana-whisper は GPU、komimi は CPU 1 スレッド。

### 遅延 (1 発話、TTS の 1〜3 秒の発話)

| エンジン | サーバー振り分け (共有 ASR、CPU/GPU) | ブラウザ振り分け (WebAssembly SIMD) | うち振り分け (絞り込み + 採点 + 判断) | ブラウザが取得する大きさ |
|---|---|---|---|---|
| kana-whisper | 43–83 ms (GPU) | — (decoder が要るので不可) | — | — |
| komimi v12 (100M) | 145–209 ms | 207–352 ms | 3–6 ms | 106 MB |
| komimi v12a (30M) | 28–92 ms | 67–119 ms | 3–6 ms | 28 MB |
| komimi v12m (13M) | 40–57 ms | 36–81 ms | 3–6 ms | 14 MB |
| komimi v12s (6.5M) | 21–30 ms | 23–47 ms | 3–6 ms | 7 MB |

端末内デモ (ondevice) のセルフテスト 4 発話は komimi v12m (WebAssembly) で 4/4 一致・平均 80 ms (SIMD カーネルを入れる前は 252 ms)。

- ブラウザは headless Chromium (このサーバーの CPU、1 スレッド)。WebAssembly SIMD128 の int8 内積カーネル (komimi の `KM_KERNEL_WASM_SIMD`) で
  encoder が 3〜4 倍速くなった (v12m 247 → 60 ms)。SIMD の無いブラウザは自動で generic 版 (`ovkana_nosimd.wasm`) を使う (結果は同じ)。
- ブラウザ振り分けは発話の終わり (VAD) から全文脈で 1 回計算する。komimi は逐次 (チャンク) でも動くので、話しながら計算して
  終わった瞬間に判断する形にもできる (未実装)。

### GPU メモリ

| | 以前 | 今 |
|---|---|---|
| 指令 / 駅 / 河川 / 端末内 (比較用) | 6.6 + 3.2 + 4.8 + 2.5 GB (各自 kana-whisper) | 0 (GPU を使わない) |
| 共有 ASR サーバー | — | 2.5 GB (kana-whisper 1 つ) |

## 制限と次の手

- kana-whisper はブラウザ振り分けに使えない (候補採点に decoder が要る)。ブラウザは komimi だけ。
- 校正の既定値は合成音声 (VOICEVOX) で fit したもの。実マイクの発話 (`state/<app>/utts/`) で再校正するのが次。
- komimi の自由認識の CER (dev 17〜20%) は kana-whisper より大きいが、コマンドの選択では候補の CTC 採点が効き、精度の差はほぼ無い (上の表)。
  ただし有限の選択肢の外 (自由文) を聞き取る用途には向かない。
- ブラウザの komimi は 1 スレッド。SharedArrayBuffer (COOP/COEP) を有効にしてスレッドを使えばさらに速くなる。
