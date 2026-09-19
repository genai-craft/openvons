# judge — 動画の審判デモ (judge.openvons.com)

**日付:** 2026-09-20 **コード:** `examples/judge/`、`openvons/vision/video_judge.py`

## 何をするか

15 秒〜1 分の動画に、決めた事象があるかを **文章を生成せず、選択肢に確率で答えて**判定する。項目は 10 個:

| 項目 | シーン | fps × 窓 | 危険度 | VLM への質問 |
|---|---|---|---|---|
| 喧嘩・暴力 | 店舗・街 | 2 × 4 s | high | Are people fighting or physically attacking each other? |
| 万引き | 店舗・街 | 2 × 4 s | medium | Is a person hiding merchandise in a bag, pocket or clothing without paying? |
| 高所作業のハーネス未着用 | 工事現場 | 2 × 4 s | high | Is a worker at height working without a safety harness attached? |
| ヘルメット未着用 | 工事現場 | 2 × 4 s | medium | Is anyone on the construction site not wearing a hard hat? |
| バスケのダブルドリブル | スポーツ | **10 × 1.6 s** | low | Does the player stop dribbling, hold the ball with both hands, and dribble again? |
| 人が倒れている | 介護・監視 | 2 × 4 s | high | Is a person lying on the floor or collapsed? |
| 立入禁止区域への侵入 | 工事現場 | 2 × 4 s | medium | Does a person climb over or pass through a fence, gate or barrier? |
| 火・煙 | 介護・監視 | 2 × 4 s | high | Is there fire or smoke that is not from normal cooking? |
| 運転中のスマホ操作 | 道路 | 2 × 4 s | medium | Is the driver looking at or holding a smartphone while the vehicle is moving? |
| 放置された荷物 | 店舗・街 | 2 × 4 s | medium | Does a person leave a bag or suitcase behind and walk away from it? |

答えは常に 3 択 **はい / いいえ / 判別できない**。3 つ目が openvons の「該当なし」で、映っていない・見えないときにここへ倒れる (無関係な vlog を入れると、喧嘩・火は「問題なし」、工事現場や車内が無い項目は「判別できない」になる)。

## しくみ

1. **窓に切る** — 項目ごとの fps で動画をサンプリングし、窓長の半分ずつずらす (`read_frames`、PyAV)。
2. **窓を 1 回だけ符号化** — Qwen3-VL-4B-Instruct にフレーム列 + `State:` を入れて KV cache を作る。
3. **質問ごとに 1 token** — cache を複製し `Question / Options / Decision:` を流して、選択肢の文字 (A/B/C) の logit を読む。質問を 10 個にしても符号化はやり直さない (openvons.lm の kv_shared と同じ)。
4. **校正と判断** — 温度 T (項目別、`state/judge/calibration.json`) で校正し、`openvons.core.decide` の 3 段へ。危険度 high は確信があっても「要確認」。どの窓も「はい」が低いとき、「いいえ」の中央値 ≥ 0.6 なら「問題なし」、そうでなければ「判別できない」。

速さ (RTX PRO 6000、bf16): 8 フレームの窓の符号化 ≈ 0.17 秒、質問 1 つ ≈ 25 ms。12 秒の動画 (5 窓 × 3 問) を 1.4 秒、15 秒の vlog (23 窓 × 71 問、スポーツ窓込み) を 5.9 秒。**ライブ**は 1 フレーム 24 ms + 質問 1 つ 23 ms → 10 fps でも回る。

## ライブ (カメラ)

ブラウザからフレームを 2 / 5 / 10 fps で送る。モードは 2 つ:

- **ロボットの進路**: Choice「左 / 右 / 直進 / 停止」+ Noul「1 m 以内に障害物・人がいるか」+ Choice「前方の床 (床 / 段差 / 物 / 人)」。確信が低いときは HUD を「停止」に倒す (該当なしと同じ作法)。
- **審判項目**: 10 項目を 1 フレーム (または直近数フレーム) で答える。

## 試験動画

各項目について「問題あり」3 本と「問題なし」3 本、計 60 本の 15 秒動画を MiniMax H3 (feel-dance / mojidance の ComfyUI クラスタ) で生成。
問題ありは `[0-5s] 通常 → [5-10s] 事象 → [10-15s] 通常` の絵コンテで、事象の時刻の当たりも見る。生成: `examples/judge/gen_clips.py` (mojidance の workflow builder を Python 化、turbo LoRA 8 step、864×480、1 本 ≈ 76 秒/台)。
評価と温度校正: `examples/judge/eval_clips.py` → `state/judge/eval.json`、`calibration.json`。

## 結果 (60 本、Qwen3-VL-4B、生の確率で判定)

動画単位の正解率 **75%**（60 本中 45 本）。同じシーンの他項目への誤検出 0.02 件/本、全項目では 0.08 件/本。

| 項目 | 問題あり 3 本の検出 (再現率) | 問題なし 3 本の見送り (特異度) | 窓 AUROC | 温度 T |
|---|---:|---:|---:|---:|
| 喧嘩・暴力 | 100% | 100% | 0.69 | 5.9 |
| 万引き | 33% | 100% | 0.47 | 7.9 |
| 高所作業のハーネス未着用 | 67% | 67% | 0.55 | 5.2 |
| ヘルメット未着用 | 0% | 100% | 0.59 | 7.9 |
| バスケのダブルドリブル | 0% | 100% | 0.55 | 2.7 |
| 人が倒れている | 100% | 100% | 0.74 | 5.5 |
| 立入禁止区域への侵入 | 67% | 100% | 0.52 | 6.6 |
| 火・煙 | 100% | 100% | 0.79 | 4.6 |
| 運転中のスマホ操作 | 100% | 67% | 0.67 | 5.9 |
| 放置された荷物 | 100% | 0% | 0.63 | 3.2 |

読み方:

- **はっきり映る事象は当たる**: 喧嘩・倒れている人・火や煙・運転中のスマホは再現率 100%、問題なしも見送れる (火・倒れている人は特異度 100%)。喧嘩は 6〜9.5 秒に p=1.0 と、事象の時刻も合う。
- **細かい動作は 4B の VLM では拾えない**: ダブルドリブル (再現率 0%)、万引きの「鞄に隠す」(33%)、ヘルメットの有無 (0%)。ダブルドリブルは 10 fps でも「一度持って再開する」が読み取れず、ヘルメットは質問を厳密にした結果、生成動画側の描写不足 (問題あり 3 本の中央フレームで全員ヘルメット着用) も重なって落ちた。
- **「問題なし」側の誤りは意味の取り違え**: 放置荷物は「座っている人の足元の鞄」に「はい」と答える (特異度 0%)。質問を「持ち主が離れた」にしても直らなかった。ハーネス・スマホの 1 本ずつも同種。
- 温度 T は 3〜8 と大きく、生の確率は過信。校正後の正解率は {e['accuracy_calibrated']*100:.0f}% (同じ動画で T を当てているので目安)。
- 生成動画の限界: 絵コンテ通りに事象が描かれない本がある (万引き・ヘルメット)。実運用の精度はこの数字とは別物で、実写での確認が要る。

## 次の一手

1. **head を学習する (openvons の本筋)**: VLM に質問文で答えさせる代わりに、凍結した VLM の hidden state に数万パラメータの head を載せ、ラベル付き動画で学習する (画像デモの FairFace / PA-100K head と同じ)。細かい動作 (ダブルドリブル、隠す) はここで伸ばす。
2. VLM を 8B / 32B に上げて質問方式のまま比較する (符号化 1 回 + 質問 25 ms の構造は変わらない)。
3. 実写 (JAF の危険予知動画などを手元で) で「道路」項目を評価する。


## 注意

- 試験動画は生成物なので「方式が動くか」の確認であり、実環境の精度ではない。実写での評価には手元の動画を使う (JAF の危険予知トレーニング動画は © JAF で再配布不可、手元評価のみ)。
- 属性・行動の推定は用途に応じて倫理・法令の確認を。動画はサーバーに保存しない (試験動画を除く)。
- 4B の VLM の限界: 細かい反則 (ダブルドリブル) や小物 (スマホ) は窓の fps と解像度に依存する。
