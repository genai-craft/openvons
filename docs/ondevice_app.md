# 端末アプリ (app/) — 声と映像を端末の中で判断する

Android / iOS 用の Flutter アプリ。**推論はすべて端末の中**で動き、サーバーからはモデルと
「選べるものの一覧」だけを取る。音声も映像も端末から出ない。

    app/            Flutter (org com.genaicraft, name openvons)
      lib/engine/   kana (正規化と絞り込み) / decision (校正と判断) / server (配布物の取得)
                    voice (音声の一連) / vision (画像の一連) / console (河川カメラの指令卓)
      lib/pages.dart  3 つの画面
    examples/ondevice/server.py   モデル・コマンド集合・地点一覧・ライブ画像の中継を配るサーバー

## デモとして何を見せているか

技術の説明ではなく、**確率で答える判断層が何を防いでいるか**が画面に出るようにしてある。

1. **声で操作**: 河川カメラの指令卓。地図か一覧から地点名を言うと実画像が出て、
   「ひとつ下流」「更新して」「もっと寄って」で動く。お気に入り登録だけは確認を挟む。
   発話のたびに 2 列で並べる:

   | 列 | 中身 |
   |---|---|
   | 従来のやり方 | 文字起こしに一番近い言い方を選ぶだけ。棄却する手段が無いので必ず何かを実行する |
   | openvons | 候補を確率で採点し、「該当なし」を含めて 実行 / 確認 / 却下 に分ける |

   上部に「実行 n・確認 n・はねた n・**従来なら誤作動 n**」を数える。
   電話をしているつもりで雑談すると、従来の列だけが赤くなる。

2. **いま言えること**: 現在の状態で受理される言い方を常に画面に出す。状態が変わると中身が変わる
   (地図では地点名 37 件、カメラ表示中は操作 8 種、確認待ちは はい / いいえ の 2 つだけ)。
   覚えていなくても画面を見れば分かる、が狙い。

3. **映像の状態**: 端末のカメラか、河川ライブカメラの実画像を判定する。物体検出ではなく
   場面の**状態**を選択肢で答える。時間の内訳を出して「質問を増やしてもほぼタダ」を見せる:

       画像を数値にする 289 ms  +  9 個の質問に答える 11.4 ms

   選択肢には「暗くて判別できない」を入れてある。夜の河川カメラで水位を聞かれたときに
   自信満々で「一面が水」と答えるのではなく、判別できないと言う方が運用では正しい。

4. **マイク無しで試す**: サーバーの合成音声 4 件 (地点名 / 更新 / ひとつ下流 / 電話中の雑談) を
   順に流す。人が喋らなくても一連の流れと、雑談をはねるところまで見られる。

## 端末に置くもの

| ファイル | 大きさ | 役割 |
|---|---|---|
| `mel.onnx` | 0.7MB | 16kHz の波形 → log-mel (torch.stft は書き出せないので窓付き DFT を conv1d で) |
| `encoder_model_quantized.onnx` | 98MB | kana-small-4l の encoder (int8) |
| `decoder_head_quantized.onnx` | 199MB | decoder + 採点 + 次トークン (int8) |
| `image_encoder_quantized.onnx` | 102MB | SigLIP2 の画像側 (int8) |
| `choices*.json` | 数百 KB | 質問と選択肢のテキスト埋め込み (サーバーで計算済み) |

テキスト側のエンコーダは端末に要らない。質問を足す・言い換えるのは JSON を配り直すだけで、
アプリの更新は要らない。

## 罠 (踏んだもの)

- **語彙全体のロジットを端末に渡すと落ちる**。素の `decoder_model.onnx` は (B, T, 51865) を返す。
  候補 20 件で 1300 万要素になり、Flutter のメソッドチャネルを通る途中で Java ヒープ (268MB) を
  超えて `OutOfMemoryError`。`scripts/export_decoder_heads.py` で
  **採点 (scores) と次トークン (next_id) だけを返すグラフ**に包み直した。
  カナ以外の抑制トークン 5 万件もグラフに焼いてある。
- **encoder の出力を端末側で候補数ぶんに複製しない**。(1,1500,768) を 20 回コピーすると 2300 万要素。
  グラフの中で `expand` する。
- **旧 (TorchScript) エクスポータは系列長を焼き込む**。書き出し例が T=6 だと、T=4 の入力で
  `Reshape` が落ちる (`input_shape:{4}, requested shape:{6,1}`)。`dynamo=True` で記号次元のまま出る。
- **dynamo は重みを外部ファイル (.onnx.data) に出す**。配布は量子化後の 1 ファイルで済む。
- **tied weight が 2 回焼かれる**。定数畳み込みが転置済みの複製を作るので fp32 で +160MB。
  畳み込みを切ると重複は消えるが、MatMul の B が定数でなくなり動的量子化が効かない
  (312MB のまま)。量子化後の小ささを取って畳み込みは有効のままにしてある。
- **SigLIP2 に日本語の質問文を直接入れると不安定**。「散らかった机 / 片付いた机」で答えが反転した。
  プロンプトは英語で書き、画面に出す名前だけ日本語にする。
- **夜の監視カメラ**。言い方 1 つだと「乱れ・ノイズ」に倒れる。選択肢ごとに昼と夜の言い方を持たせて
  平均する (prompt ensembling) と収まる。

## 実測 (Nothing Phone 3, NNAPI, int8)

| 処理 | 時間 |
|---|---|
| 音声 1 回 (前処理 → 認識 → 候補採点 → 判断) | 2.0–2.2 秒 |
| 画像を数値にする | 260–290 ms |
| 9 個の質問に答える | 11 ms |

音声の 2 秒は実装の余地であって、機械の限界ではない。KV cache が無く、音声窓が Whisper 既定の
30 秒固定で、精度は fp32 相当の int8。KV cache と 10 秒窓にすれば 1 秒を切る見込み。

## 作り方

    # サーバー (モデルと選択肢の配布元)
    scripts/serve_ondevice.sh start 2 8606

    # 端末に置く ONNX
    .venv/bin/python scripts/export_mel.py
    /data/openjev/onnx_venv/bin/python scripts/export_decoder_heads.py --model <蒸留した学生>
    .venv/bin/python scripts/export_vision_encoder.py
    .venv/bin/python scripts/build_vision_choices.py --set general
    .venv/bin/python scripts/build_vision_choices.py --set kasen

    # アプリ
    cd app && flutter build apk --debug        # Android
    cd app && flutter build ios                # iOS (Mac が要る)

iOS はプロジェクトの設定は済んでいるが、ビルドには macOS と Apple の署名が要る。
