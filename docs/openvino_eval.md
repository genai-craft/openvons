# Intel CPU / iGPU (OpenVINO) で openvons の端末モデルは動くか

**日付:** 2026-09-20 **機材:** Core i9-13900HK (Raptor Lake-P、20 スレッド) + 内蔵 UHD Graphics、RAM 31 GB、Ubuntu 24.04、OpenVINO 2026.4。NPU は無い世代。
**スクリプト:** `experiments/openvino/bench_ov.py` (ONNX を OpenVINO で CPU / GPU にコンパイルして計測)。iGPU を使うには `intel-opencl-icd` と `render` グループが要る。

## 結果 (1 件あたりの推論時間、中央値)

| モデル | 用途 | 精度 | CPU | iGPU |
|---|---|---|---:|---:|
| kana-whisper 蒸留 encoder (whisper-small, 30 秒窓) | 音声コマンドの認識 | fp32 | 651 ms | **231 ms** |
| 同 int8 (ONNX QDQ) | | int8 | 893 ms | 8,959 ms |
| decoder (8 token, KV なし) | 自由認識 1 step | fp32 | 8.8 ms | 6.3 ms |
| decoder head (候補 8 個の一括採点) | 候補採点 | int8 | 45 ms | 568 ms |
| SigLIP2-base image encoder | 画像の状態判定 (端末アプリ) | fp16 | 70 ms | **30 ms** |
| 同 int8 (ONNX QDQ) | | int8 | 89 ms | 1,270 ms |
| Qwen3-0.6B (INT4, optimum-intel) | テキストの判断 (prompt 195 token → 選択肢 logit) | int4 | 564 ms | **348 ms** |
| Qwen3-1.7B (INT4) | 同上 | int4 | 632 ms | **410 ms** |
| 同、生成 1 token あたり | (参考: 生成させた場合) | int4 | 71〜102 ms | 71〜91 ms |

## 読み方

- **動く**。音声 1 回 (encoder + 認識 + 採点) は iGPU で 0.3〜0.4 秒、CPU でも 0.7 秒。スマホ (Nothing Phone 3、NNAPI int8) の 2.0〜2.2 秒より速い。画像は iGPU 30 ms で「9 問 11 ms」を足しても 50 ms 以内。
- **ONNX の int8 (QDQ) は OpenVINO では逆効果**。CPU でも fp32 より遅く、iGPU では 10〜40 倍遅い。Intel 向けには fp16 (GPU) / fp32 (CPU) をそのまま使い、量子化するなら OpenVINO 側 (NNCF) でやる。
- decoder head (採点) だけは GPU より CPU が速い (568 vs 45 ms)。小さい行列と動的な形が多い処理は CPU に置く、と分けるのが良い。
- **テキストの判断 (openvons.lm) も 0.35〜0.6 秒/状態で回る** (1 回の prefill で全質問の logit が出る形なので、質問を足しても増えない)。生成に切り替えると 1 token 70〜100 ms なので、ここでも「生成しない」設計の差が出る。
- Qwen3-VL-2B (judge のライブ判定) の OpenVINO 変換は optimum-intel が transformers 5.0 系までしか対応せず、5.0 系に下げて再試行中 (結果は追記)。
- openvons の端末アプリ (Android) と同じ ONNX がそのまま読めたので、**Intel ミニ PC / ノートを「置き型の端末」にする経路は成立する**。
