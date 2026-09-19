# Laya (convaiinnovations/laya) の評価 — openvons.lm との比較

**日付:** 2026-09-20 **スクリプト:** `experiments/laya/eval_laya.py`（Laya 専用 venv `/data/openvons/choice_spec/venvs/laya`）、結果 `experiments/laya/results.json`

## Laya とは

ModernBERT-large (395M) + 判断 head (2 層 Transformer + option-marker scorer + act/escalate)、計 421M、Apache-2.0。state (文章 / JSON) と typed questions (choice / score / noul) を **1 回の forward で答える、生成しない判断モデル**。RLCD (strictly proper scoring rule の強化学習) で確率の正直さを学習したと説明されている。openvons.lm と同じ思想で、公開 README では Jev より高い (typed-decisions 400 件で 0.766 vs 0.727) と主張している。

## 手元のテストセットでの数字

Laya は **ゼロショット**（これらのタスクで学習していない）、openvons.lm は各タスクの train で head を学習している。「学習なしでどこまで出るか」として読む。

| タスク | 問題 | Laya (zero-shot) | openvons.lm 4B 凍結 + head | Laya の ECE | Laya 1 問の遅延 |
|---|---|---:|---:|---:|---:|
| massive_scenario_en | Choice 18 | 0.785 | **0.916** | 0.18 | 9 ms |
| massive_intent_en | Choice 60 | 0.494 | (未計測) | 0.43 | 10 ms |
| massive_scenario_ja | Choice 18 (multilingual ckpt) | 0.672 | **0.898** | 0.10 | 8 ms |
| tweet_offensive | Noul | 0.771 | **0.816** | 0.06 | 9 ms |
| glaive_tools | 可変 Choice (tool 選択) | 0.702 | **0.952** | 0.23 | 8 ms |

n = 1,000 (tweet は 860)。GPU は RTX PRO 6000。

## 読み方

- **速い**: 1 問 8〜10 ms は openvons.lm (8 問 22.6 ms) と同じ帯。421M なので CPU でも回る。
- **精度は学習した head に 13〜25pt 負ける**。選択肢が多い (intent 60) と 0.49 まで落ち、確率も過信 (ECE 0.43)。noul は 0.77 / ECE 0.06 と素直。日本語は multilingual ckpt で 0.67。
- openvons への取り込み方: **学習データが無い立ち上げ時の基準線 / fallback backend** としての価値。`openvons/lm/backends/laya_backend.py` (任意依存、`pip install laya`) を置いた。タスクの学習データが集まったら 4B 凍結 + head に切り替える、という運用。
- 設計として参考になる点: option-marker (選択肢ごとの位置で採点) は openvons の embed head と同型、RLCD による校正は温度校正の代替候補。
