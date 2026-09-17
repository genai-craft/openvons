"""openvons.core — 有限選択肢に確率で答えるための共通層.

primitives   Noul / Choice / Score と内部表現 Question (typesafe 由来)
formats      Sample / JSONL 形式
decision     判断ポリシー (実行 / 確認 / 棄却 / 該当なし)、確信度ゲート
temperature  温度・ベクトルスケーリング、isotonic  等温回帰 (分類器の校正)
none_calibration  該当なし付き校正 (候補尤度 + 自由仮説尤度、音声で使用)
metrics      ECE / Brier / NLL / accuracy / macro-F1
"""
from .primitives import Choice, Noul, Option, Question, Score, to_questions  # noqa: F401
from .formats import Sample, read_jsonl, write_jsonl  # noqa: F401
from .decision import Action, Risk, Thresholds, coverage_precision, decide  # noqa: F401
