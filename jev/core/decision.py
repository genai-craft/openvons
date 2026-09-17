"""判断ポリシー: 校正済みの確率分布から「実行 / 確認 / 棄却 / 該当なし」を決める (全モダリティ共通).

typesafe (Decision Model) の知見:
  - 低確信をより弱いモデルに回すカスケードは逆効果。有効なのは「確信度が高いものだけ自動処理し、残りを人 (確認) に回す」
  - 「該当なし」を明示的な選択肢にすると、劣化時に誤答でなく棄却に倒れる
音声コマンド (jev.voice) の知見:
  - 危険度 (risk) の高い操作は確信度に関わらず確認する。確認状態は選択肢が小さいので雑音に強い
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Action = Literal["execute", "confirm", "reject", "none"]
Risk = Literal["low", "medium", "high"]


@dataclass
class Thresholds:
    execute: float = 0.85          # これ以上なら実行 (危険度 low)
    confirm: float = 0.40          # これ以上なら確認、未満は棄却
    execute_medium: float = 0.95   # 危険度 medium はより高い確信を要求
    # 危険度 high は確信度に関わらず必ず確認


def decide(p_top: float, none_prob: float, risk: Risk = "low", th: Thresholds | None = None) -> tuple[Action, str]:
    """最尤候補の確率 p_top と該当なしの確率から行動を決める。"""
    th = th or Thresholds()
    if none_prob > p_top:
        return "none", f"該当なしが最尤 ({none_prob:.2f})"
    if risk == "high":
        return ("confirm", "危険度 high は常に確認") if p_top >= th.confirm else ("reject", "確信度不足")
    if risk == "medium":
        if p_top >= th.execute_medium:
            return "execute", f"p={p_top:.2f} >= {th.execute_medium}"
        return ("confirm", f"p={p_top:.2f} (medium)") if p_top >= th.confirm else ("reject", "確信度不足")
    if p_top >= th.execute:
        return "execute", f"p={p_top:.2f} >= {th.execute}"
    if p_top >= th.confirm:
        return "confirm", f"p={p_top:.2f} in [{th.confirm},{th.execute})"
    return "reject", f"p={p_top:.2f} < {th.confirm}"


def coverage_precision(probs: np.ndarray, labels: np.ndarray, thresholds=(0.5, 0.85, 0.95)) -> dict[str, dict[str, float]]:
    """確信度ゲート: しきい値以上を自動処理したときの網羅率と精度 (Decision Model の cascade_analysis と同じ物差し)。"""
    probs = np.asarray(probs); labels = np.asarray(labels)
    conf = probs.max(1); correct = probs.argmax(1) == labels
    out = {}
    for t in thresholds:
        m = conf >= t
        out[str(t)] = {"coverage": float(m.mean()), "precision": float(correct[m].mean()) if m.any() else 0.0}
    return out
