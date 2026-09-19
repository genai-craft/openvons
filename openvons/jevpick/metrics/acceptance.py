"""oracle / prior-top1 の受理長集計と replay 速度推定 (§11.1, §22)。"""
from __future__ import annotations

import numpy as np


def replay_tokens_per_step(acc: np.ndarray, block_len: int, step_cost=None) -> tuple[float, float]:
    """acc[t] = 位置 t で投機したときの受理長。順に辿って verification 回数を数える。
    戻り値: (tokens/step, 推定 speedup)。step_cost(k) は k token を一括 forward する時間 / 1 token decode 時間。"""
    n = len(acc)
    t = 0
    steps = 0
    cost = 0.0
    while t < n:
        a = int(acc[t])
        steps += 1
        if a == 0 and step_cost is not None:
            # 候補無し (or 受理 0) でも投機した場合は block forward の費用。ここでは投機した想定
            cost += step_cost(block_len + 1)
        elif step_cost is not None:
            cost += step_cost(block_len + 1)
        t += a + 1
    tps = n / steps if steps else 0.0
    speedup = n / cost if step_cost is not None and cost > 0 else tps
    return tps, speedup


def replay_with_skip(acc: np.ndarray, has_cand: np.ndarray, block_len: int, step_cost) -> float:
    """候補が無い位置は通常 decode (cost 1) に fallback する controller の推定 speedup。"""
    n = len(acc)
    t = 0
    cost = 0.0
    while t < n:
        if has_cand[t]:
            cost += step_cost(block_len + 1)
            t += int(acc[t]) + 1
        else:
            cost += 1.0
            t += 1
    return n / cost if cost > 0 else 0.0
