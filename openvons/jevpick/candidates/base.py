"""候補の共通形式 (§6)。"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Candidate:
    token_ids: tuple[int, ...]
    source: str
    prior_score: tuple  # 大きいほど優先。source 内でのみ比較可能
    template_id: str | None = None
    slots: dict | None = None
    metadata: dict = field(default_factory=dict)


def match_len(cand: tuple[int, ...], truth) -> int:
    """候補と正解の共通 prefix 長 (§9.1 の受理長)。"""
    n = 0
    for a, b in zip(cand, truth):
        if a != b:
            break
        n += 1
    return n
