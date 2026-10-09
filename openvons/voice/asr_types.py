"""ASR backend の共通の型 (torch に依存しない。共有 ASR サーバーの client や komimi backend から使う)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Transcript:
    kana: str
    logprob: float          # 生成列の対数尤度の合計 (prefix と EOT を除く内容トークン)
    n_tokens: int
    ms: float
    tokens: list[int] | None = None


@dataclass
class Encoded:
    hidden: Any             # kana-whisper: encoder 出力 (1, T, d) の torch.Tensor / komimi: CTC の log-softmax (T', 1025) / remote: サーバー側の handle
    seconds: float
    ms: float
