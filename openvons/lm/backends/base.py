from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from openvons.core.primitives import Question


@dataclass
class Decision:
    probs: list[float]                  # over question.options
    latency_ms: float = 0.0
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def argmax(self) -> int:
        return max(range(len(self.probs)), key=lambda i: self.probs[i])

    @property
    def confidence(self) -> float:
        return max(self.probs)


class DecisionBackend:
    name = "base"

    def decide(self, state: str, questions: list[Question]) -> list[Decision]:
        raise NotImplementedError

    async def adecide(self, state: str, questions: list[Question]) -> list[Decision]:
        return self.decide(state, questions)
