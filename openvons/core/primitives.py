"""Noul / Choice / Score primitives (TASK-002).

All three are normalised into a single internal `Question` representation:
an ordered list of options, each with an id and a description.  Every backend
(LLM baseline, decision model) only has to deal with `Question`.

  Noul   -> 2 options  ["true", "false"]              probability = p[true]
  Choice -> n options  (2 <= n <= 255)                 choice = argmax
  Score  -> n ordered levels                           score = sum(i * p_i)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

MAX_CHOICES = 255
QType = Literal["noul", "choice", "score"]


@dataclass
class Option:
    id: str
    description: str = ""

    def label(self) -> str:
        return self.id if not self.description or self.description == self.id else f"{self.id}: {self.description}"


@dataclass
class Question:
    type: QType
    text: str
    options: list[Option]
    key: str = ""

    def __post_init__(self):
        n = len(self.options)
        if self.type == "noul" and n != 2:
            raise ValueError("noul must have exactly 2 options")
        if not (2 <= n <= MAX_CHOICES):
            raise ValueError(f"number of options must be in [2, {MAX_CHOICES}], got {n}")
        ids = [o.id for o in self.options]
        if len(set(ids)) != n:
            raise ValueError(f"duplicate option ids: {ids}")

    @property
    def n(self) -> int:
        return len(self.options)

    @property
    def ids(self) -> list[str]:
        return [o.id for o in self.options]

    # ---- (de)serialisation -------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "question": self.text,
            "choices": [{"id": o.id, "description": o.description} for o in self.options],
        }

    @staticmethod
    def from_dict(d: dict[str, Any], key: str = "") -> "Question":
        return Question(
            type=d["type"],
            text=d.get("question", d.get("description", "")),
            options=[Option(c["id"], c.get("description", "")) for c in d["choices"]],
            key=key or d.get("key", ""),
        )

    # ---- formatting of results --------------------------------------------
    def format_result(self, probs: list[float]) -> dict[str, Any]:
        assert len(probs) == self.n, (len(probs), self.n)
        if self.type == "noul":
            return {"probability": float(probs[0]), "probabilities": {"true": float(probs[0]), "false": float(probs[1])}}
        if self.type == "choice":
            best = max(range(self.n), key=lambda i: probs[i])
            return {"probabilities": {o.id: float(p) for o, p in zip(self.options, probs)}, "choice": self.options[best].id}
        # score
        return {
            "probabilities": [float(p) for p in probs],
            "levels": self.ids,
            "score": float(sum(i * p for i, p in enumerate(probs))),
        }


# ---- user-facing primitives ------------------------------------------------
class Noul:
    """Boolean decision."""

    def __init__(self, description: str, true_label: str = "yes", false_label: str = "no"):
        self.description = description
        self.true_label = true_label
        self.false_label = false_label

    def to_question(self, key: str = "") -> Question:
        return Question("noul", self.description, [Option("true", self.true_label), Option("false", self.false_label)], key)


class Choice:
    """Categorical decision over a dict {id: description} or list of ids."""

    def __init__(self, choices: dict[str, str] | list[str], description: str = "Which option applies?"):
        if isinstance(choices, dict):
            self.choices = [Option(k, v) for k, v in choices.items()]
        else:
            self.choices = [Option(c, "") for c in choices]
        self.description = description

    def to_question(self, key: str = "") -> Question:
        return Question("choice", self.description, list(self.choices), key)


class Score:
    """Ordered levels; expected score = sum(i * p_i)."""

    def __init__(self, levels: list[str], description: str = "Which level applies?"):
        self.levels = levels
        self.description = description

    def to_question(self, key: str = "") -> Question:
        return Question("score", self.description, [Option(str(i), lv) for i, lv in enumerate(self.levels)], key)


Primitive = Noul | Choice | Score


def to_questions(questions: dict[str, Primitive | Question | dict]) -> list[Question]:
    out = []
    for key, q in questions.items():
        if isinstance(q, Question):
            q.key = key
            out.append(q)
        elif isinstance(q, dict):
            out.append(Question.from_dict(q, key))
        else:
            out.append(q.to_question(key))
    return out
