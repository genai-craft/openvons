"""Shared data format + prompt construction.

Sample (JSONL, spec Phase 2.1):
{
  "id": "massive_en_train_12",
  "state": "wake me up at five am this week",
  "question": "What is the scenario of this utterance?",
  "type": "choice",
  "choices": [{"id": "alarm", "description": "..."}, ...],
  "label": 0,                       # hard label index (may be null)
  "target_probs": [0.9, 0.05, ...]  # soft label (may be null)
}

Option labels used in LLM prompts: A..Z for n <= 26, otherwise 1..n.
"""
from __future__ import annotations

import json
import string
from dataclasses import dataclass, asdict
from typing import Any, Iterator

from .primitives import Question, Option


def option_labels(n: int) -> list[str]:
    if n <= 26:
        return list(string.ascii_uppercase[:n])
    return [str(i + 1) for i in range(n)]


@dataclass
class Sample:
    id: str
    state: str
    question: Question
    label: int | None = None
    target_probs: list[float] | None = None
    meta: dict[str, Any] | None = None

    def to_json(self) -> str:
        d = {"id": self.id, "state": self.state, **self.question.to_dict(), "label": self.label,
             "target_probs": self.target_probs}
        if self.meta:
            d["meta"] = self.meta
        return json.dumps(d, ensure_ascii=False)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Sample":
        return Sample(
            id=str(d["id"]),
            state=d["state"] if isinstance(d["state"], str) else json.dumps(d["state"], ensure_ascii=False),
            question=Question.from_dict(d),
            label=d.get("label"),
            target_probs=d.get("target_probs"),
            meta=d.get("meta"),
        )

    def targets(self) -> list[float]:
        """Soft target if available, else one-hot of the hard label."""
        if self.target_probs is not None:
            return self.target_probs
        assert self.label is not None
        return [1.0 if i == self.label else 0.0 for i in range(self.question.n)]


def read_jsonl(path: str, limit: int | None = None) -> list[Sample]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(Sample.from_dict(json.loads(line)))
            if limit and len(out) >= limit:
                break
    return out


def write_jsonl(path: str, samples: Iterator[Sample] | list[Sample]) -> int:
    n = 0
    with open(path, "w") as f:
        for s in samples:
            f.write(s.to_json() + "\n")
            n += 1
    return n


def state_to_text(state: Any) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, indent=None)


# ---------------------------------------------------------------------------
# Prompt used for the generative LLM baseline / teacher (single question)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a decision engine. You will be given a STATE and a QUESTION with a fixed set of OPTIONS. "
    "Answer with the label of the single best option only. Do not explain."
)


def llm_single_prompt(state: str, q: Question) -> tuple[str, list[str]]:
    labels = option_labels(q.n)
    lines = [f"STATE:\n{state}", "", f"QUESTION: {q.text}", "OPTIONS:"]
    for lab, o in zip(labels, q.options):
        lines.append(f"{lab}. {o.label()}")
    lines.append("")
    lines.append("Answer with the option label only.")
    return "\n".join(lines), labels


def llm_multi_prompt(state: str, qs: list[Question]) -> tuple[str, dict[str, Any]]:
    """One prompt asking all questions, answered as a JSON object (structured output)."""
    lines = [f"STATE:\n{state}", "", "Answer every question below. Reply with a JSON object whose keys are the "
             "question keys and whose values are the chosen option ids.", ""]
    props = {}
    for q in qs:
        lines.append(f"[{q.key}] {q.text}")
        for o in q.options:
            lines.append(f"  - {o.label()}")
        props[q.key] = {"type": "string", "enum": q.ids}
    schema = {"type": "object", "properties": props, "required": [q.key for q in qs], "additionalProperties": False}
    return "\n".join(lines), schema
