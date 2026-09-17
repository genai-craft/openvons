from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class ChoiceIn(BaseModel):
    id: str
    description: str = ""


class QuestionIn(BaseModel):
    key: str
    type: Literal["noul", "choice", "score"]
    question: str = Field(default="", description="natural-language question / description")
    choices: list[ChoiceIn] | None = None          # choice / score
    levels: list[str] | None = None                # score shorthand
    true_label: str = "yes"                        # noul
    false_label: str = "no"

    @model_validator(mode="after")
    def _fill(self):
        if self.type == "noul":
            self.choices = [ChoiceIn(id="true", description=self.true_label), ChoiceIn(id="false", description=self.false_label)]
        elif self.type == "score" and self.levels and not self.choices:
            self.choices = [ChoiceIn(id=str(i), description=lv) for i, lv in enumerate(self.levels)]
        if not self.choices or not (2 <= len(self.choices) <= 255):
            raise ValueError("choices must contain between 2 and 255 entries")
        return self


class DecisionRequest(BaseModel):
    state: Any
    questions: list[QuestionIn]
    backend: str | None = None            # llm | model | auto (model with LLM fallback)
    mode: str | None = None               # model: naive|batched|kv_shared|block_diag ; llm: logprob|greedy|sample
    fallback_threshold: float = 0.6       # auto: below this max-probability -> fallback to LLM


class DecisionResponse(BaseModel):
    results: dict[str, Any]
    backend: str
    latency_ms: float
    meta: dict[str, Any] = {}
