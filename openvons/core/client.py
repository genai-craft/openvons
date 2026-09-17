"""Python SDK: client.decide(state=..., questions={...}) against POST /v1/decision."""
from __future__ import annotations

from typing import Any
import httpx

from .primitives import to_questions
from .formats import state_to_text


class Client:
    def __init__(self, base_url: str = "http://127.0.0.1:8400", timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self._http = httpx.Client(timeout=timeout)

    def decide(self, state: Any, questions: dict, backend: str | None = None) -> dict[str, Any]:
        qs = to_questions(questions)
        payload = {"state": state_to_text(state), "questions": [{"key": q.key, **q.to_dict()} for q in qs]}
        if backend:
            payload["backend"] = backend
        r = self._http.post(f"{self.base_url}/v1/decision", json=payload)
        r.raise_for_status()
        return r.json()["results"]
