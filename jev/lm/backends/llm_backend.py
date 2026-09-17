"""TASK-003: OpenAI-compatible generative LLM backend (Phase 0 baseline + teacher).

Modes (per question):
  logprob : guided choice over option labels, max_tokens=1, read the (processed) logprobs of the
            first token -> full probability distribution over options in ONE forward pass.
            Requires single-token labels (n <= 26 -> A..Z). Otherwise falls back to `sample`.
  sample  : guided choice, temperature=1, n=K samples -> empirical distribution (Method B).
  greedy  : guided choice, temperature=0 -> one-hot (pure "structured output" baseline).
Multi-question:
  json    : all questions in one request, JSON-schema constrained decoding -> one-hot each.
"""
from __future__ import annotations

import asyncio
import math
import time
from typing import Any

import httpx

from jev.core.primitives import Question
from jev.core.formats import llm_single_prompt, llm_multi_prompt, option_labels, SYSTEM_PROMPT
from .base import Decision, DecisionBackend


class LLMBackend(DecisionBackend):
    name = "llm"

    def __init__(self, base_url: str = "http://127.0.0.1:8300/v1", model: str = "qwen3-4b", mode: str = "logprob",
                 n_samples: int = 20, timeout: float = 120.0, disable_thinking: bool = True, api_key: str = "EMPTY",
                 concurrency: int = 64, json_multi: bool = False):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.mode = mode
        self.n_samples = n_samples
        self.disable_thinking = disable_thinking
        self.json_multi = json_multi
        self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        self._client = httpx.AsyncClient(timeout=timeout, limits=httpx.Limits(max_connections=concurrency + 8))
        self._sem = asyncio.Semaphore(concurrency)

    # ------------------------------------------------------------------ http
    async def _chat(self, body: dict[str, Any]) -> dict[str, Any]:
        body = {"model": self.model, **body}
        if self.disable_thinking:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        async with self._sem:
            for attempt in range(3):
                try:
                    t0 = time.perf_counter()
                    r = await self._client.post(f"{self.base_url}/chat/completions", json=body, headers=self.headers)
                    r.raise_for_status()
                    out = r.json()
                    out["_latency_ms"] = (time.perf_counter() - t0) * 1000   # measured inside the concurrency gate
                    return out
                except (httpx.HTTPError, httpx.TimeoutException) as e:  # retry transient errors
                    if attempt == 2:
                        raise
                    await asyncio.sleep(0.5 * (attempt + 1))

    @staticmethod
    def _messages(user: str) -> list[dict[str, str]]:
        return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]

    # --------------------------------------------------------- single question
    async def adecide_one(self, state: str, q: Question, mode: str | None = None) -> Decision:
        mode = mode or self.mode
        prompt, labels = llm_single_prompt(state, q)
        if mode == "logprob" and q.n > 26:
            mode = "sample"
        t0 = time.perf_counter()
        if mode in ("logprob", "greedy"):
            body = {"messages": self._messages(prompt), "max_tokens": 1 if q.n <= 26 else 4, "temperature": 0,
                    "structured_outputs": {"choice": labels}}
            if mode == "logprob":
                body.update(logprobs=True, top_logprobs=min(max(q.n, 5), 100))
            resp = await self._chat(body)
            dt = resp["_latency_ms"]
            ch = resp["choices"][0]
            content = ch["message"]["content"].strip()
            info = {"mode": mode, "usage": resp.get("usage"), "content": content}
            if mode == "logprob" and ch.get("logprobs") and ch["logprobs"].get("content"):
                lp = {lab: -math.inf for lab in labels}
                for t in ch["logprobs"]["content"][0]["top_logprobs"]:
                    tok = t["token"].strip()
                    if tok in lp and t["logprob"] > lp[tok] and t["logprob"] > -9000:
                        lp[tok] = t["logprob"]
                probs = _softmax_from_logprobs([lp[l] for l in labels])
                if probs is None:  # nothing usable; fall back to one-hot of the content
                    probs = _one_hot(labels, content)
            else:
                probs = _one_hot(labels, content)
            return Decision(probs, dt, info)
        if mode == "sample":
            body = {"messages": self._messages(prompt), "max_tokens": 4, "temperature": 1.0, "top_p": 1.0,
                    "n": self.n_samples, "structured_outputs": {"choice": labels}}
            resp = await self._chat(body)
            dt = resp["_latency_ms"]
            counts = [0] * q.n
            for ch in resp["choices"]:
                c = ch["message"]["content"].strip()
                if c in labels:
                    counts[labels.index(c)] += 1
            tot = sum(counts) or 1
            return Decision([c / tot for c in counts], dt, {"mode": "sample", "usage": resp.get("usage"), "counts": counts})
        raise ValueError(mode)

    # ---------------------------------------------------------- multi question
    async def adecide_json(self, state: str, qs: list[Question]) -> list[Decision]:
        prompt, schema = llm_multi_prompt(state, qs)
        t0 = time.perf_counter()
        body = {"messages": self._messages(prompt), "max_tokens": 64 + 16 * len(qs), "temperature": 0,
                "structured_outputs": {"json": schema}}
        resp = await self._chat(body)
        dt = resp["_latency_ms"]
        content = resp["choices"][0]["message"]["content"]
        import json
        schema_error = False
        try:
            ans = json.loads(content)
        except Exception:
            ans, schema_error = {}, True
        out = []
        for q in qs:
            v = ans.get(q.key)
            probs = [1.0 if o.id == v else 0.0 for o in q.options]
            if sum(probs) == 0:
                schema_error = True
                probs = [1.0 / q.n] * q.n
            out.append(Decision(probs, dt, {"mode": "json", "usage": resp.get("usage"), "schema_error": schema_error,
                                            "n_questions": len(qs)}))
        return out

    async def adecide(self, state: str, questions: list[Question]) -> list[Decision]:
        if self.json_multi and len(questions) > 1:
            return await self.adecide_json(state, questions)
        return list(await asyncio.gather(*[self.adecide_one(state, q) for q in questions]))

    def decide(self, state: str, questions: list[Question]) -> list[Decision]:
        return asyncio.run(self.adecide(state, questions))


def _softmax_from_logprobs(lps: list[float]) -> list[float] | None:
    m = max(lps)
    if m == -math.inf:
        return None
    ex = [math.exp(x - m) if x > -math.inf else 0.0 for x in lps]
    s = sum(ex)
    return [e / s for e in ex]


def _one_hot(labels: list[str], content: str) -> list[float]:
    c = content.strip().rstrip(".")
    return [1.0 if lab == c else 0.0 for lab in labels] if c in labels else [1.0 / len(labels)] * len(labels)
