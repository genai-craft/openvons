"""TASK-008: soft-label teacher via a local OpenAI-compatible LLM (Method A logprob / Method B sampling)."""
from __future__ import annotations

import asyncio
from typing import Iterable

from tqdm import tqdm

from jev.lm.backends.llm_backend import LLMBackend
from jev.core.formats import Sample


async def label_samples(backend: LLMBackend, samples: list[Sample], mode: str = "logprob", desc: str = "teacher") -> list[Sample]:
    out: list[Sample | None] = [None] * len(samples)
    pbar = tqdm(total=len(samples), desc=desc, mininterval=2.0)

    async def one(i: int, s: Sample):
        d = await backend.adecide_one(s.state, s.question, mode=mode)
        meta = dict(s.meta or {})
        meta[f"teacher_{backend.model}_{mode}"] = {"latency_ms": round(d.latency_ms, 1), "content": d.info.get("content")}
        out[i] = Sample(s.id, s.state, s.question, s.label, [round(p, 6) for p in d.probs], meta)
        pbar.update(1)

    await asyncio.gather(*[one(i, s) for i, s in enumerate(samples)])
    pbar.close()
    return out  # type: ignore
