"""TypeSafe Jev 互換のワイヤフォーマット (POST /v1/systemone) を openvons.core.Question に写す.

公開ドキュメント (docs.typesafe.ai) から分かる形式だけを写している (docs/jev_api.md)。
  request:  {"state": str|object|array, "model": "jev-latest", "questions": {name: {"type": noul|choice|score, "instructions": str, "criteria": ...}}}
  response: {"model": ..., "answers": {name: {"type": ..., "noul": p} | {"type": "choice", "choice": id, "confidence": c, "probabilities": {...}}
                                          | {"type": "score", "score": e, "confidence": c, "probabilities": {"0": p0, ...}, "legend": {"0": label, ...}}},
             "usage": {"input_tokens": n, "output_tokens": 0}}
confidence は公式には「分布の集中度を 0〜1 に潰した値」とだけ定義されているので、ここでは
最大確率と 2 位との差 (margin) を採る。校正した確率をそのまま出しているので、閾値運用は probabilities 側で行うこと。
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np

from openvons.core.primitives import Option, Question


def criteria_to_question(name: str, q: dict[str, Any]) -> Question:
    t = q.get("type")
    instr = q.get("instructions") or q.get("question") or ""
    crit = q.get("criteria")
    if t == "noul":
        c = crit if isinstance(crit, dict) else {}
        return Question("noul", instr, [Option("true", str(c.get("true") or "true")), Option("false", str(c.get("false") or "false"))], key=name)
    if t == "choice":
        if isinstance(crit, dict):
            opts = [Option(k, _desc(v)) for k, v in crit.items()]
        elif isinstance(crit, list):
            opts = [Option(str(x), "") for x in crit]
        else:
            raise ValueError(f"choice '{name}' needs criteria (object or array)")
        return Question("choice", instr, opts, key=name)
    if t == "score":
        if not isinstance(crit, list) or not (2 <= len(crit) <= 10):
            raise ValueError(f"score '{name}' needs criteria array of 2..10 levels")
        return Question("score", instr, [Option(str(i), _desc(lv)) for i, lv in enumerate(crit)], key=name)
    raise ValueError(f"unknown type {t!r} for question '{name}'")


def _desc(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    return json.dumps(v, ensure_ascii=False)


def state_to_text(state: Any) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def confidence(probs: list[float]) -> float:
    p = sorted(probs, reverse=True)
    return float(p[0] - (p[1] if len(p) > 1 else 0.0))


def answer(q: Question, probs: list[float]) -> dict[str, Any]:
    if q.type == "noul":
        return {"type": "noul", "noul": float(probs[0])}
    if q.type == "choice":
        best = int(np.argmax(probs))
        return {"type": "choice", "choice": q.options[best].id, "confidence": confidence(probs),
                "probabilities": {o.id: float(p) for o, p in zip(q.options, probs)}}
    return {"type": "score", "score": float(sum(i * p for i, p in enumerate(probs))), "confidence": confidence(probs),
            "probabilities": {str(i): float(p) for i, p in enumerate(probs)}, "legend": {str(i): o.description for i, o in enumerate(q.options)}}
