"""TASK-001: POST /v1/decision  (FastAPI).

  DM_BACKENDS=llm,model   which backends to load (default: llm + model if DM_CKPT is set)
  DM_LLM_URL=http://127.0.0.1:8300/v1   DM_LLM_MODEL=qwen3-4b
  DM_CKPT=/data/decision_model/checkpoints/<exp>     decision-model checkpoint
  DM_DEFAULT=auto|model|llm

Fallback strategy (spec §11): backend=auto runs the decision model; a question whose max probability is
below `fallback_threshold` is re-asked to the generative LLM.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.responses import JSONResponse
from fastapi import FastAPI, HTTPException

from openvons.lm.api.schemas import DecisionRequest, DecisionResponse
from openvons.core.formats import state_to_text
from openvons.core.primitives import Question, Option

app = FastAPI(title="Decision API", version="0.1")
BACKENDS: dict[str, object] = {}
DEFAULT = os.environ.get("DM_DEFAULT", "auto")


@app.on_event("startup")
def _load():
    want = os.environ.get("DM_BACKENDS", "llm,model" if os.environ.get("DM_CKPT") else "llm").split(",")
    if "llm" in want:
        from openvons.lm.backends.llm_backend import LLMBackend
        BACKENDS["llm"] = LLMBackend(os.environ.get("DM_LLM_URL", "http://127.0.0.1:8300/v1"), os.environ.get("DM_LLM_MODEL", "qwen3-4b"))
    if "model" in want:
        from openvons.lm.models.decision_model import DecisionModel
        from openvons.lm.backends.model_backend import ModelBackend
        BACKENDS["model"] = ModelBackend(DecisionModel.from_checkpoint(os.environ["DM_CKPT"]), mode=os.environ.get("DM_MODE", "kv_shared"))


def _to_questions(req: DecisionRequest) -> list[Question]:
    return [Question(q.type, q.question, [Option(c.id, c.description) for c in q.choices], key=q.key) for q in req.questions]


@app.get("/healthz")
def healthz():
    return {"ok": True, "backends": list(BACKENDS)}


@app.post("/v1/decision", response_model=DecisionResponse)
async def decide(req: DecisionRequest):
    t0 = time.perf_counter()
    qs = _to_questions(req)
    state = state_to_text(req.state)
    name = req.backend or DEFAULT
    meta: dict = {}
    if name == "auto":
        if "model" not in BACKENDS:
            name = "llm"
        else:
            be = BACKENDS["model"]
            if req.mode:
                be.mode = req.mode
            decs = be.decide(state, qs)
            low = [i for i, d in enumerate(decs) if d.confidence < req.fallback_threshold]
            meta["fallback_keys"] = [qs[i].key for i in low]
            if low and "llm" in BACKENDS:
                fb = await BACKENDS["llm"].adecide(state, [qs[i] for i in low])
                for i, d in zip(low, fb):
                    decs[i] = d
            results = {q.key: q.format_result(d.probs) for q, d in zip(qs, decs)}
            for q, d in zip(qs, decs):
                results[q.key]["source"] = "llm" if q.key in meta["fallback_keys"] else "model"
            return DecisionResponse(results=results, backend="auto", latency_ms=(time.perf_counter() - t0) * 1000, meta=meta)
    if name not in BACKENDS:
        raise HTTPException(400, f"unknown backend {name}; available: {list(BACKENDS)}")
    be = BACKENDS[name]
    if req.mode:
        be.mode = req.mode
    decs = await be.adecide(state, qs)
    results = {q.key: q.format_result(d.probs) for q, d in zip(qs, decs)}
    return DecisionResponse(results=results, backend=name, latency_ms=(time.perf_counter() - t0) * 1000, meta=meta)


# ---------------------------------------------------------------- TypeSafe Jev 互換 (POST /v1/systemone)
from openvons.lm.api import systemone as _so  # noqa: E402


@app.post("/v1/systemone")
async def systemone(body: dict):
    """Jev の SDK (typesafe-sdk) の base_url をこのサーバーに向ければそのまま使える形。model は無視 (ローカルの 1 モデル)。"""
    try:
        qs = [_so.criteria_to_question(name, q) for name, q in (body.get("questions") or {}).items()]
    except ValueError as e:
        return JSONResponse({"error": {"type": "invalid_request_error", "message": str(e)}}, 422)
    if not qs:
        return JSONResponse({"error": {"type": "invalid_request_error", "message": "questions is empty"}}, 422)
    state = _so.state_to_text(body.get("state", ""))
    name = "model" if "model" in BACKENDS else "llm"       # 学習済み Decision Model があればそれ、無ければ LLM 基準
    if name not in BACKENDS:
        return JSONResponse({"error": {"type": "overloaded_error", "message": "no backend loaded"}}, 529)
    decs = await BACKENDS[name].adecide(state, qs)
    answers = {q.key: _so.answer(q, list(d.probs)) for q, d in zip(qs, decs)}
    return {"model": f"open-jev/{name}", "answers": answers, "usage": {"input_tokens": len(state) // 3, "output_tokens": 0}}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.environ.get("DM_HOST", "127.0.0.1"), port=int(os.environ.get("DM_PORT", "8400")))
