"""画像用 Decision API。検証用スマホアプリはこのエンドポイントを叩けばよい。

  POST /v1/vision/decision   multipart(file=画像) または JSON {"image_base64": "..."}
      -> {"results": {"age": {"probabilities": {...}, "choice": "...", "confidence": 0.9}, ...}}

環境変数:
  DMV_KIND=vision|vlm            どちらの小型モデルを載せるか (default: vision)
  DMV_CKPT=/data/decision_model/checkpoints/<exp>
  DMV_TASKS=/data/decision_model/data/vision/fairface/tasks.json   (vlm のとき質問定義に使う)
  DMV_PORT=8500
"""
from __future__ import annotations

import base64
import io
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

app = FastAPI(title="Vision Decision API", version="0.1")
STATE: dict = {}


class ImageRequest(BaseModel):
    image_base64: str
    questions: list[dict] | None = None      # vlm のときだけ自由質問を受け付ける


@app.on_event("startup")
def _load():
    kind = os.environ.get("DMV_KIND", "vision")
    ckpt = os.environ["DMV_CKPT"]
    STATE["kind"] = kind
    if kind == "vision":
        from openvons.vision.vision_model import VisionDecisionModel
        m = VisionDecisionModel.from_checkpoint(ckpt)
        STATE["model"] = m
        STATE["questions"] = m.cfg.questions
        STATE["params"] = {"frozen_M": round(m.n_frozen() / 1e6, 1), "trainable_K": round(m.n_trainable() / 1e3, 1)}
    else:
        from openvons.core.primitives import Question
        from openvons.vision.vlm_decision_model import VLMDecisionModel
        m = VLMDecisionModel.from_checkpoint(ckpt)
        STATE["model"] = m
        tasks = json.load(open(os.environ["DMV_TASKS"]))["questions"]
        STATE["questions"] = tasks
        STATE["default_qs"] = [Question.from_dict(v, key=k) for k, v in tasks.items()]
        STATE["params"] = {"frozen_B": round(sum(p.numel() for p in m.backbone.parameters()) / 1e9, 2),
                           "trainable_M": round(sum(p.numel() for p in m.head.parameters()) / 1e6, 2)}


@app.get("/healthz")
def healthz():
    return {"ok": True, "kind": STATE.get("kind"), "questions": list(STATE.get("questions") or {}), "params": STATE.get("params")}


def _format(key: str, probs: list[float], choices: list[dict]) -> dict:
    ids = [c["id"] for c in choices][: len(probs)]
    best = max(range(len(probs)), key=lambda i: probs[i])
    return {"probabilities": {i: round(float(p), 5) for i, p in zip(ids, probs)},
            "choice": ids[best], "confidence": round(float(probs[best]), 5)}


def _run(img: Image.Image, questions: list[dict] | None) -> dict:
    m = STATE["model"]
    t0 = time.perf_counter()
    if STATE["kind"] == "vision":
        out = m.decide([img])
        results = {k: _format(k, v[0].tolist(), STATE["questions"][k]["choices"]) for k, v in out.items()}
    else:
        from openvons.core.primitives import Question
        qs = [Question.from_dict(q, key=q.get("key", f"q{i}")) for i, q in enumerate(questions)] if questions else STATE["default_qs"]
        probs = m.decide(img, qs)
        results = {q.key: _format(q.key, p, [{"id": o.id} for o in q.options]) for q, p in zip(qs, probs)}
    return {"results": results, "latency_ms": round((time.perf_counter() - t0) * 1000, 1), "kind": STATE["kind"]}


@app.post("/v1/vision/decision")
async def decide_upload(file: UploadFile = File(...)):
    try:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"cannot read image: {e}")
    return _run(img, None)


@app.post("/v1/vision/decision_json")
def decide_json(req: ImageRequest):
    try:
        img = Image.open(io.BytesIO(base64.b64decode(req.image_base64))).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"cannot decode image: {e}")
    return _run(img, req.questions)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.environ.get("DMV_HOST", "0.0.0.0"), port=int(os.environ.get("DMV_PORT", "8500")))
