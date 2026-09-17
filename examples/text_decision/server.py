"""テキストの判断デモ (openvons.lm): 状態 (文章) を 1 回読ませて、有限の選択肢への質問にまとめて確率で答えさせる.

    .venv/bin/python -m examples.text_decision.server --port 8604 --llm http://127.0.0.1:8300/v1 --model qwen3-4b

3 つの答え方を同じ質問で比べられる:
  確率 (openvons)   … guided choice + 先頭トークンの logprob。1 質問 1 forward、質問は並列。確率が出る
  JSON 生成         … 全質問を 1 回の JSON 生成で。普通の LLM の使い方。確率は出ない (one-hot)
  structured output … guided choice を温度 0 で 1 つだけ。確率は出ない

「状態は 1 回、質問は何個でも」の効果は /api/fanout で実測 (質問数 1/2/4/8 の所要時間)。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from openvons.core.decision import Thresholds, decide
from openvons.lm.api.systemone import answer as so_answer
from openvons.lm.api.systemone import criteria_to_question
from openvons.lm.backends.llm_backend import LLMBackend

log = logging.getLogger("openvons.text.demo")
app = FastAPI(title="openvons (open-Jev) text decision demo")
HERE = Path(__file__).resolve().parent
G: dict[str, Any] = {}
TH = Thresholds(execute=0.85, confirm=0.5)
LEVEL_JA = {"execute": "確定", "confirm": "要確認", "reject": "不明"}


def backend(model: str, json_multi: bool = False) -> LLMBackend:
    url = G["models"][model]
    key = (model, json_multi)
    if key not in G["backends"]:
        G["backends"][key] = LLMBackend(url, model, mode="logprob", json_multi=json_multi)
    return G["backends"][key]


def to_questions(rows: list[dict]) -> list:
    qs = []
    for i, r in enumerate(rows):
        q = dict(r)
        key = q.pop("key", None) or f"q{i + 1}"
        qs.append(criteria_to_question(key, q))
    return qs


async def run(state: str, rows: list[dict], mode: str, model: str) -> dict[str, Any]:
    qs = to_questions(rows)
    t0 = time.perf_counter()
    if mode == "json":
        decs = await backend(model, json_multi=True).adecide_json(state, qs)
    else:
        be = backend(model)
        decs = list(await asyncio.gather(*[be.adecide_one(state, q, mode=("greedy" if mode == "greedy" else "logprob")) for q in qs]))
    wall = (time.perf_counter() - t0) * 1000
    out = []
    for q, d in zip(qs, decs):
        a = so_answer(q, list(d.probs))
        top = max(d.probs)
        action, _ = decide(top, 0.0, "low", TH)
        out.append({"key": q.key, "type": q.type, "question": q.text, "answer": a, "level": LEVEL_JA[action],
                    "top_prob": round(float(top), 4), "latency_ms": round(d.latency_ms, 1),
                    "options": [{"id": o.id, "description": o.description} for o in q.options],
                    "probs": [round(float(p), 5) for p in d.probs], "info": {k: v for k, v in (d.info or {}).items() if k in ("mode", "schema_error")}})
    usage = sum((d.info or {}).get("usage", {}).get("prompt_tokens", 0) or 0 for d in decs)
    return {"mode": mode, "model": model, "wall_ms": round(wall, 1), "prompt_tokens": usage, "results": out}


@app.get("/")
def index():
    html = (HERE / "static" / "index.html").read_text(encoding="utf-8").replace("__V__", G["version"])
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})


@app.get("/api/presets")
def presets():
    return json.loads((HERE / "presets.json").read_text(encoding="utf-8"))


@app.get("/api/models")
def models():
    return {"models": list(G["models"]), "default": G["default_model"]}


@app.post("/api/decide")
async def api_decide(body: dict):
    try:
        return await run(body.get("state", ""), body.get("questions") or [], body.get("mode", "logprob"), body.get("model") or G["default_model"])
    except Exception as e:  # noqa: BLE001
        log.exception("decide failed")
        return JSONResponse({"error": str(e)}, 500)


@app.post("/api/fanout")
async def api_fanout(body: dict):
    """質問数を 1/2/4/8 と増やしたときの所要時間。確率方式は状態を 1 回読んで質問は並列、JSON 生成は 1 回の生成に全部入る。"""
    state = body.get("state", ""); rows = body.get("questions") or []; model = body.get("model") or G["default_model"]
    out = []
    for n in (1, 2, 4, 8):
        pool = (rows * ((n // max(len(rows), 1)) + 1))[:n]
        sub = [{**q, "key": f"{q.get('key', 'q')}_{i}"} for i, q in enumerate(pool)]   # JSON schema のキー衝突を避ける
        r_prob = await run(state, sub, "logprob", model)
        r_json = await run(state, sub, "json", model)
        out.append({"n": n, "prob_ms": r_prob["wall_ms"], "json_ms": r_json["wall_ms"]})
    return {"model": model, "points": out}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8604)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--llm", default=os.environ.get("OPENVONS_LLM_URL", "http://127.0.0.1:8300/v1"))
    ap.add_argument("--model", default=os.environ.get("OPENVONS_LLM_MODEL", "qwen3-4b"))
    ap.add_argument("--llm2", default=os.environ.get("OPENVONS_LLM2_URL", "http://127.0.0.1:8301/v1"))
    ap.add_argument("--model2", default=os.environ.get("OPENVONS_LLM2_MODEL", "qwen3.8-27b"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    G["models"] = {args.model: args.llm}
    if args.llm2 and args.model2:
        G["models"][args.model2] = args.llm2
    G["default_model"] = args.model
    G["backends"] = {}
    static = HERE / "static"
    G["version"] = str(int(max(p.stat().st_mtime for p in static.glob("*"))))
    app.mount("/static", StaticFiles(directory=str(static)), name="static")
    app.mount("/shared", StaticFiles(directory=str(Path(__file__).resolve().parents[2] / "openvons" / "voice" / "demo_static")), name="shared")
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
