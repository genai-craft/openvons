"""共有 ASR サーバー: kana 入力エンジン (kana-whisper / komimi) を 1 プロセスに 1 つずつ載せ、全デモから HTTP で使う.

    CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m openvons.voice.asr_server --port 8630 --preload kana-whisper,komimi-v12a

以前は音声デモ (指令・駅・河川・端末内の比較) がそれぞれ kana-whisper を読み、GPU に 4 つ載っていた (計 17 GB)。
ここに 1 つだけ載せ、各デモは openvons.voice.remote.RemoteASR で問い合わせる。

  GET  /v1/engines                       使えるエンジンと読み込み状態
  POST /v1/encode?engine=ID              本文 = 16 kHz float32 PCM (little endian)。encoder + 自由認識を 1 回で。
                                         → {handle, kana, logprob, n_tokens, tokens, seconds, ms_encode, ms_transcribe}
  POST /v1/tokenize {engine, texts}      → {ids: [[...], ...]}  (エンジンごとの語彙)
  POST /v1/score {handle, token_lists}   → {scores, counts, ms}  (log p(列 | 音声))
  GET  /healthz

encoder の出力 (kana-whisper は GPU 上の (1,1500,1280)、komimi は (T',1025) の行列) は handle で最大 128 個・5 分残す。
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from openvons.voice.engines import ENGINES, load_local  # noqa: E402

log = logging.getLogger("openvons.voice.asr_server")
app = FastAPI(title="openvons shared kana ASR")
LOADED: dict[str, Any] = {}
LOAD_LOCK = threading.Lock()
CACHE: OrderedDict[str, tuple[str, Any, float]] = OrderedDict()     # handle -> (engine, Encoded, 作成時刻)
CACHE_LOCK = threading.Lock()
MAX_CACHE, TTL = 128, 300.0
STATS: dict[str, dict[str, float]] = {}


def get_engine(eid: str):
    if eid not in ENGINES:
        raise KeyError(f"unknown engine {eid}")
    if eid in LOADED:
        return LOADED[eid]
    with LOAD_LOCK:
        if eid not in LOADED:
            t0 = time.time()
            LOADED[eid] = load_local(eid)
            log.info("loaded %s in %.1fs", eid, time.time() - t0)
    return LOADED[eid]


def _put(eid: str, enc) -> str:
    h = uuid.uuid4().hex
    now = time.time()
    with CACHE_LOCK:
        CACHE[h] = (eid, enc, now)
        while len(CACHE) > MAX_CACHE or (CACHE and now - next(iter(CACHE.values()))[2] > TTL):
            CACHE.popitem(last=False)
    return h


def _stat(eid: str, key: str, ms: float) -> None:
    s = STATS.setdefault(eid, {})
    s[key + "_n"] = s.get(key + "_n", 0) + 1
    s[key + "_ms"] = s.get(key + "_ms", 0.0) + ms


@app.get("/healthz")
def healthz():
    return {"ok": True, "loaded": list(LOADED)}


@app.get("/v1/engines")
def engines():
    out = []
    for e in ENGINES.values():
        d = e.to_dict(); d["loaded"] = e.id in LOADED
        st = STATS.get(e.id, {})
        if st.get("encode_n"):
            d["avg_encode_ms"] = round(st["encode_ms"] / st["encode_n"], 1)
        out.append(d)
    return {"engines": out}


@app.post("/v1/encode")
async def encode(request: Request, engine: str):
    body = await request.body()
    wav = np.frombuffer(body, dtype=np.float32)
    try:
        asr = get_engine(engine)
    except KeyError as e:
        return JSONResponse({"error": str(e)}, 404)
    import anyio

    def work():
        enc = asr.encode(wav)
        tr = asr.transcribe(enc)
        return enc, tr
    enc, tr = await anyio.to_thread.run_sync(work)
    h = _put(engine, enc)
    _stat(engine, "encode", enc.ms + tr.ms)
    return {"handle": h, "kana": tr.kana, "logprob": tr.logprob, "n_tokens": tr.n_tokens, "tokens": [int(t) for t in (tr.tokens or [])],
            "seconds": enc.seconds, "ms_encode": round(enc.ms, 2), "ms_transcribe": round(tr.ms, 2)}


@app.post("/v1/tokenize")
def tokenize(body: dict):
    asr = get_engine(body["engine"])
    texts = body.get("texts") or []
    tm = getattr(asr, "tokenize_many", None)
    return {"ids": tm(texts) if tm else [asr.tokenize(t) for t in texts]}


@app.post("/v1/score")
def score(body: dict):
    with CACHE_LOCK:
        item = CACHE.get(body["handle"])
    if item is None:
        return JSONResponse({"error": "handle expired"}, 410)
    eid, enc, _ = item
    t0 = time.perf_counter()
    s, c = get_engine(eid).score_tokens(enc, body.get("token_lists") or [])
    ms = (time.perf_counter() - t0) * 1000
    _stat(eid, "score", ms)
    return {"scores": [float(x) for x in s], "counts": [int(x) for x in c], "ms": round(ms, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8630)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--preload", default=os.environ.get("OPENVONS_ASR_PRELOAD", "kana-whisper,komimi-v12,komimi-v12a,komimi-v12m,komimi-v12s"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    for eid in [x for x in args.preload.split(",") if x]:
        get_engine(eid)
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
