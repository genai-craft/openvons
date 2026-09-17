"""端末内推論のデモ (openvons.voice): ブラウザの中で ASR と候補採点を動かす.

    .venv/bin/python -m examples.ondevice.server --port 8606 --models /data/openjev/models/ondevice

ページは transformers.js (WebGPU / WASM) で ONNX の kana ASR を読み、
  自由認識 → 候補の絞り込み → 強制トークン採点 → 校正 → 判断
までを端末の中で行う。サーバーは (1) モデルファイル (2) 状態ごとのコマンド集合 (読み付き) だけを配る。
比較用に、同じ音声をサーバー側 (kana-whisper 809M) に投げる経路も残してある。
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

log = logging.getLogger("openvons.ondevice")
app = FastAPI(title="openvons on-device demo")
HERE = Path(__file__).resolve().parent
G: dict[str, Any] = {}


@app.middleware("http")
async def cross_origin_isolation(request, call_next):
    """SharedArrayBuffer を有効にして WASM をマルチスレッドで動かす (単スレッドだと encoder が数秒かかる)。
    COEP は credentialless にして、CDN の CORS 付きリソースをそのまま読めるようにする。"""
    resp = await call_next(request)
    resp.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    resp.headers["Cross-Origin-Embedder-Policy"] = "credentialless"
    return resp


@app.get("/")
def index():
    html = (HERE / "static" / "index.html").read_text(encoding="utf-8").replace("__V__", G["version"])
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})


@app.get("/api/config")
def config():
    """端末側が必要とするもの: モデル一覧、校正値、特殊トークンの id、非カナの抑制リスト。
    トークナイザの内部構造に依存しないよう、id はサーバーで解決して渡す。"""
    return {"models": G["model_list"], "calibration": G["calibration"], "server_asr": G["server_asr"],
            "prefix": G["prefix"], "eot": G["eot"], "suppress": G["suppress"]}


@app.get("/api/sample_wav")
def sample_wav():
    """検証用: 手元の TTS キャッシュから 1 つ返す (マイクの無い環境で経路を通すため)。"""
    from fastapi.responses import Response
    import glob
    fs = sorted(glob.glob(str(ROOT / "state" / "kasen" / "tts_cache" / "*.wav")))
    if not fs:
        return JSONResponse({"error": "no cached wav"}, 404)
    idx = int(os.environ.get("OPENVONS_SAMPLE_INDEX", "0")) % len(fs)
    return Response(open(fs[idx], "rb").read(), media_type="audio/wav")


@app.get("/api/commands")
def commands(app_name: str = "kasen", scope: str = "", state: str = ""):
    """状態ごとのコマンド集合 (表示文とASR形カナ)。端末側はこれと照合するだけでよい。"""
    sets = G["command_sets"][app_name]
    if state:
        return sets.get(state, {})
    return sets


@app.post("/api/server_decide")
def server_decide(body: dict):
    """比較用: 同じ音声をサーバーの kana-whisper で認識して判断する。"""
    if not G.get("recognizer"):
        return JSONResponse({"error": "server ASR not loaded"}, 503)
    import soundfile as sf
    wav, sr = sf.read(io.BytesIO(base64.b64decode(body["audio_base64"])))
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(1)
    cs = G["cs_by_state"][body.get("state", "MAP")]
    d = G["recognizer"].recognize(wav, cs, G["cal_obj"])
    return d.to_dict()


def build_command_sets(app_module: str):
    """アプリの状態ごとにコマンド集合を作り、(表示文, カナ, 意図, スロット) の一覧にする。"""
    import importlib
    from openvons.voice.lexicon import Lexicon, Scope
    APP = importlib.import_module(app_module)
    app_dir = Path(APP.__file__).resolve().parent
    lex = Lexicon.load(app_dir / APP.DATA_FILE)
    scope = APP.default_scopes(lex)[0]
    inst = APP.AppClass(lex, scope)
    out: dict[str, dict] = {}
    cs_by_state = {}
    for state in APP.STATES:
        inst.sm.state = state
        cs = inst.command_set()
        cs_by_state[state] = cs
        out[state] = {
            "state": state,
            "description": APP.STATES[state].description,
            "n": len(cs),
            "hyps": [{"t": h.text, "k": h.kana, "i": h.intent, "s": h.slots, "r": h.risk} for h in cs.hyps],
        }
    inst.sm.state = list(APP.STATES)[0]
    return out, cs_by_state, scope


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8606)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--models", default=os.environ.get("OPENVONS_ONDEVICE_MODELS", "/data/openjev/models/ondevice"))
    ap.add_argument("--app", default="examples.kasen.app")
    ap.add_argument("--server-asr", action="store_true", help="比較用にサーバー側 kana-whisper も読む")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    static = HERE / "static"
    G["version"] = str(int(max(p.stat().st_mtime for p in static.glob("*"))))
    models = Path(args.models)
    G["model_list"] = sorted(p.name for p in models.iterdir() if (p / "onnx").is_dir())
    sets, cs_by_state, scope = build_command_sets(args.app)
    G["command_sets"] = {"kasen": sets}
    G["cs_by_state"] = cs_by_state
    prof = (scope.profile or {}).get("calibration") or {}
    G["calibration"] = {"temperature": prof.get("temperature", 2.5), "none_bias": prof.get("none_bias", 4.0),
                        "len_bonus": prof.get("len_bonus", 1.4), "residual_penalty": prof.get("residual_penalty", 1.0)}
    G["server_asr"] = bool(args.server_asr)
    from transformers import WhisperTokenizerFast
    tk = WhisperTokenizerFast.from_pretrained(str(models / G["model_list"][0]))
    G["prefix"] = tk.convert_tokens_to_ids(["<|startoftranscript|>", "<|ja|>", "<|transcribe|>", "<|notimestamps|>"])
    G["eot"] = tk.eos_token_id
    info = json.loads((models / G["model_list"][0] / "distill_info.json").read_text()) if (models / G["model_list"][0] / "distill_info.json").exists() else {}
    G["suppress"] = info.get("suppress_tokens_kana_only", [])
    if args.server_asr:
        from openvons.core.none_calibration import Calibration
        from openvons.voice.asr import KanaASR
        from openvons.voice.engine import Recognizer
        G["recognizer"] = Recognizer(KanaASR())
        G["cal_obj"] = Calibration.from_dict(G["calibration"])
    log.info("models: %s  states: %s", G["model_list"], {k: v["n"] for k, v in sets.items()})
    app.mount("/model", StaticFiles(directory=str(models)), name="model")
    app.mount("/static", StaticFiles(directory=str(static)), name="static")
    app.mount("/shared", StaticFiles(directory=str(ROOT / "openvons" / "voice" / "demo_static")), name="shared")
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
