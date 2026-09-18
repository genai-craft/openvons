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


#: セルフテスト用の発話 (TTS で合成)。「何が正解か」を画面に出すため、期待する答えも持つ
SAMPLES = [
    {"id": "s1", "say": "クリハシ", "show": "栗橋", "state": "MAP", "expect": "栗橋水位", "note": "地点名を選ぶ"},
    {"id": "s2", "say": "コーシンシテ", "show": "更新して", "state": "CAMERA", "expect": "更新", "note": "操作の指示"},
    {"id": "s3", "say": "ヒトツカリュー", "show": "ひとつ下流", "state": "CAMERA", "expect": "一つ下流", "note": "上流・下流の移動"},
    {"id": "s4", "say": "はい、お世話になっております", "show": "はい、お世話になっております", "state": "MAP", "expect": None, "note": "関係ない話 → 該当なしになるのが正解"},
]


@app.get("/api/samples")
def samples():
    return SAMPLES


@app.get("/api/sample/{sid}.wav")
def sample_wav(sid: str):
    """検証用の音声。TTS で作って state/ondevice/samples に置いておく (マイク無しでも経路を測れるように)。"""
    from fastapi.responses import Response
    spec = next((s for s in SAMPLES if s["id"] == sid), None)
    if spec is None:
        return JSONResponse({"error": "unknown sample"}, 404)
    d = Path(os.environ.get("OPENVONS_STATE", ROOT / "state")) / "ondevice" / "samples"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{sid}.wav"
    if not f.exists():
        import soundfile as sf
        from openvons.tts import get_backend
        tts = get_backend(os.environ.get("JEV_TTS_URL", "voicevox://127.0.0.1:50021"))
        sf.write(f, tts.synth(spec["say"], seed=1), 16000)
    return Response(f.read_bytes(), media_type="audio/wav")


@app.get("/api/commands")
def commands(app_name: str = "kasen", scope: str = "", state: str = ""):
    """状態ごとのコマンド集合 (表示文とASR形カナ)。端末側はこれと照合するだけでよい。"""
    sets = G["command_sets"][app_name]
    if state:
        return sets.get(state, {})
    return sets


@app.get("/api/tokens")
def tokens():
    """生成されうるトークンの id → バイト列。端末はこれを繋いで UTF-8 に戻すだけでよい
    (byte-level BPE なので 1 トークンが UTF-8 の途中で切れることがあり、文字列として配ると壊れる)。"""
    tk = G["tok"]
    sup = set(G["suppress"])
    out = {}
    for tid in range(G["eot"] + 1):
        if tid in sup and tid != G["eot"]:
            continue
        out[str(tid)] = list(tk.convert_ids_to_tokens(tid).encode("utf-8")) if False else list(bytes(tk.convert_tokens_to_string([tk.convert_ids_to_tokens(tid)]), "utf-8", "surrogatepass"))
    return out


@app.get("/api/vision/choices")
def vision_choices(set: str = "general"):
    """質問セット。general = 室内外の汎用、kasen = 河川カメラの監視で実際に見たい状態。
    端末に置くのは画像エンコーダだけで、質問を足すのはこの JSON を配り直すだけで済む。"""
    base = Path(os.environ.get("OPENVONS_ONDEVICE_MODELS", "/data/openjev/models/ondevice")) / "vision-choices"
    p = base / ("choices.json" if set == "general" else f"choices_{set}.json")
    if not p.exists():
        return JSONResponse({"error": f"{p.name} not found"}, 404)
    return json.loads(p.read_text(encoding="utf-8"))


@app.get("/api/vision/sets")
def vision_sets():
    base = Path(os.environ.get("OPENVONS_ONDEVICE_MODELS", "/data/openjev/models/ondevice")) / "vision-choices"
    out = []
    for f in sorted(base.glob("choices*.json")):
        doc = json.loads(f.read_text(encoding="utf-8"))
        key = "general" if f.name == "choices.json" else f.stem.replace("choices_", "")
        out.append({"key": key, "title": doc.get("title", key), "n": len(doc.get("questions", [])), "kind": "zeroshot"})
    # 学習済みヘッド (凍結した画像エンコーダの上に小さな head を足したもの)
    for f in sorted(base.glob("head_*.json")):
        doc = json.loads(f.read_text(encoding="utf-8"))
        out.append({"key": f"head:{f.stem.replace('head_', '')}", "title": doc.get("title", f.stem),
                    "n": len(doc.get("tasks", {})), "kind": "trained"})
    return out


@app.get("/api/vision/head/{task}")
def vision_head(task: str):
    """学習済みヘッドの重み。数万パラメータなので JSON のまま配り、端末側で計算する。"""
    base = Path(os.environ.get("OPENVONS_ONDEVICE_MODELS", "/data/openjev/models/ondevice")) / "vision-choices"
    p = base / f"head_{task}.json"
    if not p.exists():
        return JSONResponse({"error": f"{p.name} not found"}, 404)
    return json.loads(p.read_text(encoding="utf-8"))


@app.get("/api/sites")
def sites():
    """アプリが「指令卓」を描くための地点一覧。読み・河川・上下流の順番まで含めて配る。
    端末はこれだけで、地点の切り替えと上流・下流の移動を自分で行える。"""
    lex = G.get("lexicon")
    if lex is None:
        return JSONResponse({"error": "lexicon not loaded"}, 503)
    ents = sorted(lex.in_scope(G["scope"]),
                  key=lambda e: (e.attrs.get("river", ""), e.attrs.get("order") or 0))
    out = []
    for i, e in enumerate(ents, 1):
        a = e.attrs
        # コードは並び順で振る (アプリ側と同じ規則)。名前を覚えていなくても「C06」で呼べる
        out.append({"id": e.id, "code": f"C{i:02d}", "label": e.label, "reading": (e.readings or [""])[0],
                    "river": a.get("river", ""), "office": a.get("office", ""), "pref": a.get("pref", ""),
                    "order": a.get("order"), "lat": a.get("lat"), "lng": a.get("lng"),
                    "image": f"/api/image/{e.id}"})
    return {"scope": G["scope"].name, "attribution": G.get("attribution", ""), "sites": out}


@app.get("/api/manifest")
def manifest():
    """アプリが最初に取りに来る一覧: 落とすファイルとそのサイズ。"""
    base = Path(os.environ.get("OPENVONS_ONDEVICE_MODELS", "/data/openjev/models/ondevice"))
    def files(rel: list[str]) -> list[dict]:
        out = []
        for r in rel:
            f = base / r
            if f.exists():
                out.append({"path": r, "bytes": f.stat().st_size, "url": f"/model/{r}"})
        return out
    voice = G["model_list"][-1]
    return {
        "voice": {"name": voice, "files": files([f"{voice}/onnx/mel.onnx", f"{voice}/onnx/encoder_model.onnx", f"{voice}/onnx/decoder_model.onnx",
                                                 f"{voice}/onnx/encoder_model_quantized.onnx", f"{voice}/onnx/decoder_model_quantized.onnx",
                                                 # 採点と次トークンをグラフの中で潰して返す版 (アプリはこちらを使う)
                                                 f"{voice}/onnx/decoder_head_quantized.onnx"])},
        "vision": {"name": "siglip2-base-img", "files": files(["siglip2-base-img/image_encoder.onnx", "siglip2-base-img/image_encoder_quantized.onnx",
                                                               "siglip2-base-img/preprocess.json"])},
    }


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
            # ids = カナをトークン化したもの。端末に BPE を実装しなくて済むよう、サーバーで済ませる
            "hyps": [{"t": h.text, "k": h.kana, "i": h.intent, "s": h.slots, "r": h.risk, "p": h.params,
                      "c": h.confirmable, "y": h.positive,
                      "ids": G["tok"].encode(h.kana, add_special_tokens=False)} for h in cs.hyps],
        }
    inst.sm.state = list(APP.STATES)[0]
    G["lexicon"] = lex
    G["scope"] = scope
    G["attribution"] = getattr(APP, "ATTRIBUTION", "")
    G["app_module"] = APP
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
    from transformers import WhisperTokenizerFast
    tk = WhisperTokenizerFast.from_pretrained(str(models / G["model_list"][-1]))
    G["tok"] = tk
    G["prefix"] = tk.convert_tokens_to_ids(["<|startoftranscript|>", "<|ja|>", "<|transcribe|>", "<|notimestamps|>"])
    G["eot"] = tk.eos_token_id
    info_p = models / G["model_list"][-1] / "distill_info.json"
    info = json.loads(info_p.read_text()) if info_p.exists() else {}
    G["suppress"] = info.get("suppress_tokens_kana_only", [])
    sets, cs_by_state, scope = build_command_sets(args.app)
    G["command_sets"] = {"kasen": sets}
    G["cs_by_state"] = cs_by_state
    prof = (scope.profile or {}).get("calibration") or {}
    G["calibration"] = {"temperature": prof.get("temperature", 2.5), "none_bias": prof.get("none_bias", 4.0),
                        "len_bonus": prof.get("len_bonus", 1.4), "residual_penalty": prof.get("residual_penalty", 1.0)}
    G["server_asr"] = bool(args.server_asr)
    if args.server_asr:
        from openvons.core.none_calibration import Calibration
        from openvons.voice.asr import KanaASR
        from openvons.voice.engine import Recognizer
        G["recognizer"] = Recognizer(KanaASR())
        G["cal_obj"] = Calibration.from_dict(G["calibration"])
    log.info("models: %s  states: %s", G["model_list"], {k: v["n"] for k, v in sets.items()})
    # アプリ固有のルート (kasen ならライブ画像の中継)。端末アプリが指令卓を描くのに要る
    if hasattr(G.get("app_module"), "register_routes"):
        G["app_module"].register_routes(app, G, Path(os.environ.get("OPENVONS_STATE", ROOT / "state")) / "ondevice")
    app.mount("/model", StaticFiles(directory=str(models)), name="model")
    app.mount("/static", StaticFiles(directory=str(static)), name="static")
    app.mount("/shared", StaticFiles(directory=str(ROOT / "openvons" / "voice" / "demo_static")), name="shared")
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
