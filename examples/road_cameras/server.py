"""カメラ監視 音声コマンド デモサーバー.

起動:
    CUDA_VISIBLE_DEVICES=2 .venv/bin/python examples/road_cameras/server.py --port 8600 --tts voicevox://127.0.0.1:50021

  GET  /                         UI
  WS   /ws                       音声 (PCM16 16kHz バイナリ) と制御 (JSON) の双方向
  GET  /api/hierarchy            整備局 -> 事務所 -> 路線 (登録画面用)
  GET  /api/cameras?office=..    カメラ一覧
  GET/POST/DELETE /api/scopes    担当範囲の CRUD
  POST /api/session/scope        このセッションの担当範囲を切り替え
  POST /api/say {text}           テキストを TTS で喋らせて認識に通す (マイク無しの試験用)
  POST /api/pretrain {scope_id}  事前学習を開始 (バックグラウンド)、進捗は WS の "pretrain" イベント
  GET  /api/state                状態のスナップショット
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from examples.road_cameras.app import CALIBRATION_STATES, CameraApp  # noqa: E402
from jev.voice.asr import KanaASR  # noqa: E402
from jev.voice.calibration import Calibration  # noqa: E402
from jev.voice.engine import Recognizer, Thresholds  # noqa: E402
from jev.voice.lexicon import Lexicon, Scope, ScopeStore  # noqa: E402
from jev.voice.synth import Pretrainer, PretrainConfig, TTSClient  # noqa: E402
from jev.voice.vad import StreamingVad  # noqa: E402

log = logging.getLogger("sashizu.demo")
HERE = Path(__file__).resolve().parent
STATE_DIR = Path(os.environ.get("JEV_STATE_DIR", "/data/openjev/state/road_cameras"))
STATE_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="sashizu camera demo")
G: dict[str, Any] = {}          # グローバル資源 (asr, recognizer, lexicon, scopes, tts, hierarchy)
SESSIONS: dict[str, "Session"] = {}


class Session:
    def __init__(self, sid: str, scope: Scope):
        self.id = sid
        self.app = CameraApp(G["lexicon"], scope)
        self.vad = StreamingVad()
        self.ws: WebSocket | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.busy = False

    def calibration(self) -> Calibration:
        return Calibration.from_dict(self.app.scope.profile.get("calibration"))

    def handle_utterance(self, wav: np.ndarray) -> dict[str, Any]:
        expired = self.app.expire_confirm()
        cs = self.app.command_set()
        d = G["recognizer"].recognize(wav, cs, self.calibration())
        ev = self.app.apply(d)
        ev["audio_sec"] = round(len(wav) / 16000, 2)
        if expired:
            ev["note"] = "確認待ちがタイムアウトしたため取り消しました"
        return ev

    async def send(self, msg: dict[str, Any]) -> None:
        if self.ws is not None:
            try:
                await self.ws.send_text(json.dumps(msg, ensure_ascii=False))
            except Exception:
                pass

    def send_threadsafe(self, msg: dict[str, Any]) -> None:
        if self.loop is not None:
            asyncio.run_coroutine_threadsafe(self.send(msg), self.loop)


def default_scope() -> Scope:
    st: ScopeStore = G["scopes"]
    if not st.scopes:
        st.upsert(Scope("shuto", "首都国道事務所 (千葉 R14/R357/R6)", filters={"office": ["首都国道事務所"]}))
        st.upsert(Scope("kanto_chiba", "千葉県 全域", filters={"pref": ["千葉県"]}))
    return next(iter(st.scopes.values()))


def get_session(sid: str | None) -> Session:
    if sid and sid in SESSIONS:
        return SESSIONS[sid]
    sid = sid or uuid.uuid4().hex[:8]
    SESSIONS[sid] = Session(sid, default_scope())
    return SESSIONS[sid]


# ---------------------------------------------------------------- REST
@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html", headers={"Cache-Control": "no-store"})


@app.middleware("http")
async def no_cache_static(request, call_next):
    resp = await call_next(request)
    if request.url.path.startswith("/static/"):
        resp.headers["Cache-Control"] = "no-cache, must-revalidate"
    return resp


@app.get("/api/hierarchy")
def hierarchy():
    return G["hierarchy"]


@app.get("/api/cameras")
def cameras(office: str | None = None, pref: str | None = None, route: int | None = None, q: str | None = None, limit: int = 500):
    out = []
    for e in G["lexicon"]:
        a = e.attrs
        if office and a.get("office") != office: continue
        if pref and a.get("pref") != pref: continue
        if route and a.get("route") != route: continue
        if q and q not in e.label and not any(q in r for r in e.readings): continue
        out.append({"id": e.id, "label": e.label, "readings": e.all_readings(), "attrs": a})
        if len(out) >= limit: break
    return out


@app.post("/api/cameras/{cid}/readings")
async def add_reading(cid: str, body: dict):
    ok = G["lexicon"].add_reading(cid, body.get("reading", ""))
    if ok:
        G["lexicon"].save(G["lexicon_path"])
        for s in SESSIONS.values():
            s.app.invalidate()
    e = G["lexicon"].get(cid)
    return {"ok": ok, "readings": e.all_readings() if e else []}


@app.get("/api/scopes")
def list_scopes():
    st: ScopeStore = G["scopes"]
    lex: Lexicon = G["lexicon"]
    return [{"id": s.id, "name": s.name, "filters": s.filters, "ids": s.ids, "exclude_ids": s.exclude_ids,
             "n_cameras": len(lex.in_scope(s)), "profile": s.profile} for s in st.scopes.values()]


@app.post("/api/scopes")
async def upsert_scope(body: dict):
    st: ScopeStore = G["scopes"]
    sid = body.get("id") or uuid.uuid4().hex[:6]
    old = st.get(sid)
    s = Scope(sid, body.get("name") or sid, body.get("filters") or {}, body.get("ids") or [], body.get("exclude_ids") or [],
              old.profile if old else {})
    st.upsert(s)
    for sess in SESSIONS.values():
        if sess.app.scope.id == sid:
            sess.app.set_scope(s)
    return {"ok": True, "id": sid, "n_cameras": len(G["lexicon"].in_scope(s))}


@app.delete("/api/scopes/{sid}")
def delete_scope(sid: str):
    G["scopes"].delete(sid)
    return {"ok": True}


@app.post("/api/session/scope")
async def set_session_scope(body: dict):
    sess = get_session(body.get("session"))
    s = G["scopes"].get(body.get("scope_id"))
    if s is None:
        return JSONResponse({"error": "scope not found"}, 404)
    sess.app.set_scope(s)
    return {"ok": True, "state": sess.app.snapshot()}


@app.get("/api/state")
def state(session: str | None = None):
    sess = get_session(session)
    return {"session": sess.id, "state": sess.app.snapshot()}


@app.post("/api/say")
async def say(body: dict):
    """テキスト -> TTS -> 認識。マイクのない環境や自動テストで全経路を通す。"""
    sess = get_session(body.get("session"))
    tts: TTSClient | None = G.get("tts")
    if tts is None or not tts.ok():
        return JSONResponse({"error": "TTS unavailable"}, 503)
    t0 = time.time()
    wav = await asyncio.to_thread(tts.synth, body["text"], int(body.get("seed", 1)))
    t_tts = (time.time() - t0) * 1000
    if body.get("snr_db") is not None:
        import random
        from jev.voice.synth import add_noise
        wav = add_noise(wav, float(body["snr_db"]), random.Random(0))
    ev = await asyncio.to_thread(sess.handle_utterance, wav)
    ev["tts_ms"] = round(t_tts)
    ev["session"] = sess.id
    await sess.send(ev)
    return ev


@app.post("/api/pretrain")
async def pretrain(body: dict):
    sess = get_session(body.get("session"))
    st: ScopeStore = G["scopes"]
    s = st.get(body.get("scope_id") or sess.app.scope.id)
    if s is None:
        return JSONResponse({"error": "scope not found"}, 404)
    if G.get("pretrain_busy"):
        return JSONResponse({"error": "busy"}, 409)
    if G.get("tts") is None or not G["tts"].ok():
        return JSONResponse({"error": "TTS unavailable"}, 503)
    cfg = PretrainConfig(seeds=body.get("seeds") or [1, 2, 3], carriers=body.get("carriers") or ["{camera}", "{camera}を表示"],
                         n_out_of_grammar=int(body.get("n_out_of_grammar", 12)), extra_states=CALIBRATION_STATES)
    G["pretrain_busy"] = True

    def work():
        try:
            pt = Pretrainer(G["tts"], G["recognizer"], sess.app.grammar, G["lexicon"])
            last = [0.0]

            def prog(kw):
                if time.time() - last[0] > 0.3 or kw.get("step") == kw.get("total"):
                    last[0] = time.time()
                    sess.send_threadsafe({"type": "pretrain", "status": "running", **kw})
            res = pt.run(s, ["select_camera"], cfg=cfg, progress=prog)
            s.profile = {"calibration": res.calibration, "accuracy": res.accuracy, "accuracy_after": res.accuracy_after, "n_utts": res.n_utts,
                         "none_recall": res.none_recall, "false_accept": res.false_accept, "ece_after": res.ece_after,
                         "trained_at": time.strftime("%Y-%m-%d %H:%M"), "confusions": res.confusions,
                         "added_readings": res.added_readings, "warnings": res.warnings[:20]}
            st.upsert(s)
            G["lexicon"].save(G["lexicon_path"])
            for x in SESSIONS.values():
                x.app.invalidate()
            sess.send_threadsafe({"type": "pretrain", "status": "done", "result": res.to_dict(), "state": sess.app.snapshot()})
        except Exception as ex:  # noqa: BLE001
            log.exception("pretrain failed")
            sess.send_threadsafe({"type": "pretrain", "status": "error", "error": str(ex)})
        finally:
            G["pretrain_busy"] = False

    threading.Thread(target=work, daemon=True).start()
    return {"ok": True, "started": True}


@app.get("/api/log")
def get_log(session: str | None = None):
    return get_session(session).app.log[-50:]


# ---------------------------------------------------------------- WebSocket
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    sid = ws.query_params.get("session")
    sess = get_session(sid)
    sess.ws = ws
    sess.loop = asyncio.get_running_loop()
    await sess.send({"type": "hello", "session": sess.id, "state": sess.app.snapshot(), "scopes": list_scopes()})
    speaking = False
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if msg.get("bytes") is not None:
                pcm = np.frombuffer(msg["bytes"], dtype=np.int16).astype(np.float32) / 32768.0
                utts = sess.vad.feed(pcm)
                if sess.vad.speaking != speaking:
                    speaking = sess.vad.speaking
                    await sess.send({"type": "vad", "speaking": speaking})
                for u in utts:
                    if sess.busy:
                        continue
                    sess.busy = True
                    try:
                        ev = await asyncio.to_thread(sess.handle_utterance, u)
                    finally:
                        sess.busy = False
                    await sess.send(ev)
            elif msg.get("text"):
                data = json.loads(msg["text"])
                t = data.get("type")
                if t == "state":
                    await sess.send({"type": "state", "state": sess.app.snapshot()})
                elif t == "reset":
                    sess.vad.reset()
                    sess.app.sm.goto("WALL", camera=None, pending=None)
                    await sess.send({"type": "state", "state": sess.app.snapshot()})
                elif t == "click_camera":     # マウス操作も同じ状態機械を通す
                    from jev.voice.engine import Candidate, Decision
                    from jev.voice.grammar import Hypothesis
                    h = Hypothesis("select_camera", data["label"], "", {"camera": data["id"]})
                    d = Decision("execute", Candidate(h, 1.0, 0.0), [], 0.0, "(click)", 0.0, {}, sess.app.sm.state, "UI")
                    await sess.send(sess.app.apply(d))
                elif t == "intent":          # ボタン・キー操作。音声と同じ apply() を通す (状態が許す意図だけ)
                    from jev.voice.engine import Candidate, Decision
                    from jev.voice.grammar import Hypothesis
                    name = data.get("intent", "")
                    if name in sess.app.sm.allowed_intents() and not sess.app.grammar.intents[name].patterns[0].startswith("{"):
                        it = sess.app.grammar.intents[name]
                        h = Hypothesis(name, it.description or name, "", {}, dict(it.params), it.risk)
                        d = Decision("execute", Candidate(h, 1.0, 0.0), [], 0.0, "(ui)", 0.0, {}, sess.app.sm.state, "UI")
                        await sess.send(sess.app.apply(d))
                elif t == "vad_config":
                    for k, v in data.get("config", {}).items():
                        if hasattr(sess.vad.cfg, k):
                            setattr(sess.vad.cfg, k, type(getattr(sess.vad.cfg, k))(v))
    except WebSocketDisconnect:
        pass
    finally:
        sess.ws = None


# ---------------------------------------------------------------- 起動
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8600)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--tts", default=os.environ.get("JEV_TTS_URL", "voicevox://127.0.0.1:50021"), help="voicevox:// | irodori:// | openai://")
    ap.add_argument("--cameras", default=str(HERE / "cameras.json"))
    ap.add_argument("--hierarchy", default=str(HERE / "hierarchy.json"))
    ap.add_argument("--ssl-dir", default=None, help="cert.pem/key.pem のあるディレクトリ (マイクは https か localhost が必須)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    lex_path = STATE_DIR / "cameras.json"
    if not lex_path.exists():
        import shutil
        shutil.copy(args.cameras, lex_path)
    G["lexicon_path"] = lex_path
    G["lexicon"] = Lexicon.load(lex_path)
    G["hierarchy"] = json.loads(Path(args.hierarchy).read_text(encoding="utf-8"))
    G["scopes"] = ScopeStore(STATE_DIR / "scopes.json")
    G["tts"] = TTSClient(args.tts, cache_dir=STATE_DIR / "tts_cache")
    log.info("tts backend: %s ok=%s", args.tts, G["tts"].ok())
    log.info("loading kana-whisper ...")
    G["asr"] = KanaASR()
    G["recognizer"] = Recognizer(G["asr"], Calibration(), Thresholds())
    log.info("ready: %d cameras, tts=%s", len(G["lexicon"]), G["tts"].ok())
    app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")
    import uvicorn
    kw = {}
    if args.ssl_dir:
        kw = {"ssl_certfile": f"{args.ssl_dir}/cert.pem", "ssl_keyfile": f"{args.ssl_dir}/key.pem"}
    uvicorn.run(app, host=args.host, port=args.port, log_level="info", **kw)


if __name__ == "__main__":
    main()
