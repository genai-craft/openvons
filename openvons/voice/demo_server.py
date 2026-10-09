"""音声コマンド デモサーバー (アプリ差し替え可).

起動:
    CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m openvons.voice.demo_server --app examples.road_cameras.app --port 8600
    CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m openvons.voice.demo_server --app examples.stations.app     --port 8601

アプリモジュールの契約: AppClass (command_set/apply/snapshot/set_scope/invalidate/expire_confirm/entities/allowed_commands/grammar/sm/log),
CALIBRATION_STATES, SLOT, DATA_FILE, default_scopes(lexicon), TITLE。静的 UI はモジュールと同じディレクトリの static/。

  GET  /                         UI
  WS   /ws                       音声 (PCM16 16kHz バイナリ) と制御 (JSON) の双方向
  GET  /api/hierarchy            整備局 -> 事務所 -> 路線 (登録画面用)
  GET  /api/cameras?office=..    カメラ一覧
  GET/POST/DELETE /api/scopes    担当範囲の CRUD
  POST /api/session/scope        このセッションの担当範囲を切り替え
  POST /api/say {text}           テキストを TTS で喋らせて認識に通す (マイク無しの試験用)
  POST /api/pretrain {scope_id}  事前学習を開始 (バックグラウンド)、進捗は WS の "pretrain" イベント
  GET  /api/state                状態のスナップショット
  GET  /api/kana/config           kana 入力エンジンの一覧 (サーバー / ブラウザ)、ブラウザ用モデル・語彙・WASM の場所
  GET  /api/kana/commands         いまの状態のコマンド集合 (ブラウザの WebAssembly 振り分け用) と校正値
  WS   {"type":"engine"}          このセッションの kana 入力エンジンを切り替える
  WS   {"type":"decision"}        ブラウザで振り分けた結果 (仮説 index と確率) を状態機械に通す

kana 入力は --asr で共有 ASR サーバー (openvons.voice.asr_server) を指せば、そこに載ったエンジンを使う (このプロセスは GPU を使わない)。
--asr local なら必要なエンジンをこのプロセスに読む。
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
import importlib  # noqa: E402
from openvons.core.none_calibration import Calibration  # noqa: E402
from openvons.voice.engine import Candidate, Decision, Recognizer, Thresholds  # noqa: E402
from openvons.voice.engines import DEFAULT_ENGINE, ENGINES, default_calibration, load_local  # noqa: E402
from openvons.voice.lexicon import Lexicon, Scope, ScopeStore  # noqa: E402
from openvons.voice.synth import Pretrainer, PretrainConfig, TTSClient  # noqa: E402
from openvons.voice.vad import StreamingVad  # noqa: E402

log = logging.getLogger("openvons.voice.demo")
APP = None            # アプリモジュール (main で読み込む)
HERE = Path(__file__).resolve().parent
STATE_ROOT = Path(os.environ.get("OPENVONS_STATE", Path(__file__).resolve().parents[2] / "state"))
STATE_DIR = Path(os.environ.get("JEV_STATE_DIR", STATE_ROOT / "demo"))

app = FastAPI(title="jev voice demo")
G: dict[str, Any] = {}          # グローバル資源 (asr, recognizer, lexicon, scopes, tts, hierarchy)
SESSIONS: dict[str, "Session"] = {}


RISK_CODE = {"low": 0, "medium": 1, "high": 2}


def recognizer(engine: str) -> Recognizer:
    """エンジンごとの Recognizer (全セッションで共有。状態は持たない)。共有 ASR サーバーがあればそこへ問い合わせる。"""
    rec = G["recognizers"].get(engine)
    if rec is None:
        if G["asr_url"]:
            from openvons.voice.remote import RemoteASR
            asr = RemoteASR(G["asr_url"], engine)
        else:
            asr = load_local(engine)
        rec = G["recognizers"][engine] = Recognizer(asr, Calibration.from_dict(default_calibration(engine)), Thresholds())
    return rec


def engine_calibration(scope: Scope, engine: str) -> tuple[dict, bool]:
    """(校正値, 事前学習済みか)。範囲の事前学習はエンジンごと (尤度の尺度が違う)。kana-whisper は旧形式 (profile.calibration) も読む。"""
    prof = scope.profile or {}
    cal = (prof.get("calibrations") or {}).get(engine)
    if cal is None and engine == "kana-whisper":
        cal = prof.get("calibration")
    return (cal, True) if cal else (default_calibration(engine), False)


def cs_key(app_inst, cs) -> str:
    return f"{app_inst.scope.id}:{cs.state}:{id(cs):x}:{len(cs)}"


class Session:
    def __init__(self, sid: str, scope: Scope):
        self.id = sid
        self.app = APP.AppClass(G["lexicon"], scope)
        self.vad = StreamingVad()
        self.ws: WebSocket | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.busy = False
        self.engine = G["default_engine"]

    def calibration(self, engine: str | None = None) -> Calibration:
        return Calibration.from_dict(engine_calibration(self.app.scope, engine or self.engine)[0])

    def handle_utterance(self, wav: np.ndarray, source: str = "mic") -> dict[str, Any]:
        expired = self.app.expire_confirm()
        cs = self.app.command_set()
        d = recognizer(self.engine).recognize(wav, cs, self.calibration())
        ev = self.app.apply(d)
        ev["engine"] = self.engine; ev["route"] = "server"
        if source == "mic" and os.environ.get("JEV_DUMP_UTTS", "1") == "1":
            # 実音声の収集 (再校正・実測用)。wav と判断結果を並べて保存する
            try:
                import soundfile as sf
                d_dir = STATE_DIR / "utts"; d_dir.mkdir(parents=True, exist_ok=True)
                stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{self.id}_{int(time.time() * 1000) % 1000:03d}"
                sf.write(d_dir / f"{stamp}.wav", wav, 16000)
                (d_dir / f"{stamp}.json").write_text(json.dumps({"decision": d.to_dict(), "state_before": d.state, "scope": self.app.scope.id, "engine": self.engine}, ensure_ascii=False), encoding="utf-8")
            except Exception:
                log.exception("dump failed")
        ev["audio_sec"] = round(len(wav) / 16000, 2)
        if expired:
            ev["note"] = "確認待ちがタイムアウトしたため取り消しました"
        return ev

    def apply_client_decision(self, data: dict[str, Any]) -> dict[str, Any]:
        """ブラウザ (WebAssembly) で振り分けた結果を状態機械に通す。ブラウザは仮説の index と確率だけを送り、
        意味ごとの合算と判断 (実行 / 確認 / 棄却) はここでやり直す (同じ規則。閾値はサーバーが持つ)。
        コマンド集合が変わっていたら (確認の時間切れ・別の操作) stale を返し、ブラウザが同じ音声で振り分け直す。"""
        expired = self.app.expire_confirm()
        cs = self.app.command_set()
        key = cs_key(self.app, cs)
        if data.get("cs_key") != key:
            return {"type": "stale", "cs_key": key, "req": data.get("req")}
        cands = [(int(i), float(p), float(sc)) for i, p, sc, *_ in data.get("cands") or [] if 0 <= int(i) < len(cs.hyps)]
        none_prob = float(data.get("none_prob", 1.0))
        agg: dict[tuple, Candidate] = {}
        for i, p, sc in cands:
            h = cs.hyps[i]
            c = agg.get(h.meaning)
            if c is None:
                agg[h.meaning] = Candidate(h, p, sc, {h.kana: p})
            else:
                c.prob += p; c.surface_probs[h.kana] = p
                if sc > c.score:
                    c.score = sc; c.hypothesis = h
        ranked = sorted(agg.values(), key=lambda c: -c.prob)
        top = ranked[0] if ranked else None
        rec = recognizer(self.engine) if G["recognizers"] else None
        if top is None:
            action, reason = "none", data.get("reason") or "no candidates"
        else:
            action, reason = (rec or Recognizer(None))._decide(top, none_prob)
        timings = {k: float(v) for k, v in (data.get("timings_ms") or {}).items()}
        d = Decision(action, top, ranked, none_prob, data.get("free_kana", ""), float(data.get("free_score", 0.0)), timings, cs.state, reason)
        ev = self.app.apply(d)
        ev["engine"] = data.get("engine"); ev["route"] = "wasm"; ev["audio_sec"] = data.get("audio_sec")
        if data.get("action") and data["action"] != action:
            ev["route_note"] = f"ブラウザの判断 {data['action']} をサーバーの規則で {action} に"
        if expired:
            ev["note"] = "確認待ちがタイムアウトしたため取り消しました"
        if os.environ.get("JEV_DUMP_UTTS", "1") == "1":
            try:
                d_dir = STATE_DIR / "utts"; d_dir.mkdir(parents=True, exist_ok=True)
                stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{self.id}_{int(time.time() * 1000) % 1000:03d}_wasm"
                (d_dir / f"{stamp}.json").write_text(json.dumps({"decision": d.to_dict(), "state_before": d.state, "scope": self.app.scope.id,
                                                                 "engine": data.get("engine"), "route": "wasm"}, ensure_ascii=False), encoding="utf-8")
            except Exception:
                log.exception("dump failed")
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
        for sc in APP.default_scopes(G["lexicon"]):
            st.upsert(sc)
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
    from fastapi.responses import HTMLResponse
    html = (G["static"] / "index.html").read_text(encoding="utf-8").replace("__V__", G.get("version", "0"))
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})


@app.middleware("http")
async def no_cache_static(request, call_next):
    resp = await call_next(request)
    if request.url.path.startswith(("/static/", "/shared/")):
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
    if not st.scopes:
        default_scope()
    return [{"id": s.id, "name": s.name, "filters": s.filters, "ids": s.ids, "exclude_ids": s.exclude_ids,
             "n_cameras": len(APP.AppClass(lex, s).entities()), "profile": s.profile} for s in st.scopes.values()]


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
    return {"ok": True, "id": sid, "n_cameras": len(APP.AppClass(G["lexicon"], s).entities())}


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
    text = body["text"]
    if body.get("readingize", True):
        # 実体の表示名 (漢字) を登録読み (カナ) に置き換えてから合成する。TTS の G2P は固有名詞に弱く、
        # 「田端」を「タハシ」と読むなど、ASR ではなく TTS の誤りを測ってしまうため
        for e in sorted(sess.app.entities(), key=lambda e: -len(e.label)):
            if e.label and e.label in text:
                text = text.replace(e.label, e.primary_readings()[0])
        for word, reading in getattr(APP, "TTS_HINTS", {}).items():   # TTS が誤読する一般語 (下流 → カリウエ 等)
            text = text.replace(word, reading)
    wav = await asyncio.to_thread(tts.synth, text, int(body.get("seed", 1)))
    t_tts = (time.time() - t0) * 1000
    if body.get("snr_db") is not None:
        import random
        from openvons.voice.synth import add_noise
        wav = add_noise(wav, float(body["snr_db"]), random.Random(0))
    if body.get("recognize") is False:
        # ブラウザで振り分ける試験用: 合成した音声 (16 kHz PCM16) だけ返す
        import base64
        pcm = (np.clip(wav, -1, 1) * 32767).astype("<i2").tobytes()
        return {"pcm16_b64": base64.b64encode(pcm).decode(), "sr": 16000, "tts_ms": round(t_tts), "tts_text": text}
    ev = await asyncio.to_thread(sess.handle_utterance, wav, "tts")
    ev["tts_ms"] = round(t_tts)
    ev["tts_text"] = text
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
    slot = APP.SLOT
    cfg = PretrainConfig(seeds=body.get("seeds") or [1, 2, 3], carriers=body.get("carriers") or ["{%s}" % slot, "{%s}を表示" % slot],
                         n_out_of_grammar=int(body.get("n_out_of_grammar", 12)), extra_states=APP.CALIBRATION_STATES)
    G["pretrain_busy"] = True
    engine = body.get("engine") or sess.engine
    if engine not in ENGINES:
        G["pretrain_busy"] = False
        return JSONResponse({"error": f"unknown engine {engine}"}, 400)
    # 読みの追加 (TTS 往復で聞こえた読みを辞書に足す) は kana-whisper のときだけ。小さいエンジンの聞き違いを全エンジン共通の辞書に入れない
    cfg.add_readings = engine == "kana-whisper"

    def work():
        try:
            pt = Pretrainer(G["tts"], recognizer(engine), sess.app.grammar, G["lexicon"])
            last = [0.0]

            def prog(kw):
                if time.time() - last[0] > 0.3 or kw.get("step") == kw.get("total"):
                    last[0] = time.time()
                    sess.send_threadsafe({"type": "pretrain", "status": "running", **kw})
            res = pt.run(s, [APP.SELECT_INTENT], slot=APP.SLOT, cfg=cfg, progress=prog, entities=sess.app.entities())
            summary = {"calibration": res.calibration, "accuracy": res.accuracy, "accuracy_after": res.accuracy_after, "n_utts": res.n_utts,
                       "none_recall": res.none_recall, "false_accept": res.false_accept, "ece_after": res.ece_after,
                       "trained_at": time.strftime("%Y-%m-%d %H:%M"), "confusions": res.confusions,
                       "added_readings": res.added_readings, "warnings": res.warnings[:20]}
            prof = dict(s.profile or {})
            prof.setdefault("calibrations", {})[engine] = res.calibration
            prof.setdefault("engines", {})[engine] = {k: v for k, v in summary.items() if k not in ("confusions", "warnings")}
            if engine == "kana-whisper" or "accuracy" not in prof:
                prof.update(summary)          # 画面の「事前学習済み」表示は従来どおり (最後に kana-whisper で学習した結果)
            prof["engine"] = engine if engine != "kana-whisper" else prof.get("engine", "kana-whisper")
            s.profile = prof
            st.upsert(s)
            G["lexicon"].save(G["lexicon_path"])
            for x in SESSIONS.values():
                x.app.invalidate()
            sess.send_threadsafe({"type": "pretrain", "status": "done", "engine": engine, "result": res.to_dict(), "state": sess.app.snapshot()})
        except Exception as ex:  # noqa: BLE001
            log.exception("pretrain failed")
            sess.send_threadsafe({"type": "pretrain", "status": "error", "error": str(ex)})
        finally:
            G["pretrain_busy"] = False

    threading.Thread(target=work, daemon=True).start()
    return {"ok": True, "started": True}


def engines_list() -> list[dict]:
    """使えるエンジン (共有 ASR サーバーに載っているもの + ブラウザで動かせるもの)。"""
    if G.get("engines_cache") and time.time() - G["engines_cache"][0] < 30:
        return G["engines_cache"][1]
    avail = None
    if G["asr_url"]:
        try:
            from openvons.voice.remote import list_engines
            avail = {e["id"]: e for e in list_engines(G["asr_url"])}
        except Exception as e:  # noqa: BLE001
            log.warning("asr server unreachable: %s", e)
            avail = {}
    out = []
    for e in ENGINES.values():
        d = e.to_dict()
        d["server"] = (e.id in avail) if avail is not None else True
        d["browser_model"] = f"/kana/models/{e.file}" if e.browser and G.get("komimi_models") else None
        out.append(d)
    G["engines_cache"] = (time.time(), out)
    return out


@app.get("/api/kana/config")
def kana_config(session: str | None = None):
    sess = get_session(session) if session else None
    return {"engines": engines_list(), "engine": sess.engine if sess else G["default_engine"], "asr": "shared" if G["asr_url"] else "local",
            "vocab": "/kana/models/ja1024_vocab.json" if G.get("komimi_models") else None, "wasm": "/shared/wasm/ovkana.js",
            "worker": "/shared/kana_worker.js"}


@app.get("/api/kana/commands")
def kana_commands(session: str, engine: str | None = None):
    """ブラウザの振り分け (WebAssembly) に渡すコマンド集合。仮説ごとに [カナ, 意味 id (初出順), 危険度 0/1/2, フラグ]。
    フラグ: 1 埋め込み可、2 確認し直せる、4 実行側 (はい)。cs_key は判断を返すときに付ける (状態が変わっていたら stale)。"""
    sess = get_session(session)
    cs = sess.app.command_set()
    mid: dict[tuple, int] = {}
    rows = []
    for h in cs.hyps:
        m = mid.setdefault(h.meaning, len(mid))
        rows.append([h.kana, m, RISK_CODE.get(h.risk, 0), (1 if h.allow_embed else 0) | (2 if h.confirmable else 0) | (4 if h.positive else 0)])
    eng = engine or sess.engine
    cal, trained = engine_calibration(sess.app.scope, eng)
    th = Thresholds(); r = Recognizer(None)
    params = [cal.get("temperature", 2.5), cal.get("none_bias", 4.0), cal.get("len_bonus", 1.4), cal.get("residual_penalty", 1.0),
              th.execute, th.confirm, th.execute_medium, th.clear_min, th.clear_ratio, th.clear_none_max, th.answer_yes, th.answer_no,
              r.shortlist_k, 1.0 if r.embed else 0.0, r.embed_min_ratio, r.embed_max_residual, r.embed_min_morae, r.embed_penalty]
    return {"cs_key": cs_key(sess.app, cs), "state": cs.state, "scope": sess.app.scope.id, "engine": eng, "calibrated": trained,
            "calibration": cal, "params": params, "n_meanings": len(mid), "hyps": rows}


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
    await sess.send({"type": "hello", "session": sess.id, "state": sess.app.snapshot(), "scopes": list_scopes(),
                     "engine": sess.engine, "engines": await asyncio.to_thread(engines_list)})
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
                    sess.app.set_scope(sess.app.scope)
                    await sess.send({"type": "state", "state": sess.app.snapshot()})
                elif t == "click_camera":     # マウス操作も同じ状態機械を通す
                    from openvons.voice.engine import Candidate, Decision
                    from openvons.voice.grammar import Hypothesis
                    h = Hypothesis(APP.SELECT_INTENT, data["label"], "", {APP.SLOT: data["id"]})
                    d = Decision("execute", Candidate(h, 1.0, 0.0), [], 0.0, "(click)", 0.0, {}, sess.app.sm.state, "UI")
                    await sess.send(sess.app.apply(d))
                elif t == "intent":          # ボタン・キー操作。音声と同じ apply() を通す (状態が許す意図だけ)
                    from openvons.voice.engine import Candidate, Decision
                    from openvons.voice.grammar import Hypothesis
                    name = data.get("intent", "")
                    if name in sess.app.sm.allowed_intents() and not sess.app.grammar.intents[name].patterns[0].startswith("{"):
                        it = sess.app.grammar.intents[name]
                        h = Hypothesis(name, it.description or name, "", {}, dict(it.params), it.risk)
                        d = Decision("execute", Candidate(h, 1.0, 0.0), [], 0.0, "(ui)", 0.0, {}, sess.app.sm.state, "UI")
                        await sess.send(sess.app.apply(d))
                elif t == "engine":         # kana 入力エンジンの切り替え (サーバー側で認識するときに使う。校正もエンジンごと)
                    e = data.get("engine")
                    if e in ENGINES:
                        sess.engine = e
                    cal, trained = engine_calibration(sess.app.scope, sess.engine)
                    await sess.send({"type": "engine", "engine": sess.engine, "calibrated": trained, "calibration": cal})
                elif t == "decision":       # ブラウザ (WebAssembly) で振り分けた結果
                    ev = await asyncio.to_thread(sess.apply_client_decision, data)
                    if ev.get("type") == "result":
                        ev["req"] = data.get("req")
                    await sess.send(ev)
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
    global APP, STATE_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", default="examples.road_cameras.app", help="アプリモジュール")
    ap.add_argument("--port", type=int, default=8600)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--tts", default=os.environ.get("JEV_TTS_URL", "voicevox://127.0.0.1:50021"), help="voicevox:// | irodori:// | openai://")
    ap.add_argument("--data", default=None, help="実体 JSON (既定: アプリの DATA_FILE)")
    ap.add_argument("--hierarchy", default=None)
    ap.add_argument("--ssl-dir", default=None, help="cert.pem/key.pem のあるディレクトリ (マイクは https か localhost が必須)")
    ap.add_argument("--asr", default=os.environ.get("OPENVONS_ASR_URL", "local"),
                    help="共有 ASR サーバーの URL (例 http://127.0.0.1:8630)。local ならこのプロセスにエンジンを読む")
    ap.add_argument("--engine", default=DEFAULT_ENGINE, help="既定の kana 入力エンジン (" + " | ".join(ENGINES) + ")")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    APP = importlib.import_module(args.app)
    app_dir = Path(APP.__file__).resolve().parent
    G["static"] = app_dir / "static"
    stamps = [p.stat().st_mtime for d in (app_dir / "static", HERE / "demo_static", HERE / "demo_static" / "wasm") if d.exists() for p in d.glob("*") if p.is_file()]
    G["version"] = str(int(max(stamps) if stamps else time.time()))     # 共通部品 (kana.js・WASM) を直したときも取り直させる
    app.title = getattr(APP, "TITLE", app.title)
    if "JEV_STATE_DIR" not in os.environ:
        STATE_DIR = STATE_ROOT / app_dir.name
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    data_file = Path(args.data) if args.data else app_dir / APP.DATA_FILE
    hier_file = Path(args.hierarchy) if args.hierarchy else app_dir / "hierarchy.json"

    lex_path = STATE_DIR / "entities.json"
    if not lex_path.exists():
        import shutil
        shutil.copy(data_file, lex_path)
    G["lexicon_path"] = lex_path
    G["lexicon"] = Lexicon.load(lex_path)
    G["hierarchy"] = json.loads(hier_file.read_text(encoding="utf-8"))
    G["scopes"] = ScopeStore(STATE_DIR / "scopes.json")
    G["tts"] = TTSClient(args.tts, cache_dir=STATE_DIR / "tts_cache")
    log.info("tts backend: %s ok=%s", args.tts, G["tts"].ok())
    G["asr_url"] = None if args.asr in ("", "local") else args.asr.rstrip("/")
    G["default_engine"] = args.engine
    G["recognizers"] = {}
    from openvons.voice.komimi_asr import KOMIMI_HOME
    G["komimi_models"] = (KOMIMI_HOME / "models") if (KOMIMI_HOME / "models" / "ja1024_vocab.json").exists() else None
    log.info("kana engine: %s via %s", args.engine, G["asr_url"] or "local")
    rec = recognizer(args.engine)
    rec.asr.warmup()
    G["recognizer"] = rec        # 互換 (事前学習の既定など)
    log.info("ready: %d cameras, tts=%s", len(G["lexicon"]), G["tts"].ok())
    if hasattr(APP, "register_routes"):        # アプリ固有の API (河川版のライブ画像プロキシなど)
        APP.register_routes(app, G, STATE_DIR)
    app.mount("/static", StaticFiles(directory=str(G["static"])), name="static")
    app.mount("/shared", StaticFiles(directory=str(HERE / "demo_static")), name="shared")
    if G["komimi_models"]:
        app.mount("/kana/models", StaticFiles(directory=str(G["komimi_models"])), name="kana_models")
    import uvicorn
    kw = {}
    if args.ssl_dir:
        kw = {"ssl_certfile": f"{args.ssl_dir}/cert.pem", "ssl_keyfile": f"{args.ssl_dir}/key.pem"}
    uvicorn.run(app, host=args.host, port=args.port, log_level="info", **kw)


if __name__ == "__main__":
    main()
