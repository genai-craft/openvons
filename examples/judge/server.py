"""judge デモサーバー (judge.openvons.com): 動画の審判とライブ判定。

    CUDA_VISIBLE_DEVICES=4 .venv/bin/python -m examples.judge.server --port 8607

- POST /api/judge (multipart video | sample=<name>): 10 項目 (シーンで絞れる) を窓ごとに確率で返す
- POST /api/live: 1〜4 フレーム (JPEG の base64) と質問セット (checks | nav) → 確率と遅延
- GET  /api/samples: 生成した試験動画の一覧 (正解は伏せて返し、判定後に開く)
動画・フレームはサーバーに保存しない (試験動画は除く)。
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from typing import Any

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.judge.checks import BY_KEY, CHECKS, SCENES  # noqa: E402
from openvons.core.decision import Thresholds, decide  # noqa: E402
from openvons.vision.video_judge import VQuestion, VideoJudge  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("JUDGE_DATA", str(HERE.parents[1] / "state" / "judge")))
CLIPS = DATA / "clips"
app = FastAPI(title="openvons judge")
G: dict[str, Any] = {}
V = {"v": "0"}
TH = Thresholds()
NAV = VQuestion("nav", "The camera is mounted on the front of a small mobile robot driving indoors. Which way should the robot go next?",
                ["turn left", "turn right", "go straight", "stop"], None, labels_ja=["左へ", "右へ", "直進", "停止"])
NAV_OBST = VQuestion("obstacle", "Is there an obstacle or a person within about one meter directly ahead of the robot?",
                     ["yes", "no", "cannot tell"], 2, labels_ja=["はい", "いいえ", "判別できない"])
NAV_FLOOR = VQuestion("floor", "What is directly ahead on the floor?", ["clear floor", "stairs or a drop", "an object", "a person or animal"], None,
                      labels_ja=["何もない床", "階段・段差", "物", "人・動物"])


def vq(c) -> VQuestion:
    return VQuestion(c.key, c.question, ["yes", "no", "cannot tell"], 2, fps=c.fps, window_s=c.window_s, risk=c.risk, labels_ja=["はい", "いいえ", "判別できない"])


def summarize(res: dict) -> dict:
    """窓ごとの確率 → 項目ごとの要約 (最大確率の窓、判断、検出区間)。"""
    out = []
    for key, q in res["questions"].items():
        c = BY_KEY[key]
        ps = [s["p"] for s in q["series"]]
        if not ps:
            continue
        yes = [p[0] for p in ps]; no = [p[1] for p in ps]; none = [p[2] for p in ps]
        i = max(range(len(yes)), key=lambda k: yes[k])
        if yes[i] >= TH.confirm:
            # 「はい」が疑われる窓がある → その窓の分布で判断 (危険度 high は必ず要確認)
            action, reason = decide(yes[i], none[i], c.risk, TH)
            if action == "none":
                action = "confirm"  # 疑いはあるが判別できない側も大きい → 人が見る
            label = {"execute": "検出", "confirm": "要確認", "reject": "問題なし", "none": "判別できない"}[action]
        else:
            # どの窓も「はい」が低い → 「いいえ」が確かなら問題なし、映っていない/見えないなら判別できない
            action = "reject" if float(np.median(no)) >= 0.6 else "none"
            label = "問題なし" if action == "reject" else "判別できない"
        segs = []
        for s, p in zip(q["series"], ps):
            if p[0] >= TH.confirm:
                if segs and s["t0"] <= segs[-1][1] + 1e-6:
                    segs[-1][1] = s["t1"]
                else:
                    segs.append([s["t0"], s["t1"]])
        out.append({"key": key, "title": c.title, "scene": c.scene, "risk": c.risk, "p_max": yes[i], "t_max": q["series"][i]["t0"],
                    "action": action, "label": label, "segments": segs, "fps": q["fps"], "window_s": q["window_s"]})
    order = {"execute": 0, "confirm": 1, "none": 2, "reject": 3}
    out.sort(key=lambda r: (order[r["action"]], -r["p_max"]))
    return {"items": out}


@app.on_event("startup")
def _load():
    model = os.environ.get("JUDGE_MODEL", "Qwen/Qwen3-VL-4B-Instruct")
    G["judge"] = VideoJudge(model, calibration=str(DATA / "calibration.json"))
    G["model"] = model
    static = HERE / "static"
    V["v"] = str(int(max(p.stat().st_mtime for p in static.glob("*"))))
    if CLIPS.exists():
        app.mount("/clips", StaticFiles(directory=str(CLIPS)), name="clips")


@app.get("/")
def index():
    return HTMLResponse((HERE / "static" / "index.html").read_text(encoding="utf-8").replace("__V__", V["v"]), headers={"Cache-Control": "no-store"})


@app.get("/healthz")
def healthz():
    return {"ok": True, "model": G.get("model")}


@app.get("/api/checks")
def checks():
    return {"scenes": SCENES, "checks": [{"key": c.key, "title": c.title, "scene": c.scene, "fps": c.fps, "window_s": c.window_s, "risk": c.risk, "question": c.question} for c in CHECKS]}


@app.get("/api/samples")
def samples():
    man = CLIPS / "manifest.json"
    if not man.exists():
        return {"samples": []}
    rows = json.load(open(man))
    # 正解は伏せる (判定後に /api/samples/answer で開く)
    return {"samples": [{"file": r["file"], "check": r["check"], "title": r["title"], "scene": r["scene"], "id": r["file"].rsplit(".", 1)[0]} for r in rows]}


@app.get("/api/samples/answer")
def sample_answer(file: str):
    man = CLIPS / "manifest.json"
    rows = json.load(open(man)) if man.exists() else []
    r = next((x for x in rows if x["file"] == file), None)
    return r and {"label": r["label"], "check": r["check"], "event_start": r.get("event_start"), "event_end": r.get("event_end")} or {}


@app.post("/api/judge")
async def judge(file: UploadFile | None = File(None), sample: str = Form(""), scene: str = Form(""), checks: str = Form(""), state: str = Form("")):
    t0 = time.time()
    keys = [k for k in checks.split(",") if k] or [c.key for c in CHECKS if not scene or c.scene == scene]
    qs = [vq(BY_KEY[k]) for k in keys if k in BY_KEY]
    tmp = None
    if sample:
        path = CLIPS / Path(sample).name
        if not path.exists():
            return JSONResponse({"error": "sample not found"}, status_code=404)
    else:
        if file is None:
            return JSONResponse({"error": "no video"}, status_code=400)
        data = await file.read()
        if len(data) > 200 * 1024 * 1024:
            return JSONResponse({"error": "動画は 200MB まで"}, status_code=413)
        tmp = tempfile.NamedTemporaryFile(suffix=Path(file.filename or "v.mp4").suffix or ".mp4", delete=False)
        tmp.write(data); tmp.close(); path = Path(tmp.name)
    try:
        st = state or "Fixed camera footage."
        res = G["judge"].judge(str(path), qs, state=st, max_s=90.0)
    finally:
        if tmp:
            os.unlink(tmp.name)
    summ = summarize(res)
    return {"duration": res["duration"], "timing": res["timing"], "summary": summ["items"],
            "series": {k: {"windows": v["series"], "fps": v["fps"], "window_s": v["window_s"]} for k, v in res["questions"].items()},
            "model": G["model"], "elapsed_s": time.time() - t0}


class LiveReq(BaseModel):
    frames: list[str]           # JPEG/PNG の data URL か base64
    mode: str = "checks"        # checks | nav
    checks: list[str] = []      # mode=checks のとき (空なら全部)
    state: str = ""


@app.post("/api/live")
def live(req: LiveReq):
    frames = []
    for s in req.frames[-4:]:
        b = s.split(",", 1)[1] if s.startswith("data:") else s
        im = Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB")
        if im.width > 512:
            im = im.resize((512, int(im.height * 512 / im.width)))
        frames.append(im)
    if not frames:
        return JSONResponse({"error": "no frames"}, status_code=400)
    if req.mode == "nav":
        qs = [NAV, NAV_OBST, NAV_FLOOR]
        st = req.state or "Front camera of a small indoor mobile robot, about 30 cm above the floor."
    else:
        keys = req.checks or [c.key for c in CHECKS]
        qs = [vq(BY_KEY[k]) for k in keys if k in BY_KEY]
        st = req.state or "Live camera."
    r = G["judge"].live(frames, qs, st)
    items = []
    for q in qs:
        p = r["probs"][q.key]
        top = max(range(len(p)), key=lambda i: p[i])
        none_p = p[q.none_index] if q.none_index is not None else 0.0
        action, _ = decide(p[top], none_p, q.risk, TH)
        items.append({"key": q.key, "title": BY_KEY[q.key].title if q.key in BY_KEY else q.text, "options": q.labels_ja or q.options, "p": p, "top": top, "action": action})
    return {"items": items, "timing": r["timing"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8607)
    ap.add_argument("--host", default="0.0.0.0")
    a = ap.parse_args()
    app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")
    import uvicorn
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
