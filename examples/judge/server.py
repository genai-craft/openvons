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

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.judge.checks import HEAD_KEYS, SCENE_OPTIONS, SCENE_QUESTION, BY_KEY, CHECKS, SCENES  # noqa: E402
from openvons.core.decision import Thresholds, decide  # noqa: E402
from openvons.vision.video_judge import VQuestion, VideoJudge  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("JUDGE_DATA", str(HERE.parents[1] / "state" / "judge")))
CLIPS = DATA / "clips"
JAF = DATA / "jaf"          # 社内確認用の実写 (© JAF、再配布しない): パスワード付きでだけ見せる


def _jaf_key() -> str:
    """JAF 動画の閲覧パスワード。環境変数 JUDGE_JAF_PASS か state/judge/jaf_pass.txt。無ければ非公開。"""
    k = os.environ.get("JUDGE_JAF_PASS", "")
    f = DATA / "jaf_pass.txt"
    if not k and f.exists():
        k = f.read_text().strip()
    return k


def _jaf_ok(key: str) -> bool:
    import hmac
    real = _jaf_key()
    return bool(real) and hmac.compare_digest(key or "", real)
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


def vqs(c) -> list[VQuestion]:
    """短い窓 (動作) + 必要なら長い窓 (前後の文脈、key に @long を付ける)。"""
    qs = [vq(c)]
    if c.long_window_s:
        q = VQuestion(c.key + "@long", c.question, ["yes", "no", "cannot tell"], 2, fps=c.long_fps, window_s=c.long_window_s, risk=c.risk, labels_ja=["はい", "いいえ", "判別できない"])
        qs.append(q)
    return qs


def scene_vq() -> VQuestion:
    return VQuestion("__scene__", SCENE_QUESTION, [d for _, d in SCENE_OPTIONS], len(SCENE_OPTIONS) - 1, fps=2.0, window_s=4.0, labels_ja=[n for n, _ in SCENE_OPTIONS])


def apply_scene_filter(res: dict, keys: list[str]) -> None:
    """シーン自動判定: 各区間の種別が項目のシーンと違う窓は判定対象外 (skipped、p=[0,0,1]) にする。"""
    scenes = res.get("scenes") or []
    for key in keys:
        q = res["questions"].get(key)
        if not q:
            continue
        want = BY_KEY[key].scene
        for w in q["series"]:
            si = w.get("scene")
            if si is None or si >= len(scenes) or "type_ja" not in scenes[si]:
                continue
            if scenes[si]["type_ja"] != want:
                w["skipped"] = True
                w["p"] = [0.0, 0.0, 1.0]


def merge_long(res: dict) -> None:
    """key@long の窓を key の窓列に合流させる (判定は両方の窓の最大値)。UI 用に元の系列も残す。"""
    for k in [k for k in res["questions"] if k.endswith("@long")]:
        base = k[: -len("@long")]
        if base in res["questions"]:
            res["questions"][base]["series"] = sorted(res["questions"][base]["series"] + [dict(w, long=True) for w in res["questions"][k]["series"]], key=lambda w: (w["t0"], w["t1"]))
        del res["questions"][k]


def summarize(res: dict) -> dict:
    """窓ごとの確率 → 項目ごとの要約 (最大確率の窓、判断、検出区間)。"""
    out = []
    for key, q in res["questions"].items():
        c = BY_KEY[key]
        ps = [s["p"] for s in q["series"] if not s.get("skipped")]
        if not ps:
            if q["series"]:
                out.append({"key": key, "title": c.title, "scene": c.scene, "risk": c.risk, "p_max": 0.0, "t_max": 0.0, "action": "skip", "label": "該当シーンなし",
                            "segments": [], "fps": q["fps"], "window_s": q["window_s"]})
            continue
        yes = [p[0] for p in ps]; no = [p[1] for p in ps]; none = [p[2] for p in ps]
        i = max(range(len(yes)), key=lambda k: yes[k])
        live = [s for s in q["series"] if not s.get("skipped")]
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
        for s, p in zip(live, ps):
            if p[0] >= TH.confirm:
                if segs and s["t0"] <= segs[-1][1] + 1e-6:
                    segs[-1][1] = s["t1"]
                else:
                    segs.append([s["t0"], s["t1"]])
        out.append({"key": key, "title": c.title, "scene": c.scene, "risk": c.risk, "p_max": yes[i], "t_max": live[i]["t0"],
                    "action": action, "label": label, "segments": segs, "fps": q["fps"], "window_s": q["window_s"]})
    order = {"execute": 0, "confirm": 1, "none": 2, "reject": 3, "skip": 4}
    out.sort(key=lambda r: (order[r["action"]], -r["p_max"]))
    return {"items": out}


def _load_heads():
    """学習 head (state/judge/head_<key>.pt) があれば読む。"""
    import torch
    from examples.judge.train_head import Head
    heads = {}
    for c in CHECKS:
        f = DATA / f"head_{c.key}.pt"
        if f.exists():
            ck = torch.load(f, map_location="cuda")
            ck.setdefault("ctx", 0)
            h = Head(ck["d_h"], ck["d_z"]).cuda().eval(); h.load_state_dict(ck["state"]); heads[c.key] = (h, ck.get("thr", 0.5), int(ck.get("ctx", 0)))
    return heads


def apply_heads(res: dict, keys: list[str]) -> None:
    """窓の確率を head の出力 [p, 1-p, 0] に置き換える (head がある項目のみ)。質問方式の 30 logit も入力に使う。
    head が ctx>0 で学習されていれば前後 ctx 窓の特徴も連結する (前後の状況)。同じシーン区間の中だけで取り、区間の端は自分で埋める = カットや PTZ でリセット。"""
    import numpy as np
    import torch
    heads = G.get("heads") or {}
    for key in keys:
        if key not in heads or key not in res["questions"]:
            continue
        h, thr, ctx = heads[key]
        q = res["questions"][key]
        wins = [w for w in q["series"] if "_hidden" in w and not w.get("long")]
        if not wins:
            continue
        def zvec(w_index: int) -> np.ndarray:
            z = np.zeros(len(HEAD_KEYS) * 3, np.float32)
            for k2, q2 in res["questions"].items():
                if k2 in HEAD_KEYS and w_index < len(q2["series"]) and (q2["fps"], q2["window_s"]) == (q["fps"], q["window_s"]):
                    z[HEAD_KEYS.index(k2) * 3 : HEAD_KEYS.index(k2) * 3 + 3] = q2["series"][w_index]["logit"]
            return z
        idx_of = {id(w): i for i, w in enumerate(q["series"])}
        H = [w["_hidden"] for w in wins]; Z = [zvec(idx_of[id(w)]) for w in wins]
        for i, w in enumerate(wins):
            nb = []
            for o in range(-ctx, ctx + 1):
                j = min(max(i + o, 0), len(wins) - 1)
                if wins[j].get("scene") != w.get("scene"):
                    j = i   # 別のシーン区間は見ない
                nb.append(j)
            hh = np.concatenate([H[j] for j in nb]); zz = np.concatenate([Z[j] for j in nb])
            with torch.no_grad():
                p = float(torch.sigmoid(h(torch.tensor(hh)[None].cuda(), torch.tensor(zz)[None].cuda()))[0])
            # head の閾値を 0.5 に写像して decide の閾値 (0.4 / 0.85) と揃える
            p = 0.5 * p / thr if p < thr else 0.5 + 0.5 * (p - thr) / max(1 - thr, 1e-6)
            if not w.get("skipped"):
                w["p"] = [p, 1 - p, 0.0]
        q["mode"] = "head"


@app.on_event("startup")
def _load():
    model = os.environ.get("JUDGE_MODEL", "Qwen/Qwen3-VL-4B-Instruct")
    G["judge"] = VideoJudge(model, calibration=str(DATA / "calibration.json"))
    G["heads"] = _load_heads()
    G["judge"].want_hidden = bool(G["heads"])
    print("heads loaded:", sorted(G["heads"]), flush=True)
    G["model"] = model
    static = HERE / "static"
    V["v"] = str(int(max(p.stat().st_mtime for p in static.glob("*"))))


@app.get("/")
def index():
    return HTMLResponse((HERE / "static" / "index.html").read_text(encoding="utf-8").replace("__V__", V["v"]), headers={"Cache-Control": "no-store"})


@app.get("/healthz")
def healthz():
    return {"ok": True, "model": G.get("model"), "heads": sorted(G.get("heads") or {})}


@app.get("/api/checks")
def checks():
    return {"scenes": SCENES, "auto_scene": True, "checks": [{"key": c.key, "title": c.title, "scene": c.scene, "fps": c.fps, "window_s": c.window_s, "risk": c.risk, "question": c.question} for c in CHECKS]}


@app.get("/api/jaf/samples")
def jaf_samples(key: str = ""):
    """パスワード付き: JAF 危険予知トレーニング動画の一覧 (手元にある分だけ)。"""
    if not _jaf_ok(key):
        return JSONResponse({"error": "パスワードが違います"}, status_code=401)
    lst = JAF / "list.json"
    rows = json.load(open(lst)) if lst.exists() else []
    out = []
    for r in rows:
        f = f'{r["cat"]}_{r["scene"]}.mp4'
        if (JAF / f).exists():
            out.append({"file": f, "title": r["title"].replace("（危険予知・事故回避トレーニング）", ""), "page": r["page"], "id": f[:-4]})
    return {"samples": out, "note": "© JAF。社内確認用。再配布しないでください。"}


@app.get("/jaf/{name}")
def jaf_file(name: str, request: Request, key: str = ""):
    if not _jaf_ok(key):
        return JSONResponse({"error": "パスワードが違います"}, status_code=401)
    path = JAF / Path(name).name
    if not path.exists() or path.suffix not in (".mp4", ".jpg"):
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(str(path), media_type="video/mp4" if path.suffix == ".mp4" else "image/jpeg")


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
    return r and {"label": r["label"], "check": r["check"], "event_start": r.get("event_start"), "event_end": r.get("event_end"), "suspect": bool(r.get("suspect"))} or {}


@app.post("/api/judge")
async def judge(file: UploadFile | None = File(None), sample: str = Form(""), scene: str = Form(""), checks: str = Form(""), state: str = Form(""), mode: str = Form("zeroshot"),
                split_scenes: str = Form("1"), key: str = Form("")):
    """scene: "" = 全項目、"auto" = カットで区切った区間ごとに場面の種類を判定し、その場面の項目だけ判定する、それ以外 = その場面の項目。"""
    t0 = time.time()
    auto = scene == "auto"
    keys = [k for k in checks.split(",") if k] or [c.key for c in CHECKS if auto or not scene or c.scene == scene]
    qs = [q for k in keys if k in BY_KEY for q in vqs(BY_KEY[k])]
    tmp = None
    if sample.startswith("jaf:"):
        if not _jaf_ok(key):
            return JSONResponse({"error": "パスワードが違います"}, status_code=401)
        path = JAF / Path(sample[4:]).name
        if not path.exists():
            return JSONResponse({"error": "sample not found"}, status_code=404)
    elif sample:
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
        res = G["judge"].judge(str(path), qs, state=st, max_s=90.0, scene_question=scene_vq() if auto else None, split_scenes=split_scenes not in ("0", "false"))
    finally:
        if tmp:
            os.unlink(tmp.name)
    if auto:
        apply_scene_filter(res, keys)
    if mode == "head":
        apply_heads(res, keys)
    merge_long(res)
    for v in res["questions"].values():
        for w in v["series"]:
            w.pop("_hidden", None)
    summ = summarize(res)
    return {"duration": res["duration"], "timing": res["timing"], "summary": summ["items"], "mode": mode, "heads": sorted(G.get("heads") or {}), "scenes": res.get("scenes", []), "auto_scene": auto,
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
    CLIPS.mkdir(parents=True, exist_ok=True)
    app.mount("/clips", StaticFiles(directory=str(CLIPS.resolve()), follow_symlink=True), name="clips")   # 起動前に mount。state/judge は symlink なので follow_symlink
    import uvicorn
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
