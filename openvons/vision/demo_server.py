"""顔・全身の属性デモサーバー (openvons.vision): Web カメラの 1 フレームを送ると、
顔ごとに年齢帯 (9 区分)・性別、全身 (フレーム全体 or 指定した矩形) の性別・年代・向き・荷物を確率付きで返す。

    CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m openvons.vision.demo_server --port 8602 \
        --face /data/decision_model/checkpoints/vis_fairface_2b_meanmax --body /data/decision_model/checkpoints/vis_pa100k_2b_balanced

どちらも凍結した Qwen3-VL-2B の視覚エンコーダ (407M) + 数万パラメータの head。生成はせず、選択肢に確率で答える。
確率は openvons.core.decide の 3 段階 (確定 / 要確認 / 不明) で見せる。画像はサーバーに保存しない。
"""
from __future__ import annotations

import argparse
import base64
import io
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from openvons.core.decision import Thresholds, decide

log = logging.getLogger("openvons.vision.demo")
app = FastAPI(title="openvons (open-Jev) vision demo")
G: dict[str, Any] = {}
HERE = Path(__file__).resolve().parent
LABELS_JA = {
    "age": {"0-2": "0〜2 歳", "3-9": "3〜9 歳", "10-19": "10 代", "20-29": "20 代", "30-39": "30 代", "40-49": "40 代", "50-59": "50 代", "60-69": "60 代", "more than 70": "70 歳以上",
            "under18": "18 歳未満", "adult": "成人", "over60": "60 歳以上"},
    "gender": {"Male": "男性", "Female": "女性", "male": "男性", "female": "女性"},
    "orientation": {"front": "正面", "side": "横", "back": "背面"},
    "carrying": {"no": "荷物なし", "yes": "荷物あり"},
}
TITLE_JA = {"age": "年齢", "gender": "性別", "orientation": "向き", "carrying": "荷物"}


def _detect_faces(img: Image.Image, max_faces: int = 4) -> list[tuple[int, int, int, int]]:
    """YuNet (OpenCV zoo、Apache-2.0、230KB) で顔を検出。"""
    det = G.get("detector")
    if det is None:
        return []
    import cv2
    arr = np.array(img)[:, :, ::-1].copy()          # RGB -> BGR
    scale = 1.0
    if max(arr.shape[:2]) > 640:                    # YuNet は 320〜640px が適正。大きい画像は縮めて検出し座標を戻す
        scale = 640 / max(arr.shape[:2])
        arr = cv2.resize(arr, (int(arr.shape[1] * scale), int(arr.shape[0] * scale)))
    det.setInputSize((arr.shape[1], arr.shape[0]))
    _, faces = det.detect(arr)
    if faces is None:
        return []
    boxes = [(int(f[0] / scale), int(f[1] / scale), int(f[2] / scale), int(f[3] / scale)) for f in faces if f[2] / scale > 24 and f[3] / scale > 24]
    return sorted(boxes, key=lambda b: -b[2] * b[3])[:max_faces]


def _crop(img: Image.Image, box: tuple[int, int, int, int], margin: float = 0.25) -> Image.Image:
    x, y, w, h = box
    m = int(max(w, h) * margin)
    return img.crop((max(0, x - m), max(0, y - m), min(img.width, x + w + m), min(img.height, y + h + m)))


AGE_ORDER = {"age": ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "more than 70"]}
AGE_LO_HI = {"0-2": (0, 2), "3-9": (3, 9), "10-19": (10, 19), "20-29": (20, 29), "30-39": (30, 39), "40-49": (40, 49), "50-59": (50, 59), "60-69": (60, 69), "more than 70": (70, 99)}


def _age_range(ids: list[str], p: list[float], target: float = 0.7) -> dict[str, Any] | None:
    """順序尺度の年齢: 最尤区分から隣へ広げ、累積確率が target を超えた幅を見出しにする (「30〜49 歳 82%」)。
    9 区分の argmax だけを見せると隣の区分に割れて「ばらける」ように見えるため。"""
    if ids != AGE_ORDER["age"]:
        return None
    best = int(np.argmax(p)); lo = hi = best; mass = p[best]
    while mass < target and (lo > 0 or hi < len(p) - 1):
        left = p[lo - 1] if lo > 0 else -1; right = p[hi + 1] if hi < len(p) - 1 else -1
        if right > left: hi += 1; mass += p[hi]
        else: lo -= 1; mass += p[lo]
    a, b = AGE_LO_HI[ids[lo]][0], AGE_LO_HI[ids[hi]][1]
    label = f"{a} 歳以上" if b >= 99 else (f"{a}〜{b} 歳" if lo != hi else LABELS_JA["age"][ids[lo]])
    expected = sum(pi * (AGE_LO_HI[i][0] + min(AGE_LO_HI[i][1], 80)) / 2 for i, pi in zip(ids, p))
    return {"label": label, "mass": round(float(mass), 4), "expected": round(float(expected))}


def _answers(model, questions: dict, imgs: list[Image.Image]) -> list[dict[str, Any]]:
    out = model.decide(imgs)
    res: list[dict[str, Any]] = [dict() for _ in imgs]
    th = G["thresholds"]
    for key, probs in out.items():
        ids = [c["id"] for c in questions[key]["choices"]]
        for i, p in enumerate(probs.tolist()):
            p = p[: len(ids)]
            best = int(np.argmax(p))
            rng = _age_range(ids, p)
            conf = float(rng["mass"]) if rng else float(p[best])       # 年齢は幅の確率で判定する
            action, _ = decide(conf, 0.0, "low", th)
            res[i][key] = {"title": TITLE_JA.get(key, key), "choice": ids[best], "choice_ja": (rng["label"] if rng else LABELS_JA.get(key, {}).get(ids[best], ids[best])),
                           "confidence": round(conf, 4), "level": {"execute": "確定", "confirm": "要確認", "reject": "不明"}.get(action, action),
                           "range": rng, "probabilities": {i_: {"label": LABELS_JA.get(key, {}).get(i_, i_), "p": round(float(v), 4)} for i_, v in zip(ids, p)}}
    return res


@app.get("/")
def index():
    html = (HERE.parent.parent / "examples" / "vision_attrs" / "static" / "index.html").read_text(encoding="utf-8").replace("__V__", G.get("version", "0"))
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})


@app.get("/api/info")
def info():
    return {"face": G.get("face_info"), "body": G.get("body_info"), "thresholds": G["thresholds"].__dict__}


@app.post("/api/analyze")
async def analyze(body: dict):
    """{"image_base64": ..., "body_box": [x, y, w, h] | null, "want": ["face", "body"]}"""
    t0 = time.perf_counter()
    try:
        img = Image.open(io.BytesIO(base64.b64decode(body["image_base64"].split(",")[-1]))).convert("RGB")
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": f"cannot decode image: {e}"}, 400)
    want = set(body.get("want") or ["face", "body"])
    out: dict[str, Any] = {"width": img.width, "height": img.height, "faces": [], "body": None}
    if "face" in want and "face" in G:
        boxes = _detect_faces(img)
        if not boxes and max(img.width, img.height) <= 400 and 0.7 <= img.width / img.height <= 1.4:
            boxes = [(0, 0, img.width, img.height)]        # 小さな正方形画像 = 顔のクロップそのもの (FairFace 等)
        t1 = time.perf_counter()
        if boxes:
            crops = [_crop(img, b) for b in boxes]
            ans = _answers(G["face"], G["face_q"], crops)
            out["faces"] = [{"box": list(b), **a} for b, a in zip(boxes, ans)]
        out["face_ms"] = round((time.perf_counter() - t1) * 1000, 1)
        out["detect_ms"] = round((t1 - t0) * 1000, 1)
    if "body" in want and "body" in G:
        t1 = time.perf_counter()
        bb = body.get("body_box")
        crop = img.crop((bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3])) if bb else img
        out["body"] = {"box": bb or [0, 0, img.width, img.height], **_answers(G["body"], G["body_q"], [crop])[0]}
        # 顔がフレームの高さの 12% 超 = 上半身のポートレート。歩行者用の全身モデルの前提が外れるので「参考」に落とす
        faces = out.get("faces") or []
        if not bb and faces and max(f["box"][3] for f in faces) / img.height > 0.12:
            out["body"]["reference_only"] = True
            for k, v in out["body"].items():
                if isinstance(v, dict) and "level" in v:
                    v["level"] = "参考"
        out["body_ms"] = round((time.perf_counter() - t1) * 1000, 1)
    out["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8602)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--face", default=os.environ.get("OPENVONS_FACE_CKPT"))
    ap.add_argument("--body", default=os.environ.get("OPENVONS_BODY_CKPT"))
    ap.add_argument("--detector", default=os.environ.get("OPENVONS_FACE_DETECTOR", str(Path(os.environ.get("OPENVONS_STATE", HERE.parent.parent / "state")) / "models" / "face_detection_yunet_2023mar.onnx")),
                    help="YuNet の onnx (https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    import cv2
    from openvons.vision.vision_model import VisionDecisionModel
    det_path = Path(args.detector)
    if not det_path.exists():
        url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
        try:
            import httpx
            det_path.parent.mkdir(parents=True, exist_ok=True)
            det_path.write_bytes(httpx.get(url, follow_redirects=True, timeout=60).content)
            log.info("downloaded YuNet -> %s", det_path)
        except Exception as e:  # noqa: BLE001
            log.warning("face detector not found and download failed: %s (%s) — 顔検出なしで起動", det_path, e)
    if det_path.exists():
        G["detector"] = cv2.FaceDetectorYN.create(str(det_path), "", (320, 320), 0.5, 0.3, 50)
        log.info("face detector: YuNet %s", det_path)
    G["thresholds"] = Thresholds(execute=0.7, confirm=0.45)
    for name in ("face", "body"):
        ck = getattr(args, name)
        if ck:
            m = VisionDecisionModel.from_checkpoint(ck)
            G[name] = m; G[name + "_q"] = m.cfg.questions
            G[name + "_info"] = {"ckpt": Path(ck).name, "frozen_M": round(m.n_frozen() / 1e6, 1), "trainable_K": round(m.n_trainable() / 1e3, 1), "questions": list(m.cfg.questions)}
            log.info("%s model: %s", name, G[name + "_info"])
    static = HERE.parent.parent / "examples" / "vision_attrs" / "static"
    G["version"] = str(int(max(p.stat().st_mtime for p in static.glob("*"))))
    app.mount("/static", StaticFiles(directory=str(static)), name="static")
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
