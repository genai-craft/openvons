"""openvons のランディングページ (openvons.com)。静的ファイルを配るだけ。

    .venv/bin/python -m examples.landing.server --port 8605
"""
from __future__ import annotations

import argparse
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

HERE = Path(__file__).resolve().parent
app = FastAPI(title="openvons")
V = {"v": "0"}


@app.get("/")
def index():
    html = (HERE / "static" / "index.html").read_text(encoding="utf-8").replace("__V__", V["v"])
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})


@app.get("/jevpick")
def jevpick():
    html = (HERE / "static" / "jevpick.html").read_text(encoding="utf-8").replace("__V__", V["v"])
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})


@app.get("/healthz")
def healthz():
    return {"ok": True}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8605)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    static = HERE / "static"
    V["v"] = str(int(max(p.stat().st_mtime for p in static.glob("*"))))
    app.mount("/static", StaticFiles(directory=str(static)), name="static")
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
