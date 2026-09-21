"""河川ライブカメラ画像の定期保存 (台風などの記録用)。

cameras.json の全カメラ (関東地整、195 台、10 分更新) の最新画像を取得し、
  <ARCHIVE>/<camera id>/<YYYYMMDD>/<HHMM>.jpg
に保存する。前回と同じ画像 (sha256 が同じ) は保存せず、index.jsonl に "same" と記録するだけ。
1 回の実行が 1 スナップショット。cron で 10 分おきに回す:

  */10 * * * * cd ~/dev/openvons && .venv/bin/python examples/kasen/archive.py --until 2026-09-28 >> /data/openvons/kasen_archive/cron.log 2>&1

環境変数 KASEN_ARCHIVE で置き場を変えられる (既定 /data/openvons/kasen_archive)。
出典: 関東地方整備局ウェブサイト (https://www.ktr.mlit.go.jp/) の画像。公共データ利用規約 (PDL1.0)。記録用途で、公開・再配布するときは出典表記が要る。"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
ARCHIVE = Path(os.environ.get("KASEN_ARCHIVE", "/data/openvons/kasen_archive"))
UA = "Mozilla/5.0 (openvons kasen archive; +https://github.com/genai-craft/openvons)"


def fetch(client: httpx.Client, cam: dict, stamp: str, day: str) -> dict:
    a = cam["attrs"]; url = a["image_url"]
    rec = {"t": stamp, "id": cam["id"], "label": cam["label"], "river": a.get("river"), "url": url}
    try:
        r = client.get(url, headers={"Referer": a.get("page_url") or url})
        rec["status"] = r.status_code
        if r.status_code != 200 or not r.content or "image" not in r.headers.get("content-type", ""):
            rec["result"] = "error"
            return rec
        sha = hashlib.sha256(r.content).hexdigest()
        d = ARCHIVE / cam["id"]; d.mkdir(parents=True, exist_ok=True)
        last = d / "last.sha"
        if last.exists() and last.read_text().strip() == sha:
            rec["result"] = "same"; rec["sha"] = sha
            return rec
        out = d / day / f"{stamp[9:13]}.jpg"; out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(r.content); last.write_text(sha)
        rec.update({"result": "saved", "bytes": len(r.content), "sha": sha, "file": str(out.relative_to(ARCHIVE)),
                    "last_modified": r.headers.get("last-modified")})
    except Exception as e:  # noqa: BLE001
        rec["result"] = "error"; rec["error"] = str(e)[:200]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--until", default="", help="この日付 (YYYY-MM-DD、含む) を過ぎたら何もしない (cron を残しても安全)")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    now = dt.datetime.now()
    if a.until and now.date() > dt.date.fromisoformat(a.until):
        return
    stamp = now.strftime("%Y%m%d_%H%M"); day = stamp[:8]
    cams = json.load(open(HERE / "cameras.json"))
    cams = [c for c in cams if c.get("attrs", {}).get("image_url")]
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=20.0, headers={"User-Agent": UA}, follow_redirects=True) as client, cf.ThreadPoolExecutor(a.workers) as ex:
        recs = list(ex.map(lambda c: fetch(client, c, stamp, day), cams))
    with open(ARCHIVE / "index.jsonl", "a") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n = {k: sum(1 for r in recs if r["result"] == k) for k in ("saved", "same", "error")}
    print(f"{stamp} cameras={len(cams)} saved={n['saved']} same={n['same']} error={n['error']} bytes={sum(r.get('bytes', 0) for r in recs)/1e6:.1f}MB", flush=True)
    if n["error"]:
        errs = [r for r in recs if r["result"] == "error"]
        print("  errors:", [(r["id"], r.get("status"), r.get("error", "")[:60]) for r in errs[:5]], flush=True)


if __name__ == "__main__":
    sys.exit(main())
