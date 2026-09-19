"""judge デモ用のテスト動画 (15 秒) を MiniMax H3 (ComfyUI クラスタ) で作る。

  ssh -D 1080 -N -f <踏み台>            # クラスタは VPN 内。SOCKS を開いておく (JUDGE_SOCKS)
  .venv/bin/python examples/judge/gen_clips.py --list
  .venv/bin/python examples/judge/gen_clips.py            # 60 本を並列投入 (サーバー台数まで同時)

各項目 × (問題あり 3 + 問題なし 3)。問題あり = [0-5s] 通常 → [5-10s] 事象 → [10-15s] 通常、問題なし = 通常 3 ショット。
manifest.json に正解ラベルと事象の時刻を書く。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.judge.checks import CHECKS, VARIANTS  # noqa: E402
from examples.judge.h3_client import Cluster, build_t2v  # noqa: E402

OUT = Path(os.environ.get("JUDGE_DATA", str(Path(__file__).resolve().parents[2] / "state" / "judge")))
CLIPS = OUT / "clips"
TAIL = ("\n\nCamera: one continuous fixed camera position for the whole video, no cuts, no zoom, the same location and the same people throughout. "
        "Audio: ambient sound only. No text, subtitles, logos or watermarks of any kind, no animation or cartoon rendering, realistic live-action look.")


def transcode(mp4: Path):
    """ComfyUI の出力をブラウザで確実に再生できる形 (H.264 High, yuv420p, CFR 24fps, AAC 48k, faststart) に再エンコードする。"""
    import subprocess
    tmp = mp4.with_name("tmp_" + mp4.name)
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4), "-map", "0:v:0", "-map", "0:a?", "-c:v", "libx264", "-profile:v", "high", "-level", "4.0",
                        "-pix_fmt", "yuv420p", "-r", "24", "-vsync", "cfr", "-preset", "fast", "-crf", "20", "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2",
                        "-movflags", "+faststart", str(tmp)])
    if r.returncode == 0:
        tmp.replace(mp4)


def poster(mp4: Path):
    """サムネイル用に 7 秒のフレームを JPEG で置く (ブラウザの <video> は preload だけだと真っ黒になる)。"""
    import subprocess
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", "7", "-i", str(mp4), "-frames:v", "1", "-vf", "scale=432:-1", str(mp4.with_suffix(".jpg"))], check=False)


def storyboard(shots: list[str], style: str) -> str:
    lines = [f"{style}\n\nScene overview: {shots[0]}\n\nStoryboard (same camera, continuous):"]
    for i, s in enumerate(shots):
        lines.append(f"[{i*5}s-{(i+1)*5}s] Shot {i+1}: {s}")
    return "\n".join(lines) + TAIL


PER = int(os.environ.get("JUDGE_PER", "10"))  # 項目ごとの 問題あり / なし の本数


def jobs():
    """i < 3 は元の 3 本 (修飾なし)、以降は 事象/通常 の組と修飾を回して増やす。"""
    js = []
    for c in CHECKS:
        n, e = c.normal_prompts, c.event_prompts
        for i in range(PER):
            v = VARIANTS[(i // 3) % len(VARIANTS)] if i >= 3 else ""
            style = c.style + v
            js.append({"name": f"{c.key}_problem{i}", "check": c.key, "label": 1, "event_start": 5.0, "event_end": 10.0,
                       "prompt": storyboard([n[i % 3], e[i % 3], n[(i + 1) % 3]], style), "title": c.title, "scene": c.scene, "variant": i // 3})
            js.append({"name": f"{c.key}_ok{i}", "check": c.key, "label": 0, "event_start": None, "event_end": None,
                       "prompt": storyboard([n[i % 3], n[(i + 1) % 3], n[(i + 2) % 3]], style), "title": c.title, "scene": c.scene, "variant": i // 3})
    return js


def run(args):
    CLIPS.mkdir(parents=True, exist_ok=True)
    cl = Cluster()
    alive = cl.alive()
    print("alive servers:", [(u.split('//')[1].split(':')[0], n) for n, u in alive], flush=True)
    if not alive:
        sys.exit("no ComfyUI server reachable (SOCKS?)")
    todo = [j for j in jobs() if not (CLIPS / f"{j['name']}.mp4").exists()]
    if args.only:
        todo = [j for j in todo if j["name"] in args.only.split(",")]
    print(len(todo), "clips to generate", flush=True)
    lock = threading.Lock()
    queue = list(todo)
    results = []

    def worker(url):
        fast = cl.has_vae(url, "minimax_h3_video_vae_int8_convrot.safetensors")
        while True:
            with lock:
                if not queue:
                    return
                j = queue.pop(0)
            t0 = time.time()
            try:
                g = build_t2v(j["prompt"], seconds=args.seconds, aspect="16:9", megapixels=args.mp, seed=abs(hash(j["name"])) % 2**52,
                              job=j["name"], steps=args.steps, fast_vae=fast)
                pid = cl.submit(url, g)
                files = cl.wait(url, pid, timeout=args.timeout)
                f = next((x for x in files if x["filename"].lower().endswith((".mp4", ".webm", ".mkv"))), files[0])
                cl.download(url, f, CLIPS / f"{j['name']}.mp4")
                transcode(CLIPS / f"{j['name']}.mp4")
                poster(CLIPS / f"{j['name']}.mp4")
                print(f"{j['name']}  {url.split('//')[1].split(':')[0]}  {time.time()-t0:.0f}s", flush=True)
                with lock:
                    results.append(j)
            except Exception as e:  # noqa: BLE001
                print(f"FAILED {j['name']} on {url}: {str(e)[:200]}", flush=True)
                with lock:
                    queue.append(j)  # 別サーバーで再試行
                time.sleep(10)

    threads = [threading.Thread(target=worker, args=(u,), daemon=True) for _, u in alive[: args.parallel]]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    write_manifest()


def write_manifest():
    man = [{k: v for k, v in j.items() if k != "prompt"} | {"file": f"{j['name']}.mp4", "prompt": j["prompt"]} for j in jobs() if (CLIPS / f"{j['name']}.mp4").exists()]
    json.dump(man, open(CLIPS / "manifest.json", "w"), ensure_ascii=False, indent=1)
    print(len(man), "clips in manifest ->", CLIPS / "manifest.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--seconds", type=float, default=15)
    ap.add_argument("--mp", type=float, default=0.4)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--parallel", type=int, default=13)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--manifest", action="store_true")
    a = ap.parse_args()
    if a.list:
        for j in jobs():
            print(j["name"]); print("   ", j["prompt"][:160].replace("\n", " "))
    elif a.manifest:
        write_manifest()
    else:
        run(a)
