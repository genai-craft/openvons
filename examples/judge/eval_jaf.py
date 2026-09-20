"""JAF 危険予知トレーニング動画 (© JAF、手元評価のみ、再配布しない) でヒヤリハット判定を確かめる。

動画の構成は決まっている: タイトル → 走行/歩行の映像 → 「Thinking Time」の静止画 → 事象 (飛び出し・急接近) → もう一度 → 別視点 …
なので正解区間は「静止 (Thinking Time) が終わってから次のカットまで」とし、
  - 事象区間の p(はい) の最大 と、その前の平常走行区間の p(はい) の最大 を比べる (事象の方が高ければ正解)
  - シーン自動判定 (道路 / その他) がタイトル画面を外せているか
を出す。使い方: JUDGE_GPU=7 .venv/bin/python examples/judge/eval_jaf.py [--mode head]"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from examples.judge.checks import BY_KEY  # noqa: E402
from examples.judge.server import DATA, apply_scene_filter, merge_long, scene_vq, sub_vqs, summarize, vqs  # noqa: E402
from openvons.vision.video_judge import VideoJudge, read_frames  # noqa: E402

JAF = DATA / "jaf"


def find_freeze(path: str, min_s: float = 2.5) -> tuple[float, float] | None:
    """最初の「静止画が min_s 秒以上続く」区間 (Thinking Time) を返す。"""
    fr, ts, _ = read_frames(path, 2.0, 130)
    small = np.stack([np.asarray(f.convert("L").resize((32, 18)), np.float32) / 255 for f in fr])
    d = np.abs(small[1:] - small[:-1]).mean(axis=(1, 2))
    i = 0
    while i < len(d):
        if d[i] < 0.004 and ts[i + 1] > 7.0:
            j = i
            while j < len(d) and d[j] < 0.004:
                j += 1
            if ts[j] - ts[i] >= min_s:
                return ts[i], ts[j]
            i = j
        else:
            i += 1
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="zeroshot")
    ap.add_argument("--model", default=os.environ.get("JUDGE_MODEL", "Qwen/Qwen3-VL-4B-Instruct"))
    ap.add_argument("--checks", default="near_miss,phone_driving")
    a = ap.parse_args()
    judge = VideoJudge(a.model, calibration=str(DATA / "calibration.json"))
    keys = a.checks.split(",")
    qs = [q for k in keys for q in vqs(BY_KEY[k])]
    rows = []
    for m in json.load(open(JAF / "list.json")):
        path = JAF / f'{m["cat"]}_{m["scene"]}.mp4'
        if not path.exists():
            continue
        fz = find_freeze(str(path))
        res = judge.judge(str(path), qs, state="Road safety footage.", max_s=90.0, scene_question=scene_vq(), sub_questions=sub_vqs())
        apply_scene_filter(res, keys)
        merge_long(res)
        scenes = res["scenes"]
        nm = [w for w in res["questions"]["near_miss"]["series"] if not w.get("long")]
        row = {"file": path.name, "title": m["title"], "freeze": fz, "scenes": [(round(s["t0"], 1), round(s["t1"], 1), s.get("type_ja")) for s in scenes]}
        if fz:
            # 事象区間 = 静止の終わりから次のカットまで (静止の終わり以降で最初のシーン境界)
            nxt = [s["t0"] for s in scenes if s["t0"] > fz[1] + 0.6]
            ev_end = nxt[0] if nxt else fz[1] + 8.0
            ev = [w["p"][0] for w in nm if w["t0"] >= fz[1] - 0.6 and w["t0"] < ev_end and not w.get("skipped")]
            pre = [w["p"][0] for w in nm if w["t1"] <= fz[0] and w["t0"] >= 7.0 and not w.get("skipped")]
            row.update({"event": (round(fz[1], 1), round(ev_end, 1)), "p_event": max(ev) if ev else None, "p_pre": max(pre) if pre else None, "n_event_skipped": sum(1 for w in nm if w["t0"] >= fz[1] - 0.6 and w["t0"] < ev_end and w.get("skipped"))})
        title_ok = scenes and scenes[0]["t1"] < 8 and scenes[0].get("type_ja") == "その他"
        row["title_card_skipped"] = bool(title_ok)
        row["summary"] = [(it["title"], it["label"], round(it["p_max"], 2), round(it["t_max"], 1)) for it in summarize(res)["items"]]
        rows.append(row)
        print(f'{path.name:26s} freeze={fz and tuple(round(x) for x in fz)} event={row.get("event")} p_event={row.get("p_event") and round(row["p_event"], 2)} p_pre={row.get("p_pre") and round(row["p_pre"], 2)} skipped_in_event={row.get("n_event_skipped")} title_ok={title_ok}', flush=True)
        print("   scenes:", " ".join(f"{t0:.0f}-{t1:.0f}:{ty}" for t0, t1, ty in row["scenes"]), flush=True)
    ok = [r for r in rows if r.get("p_event") is not None and r.get("p_pre") is not None]
    hit = sum(1 for r in ok if r["p_event"] > r["p_pre"])
    det = sum(1 for r in ok if r["p_event"] >= 0.4)
    fa = sum(1 for r in ok if r["p_pre"] >= 0.4)
    print(f"\n事象区間 > 平常区間: {hit}/{len(ok)}   事象区間で要確認以上 (p>=0.4): {det}/{len(ok)}   平常区間で誤検出 (p>=0.4): {fa}/{len(ok)}   タイトル画面を除外: {sum(r['title_card_skipped'] for r in rows)}/{len(rows)}")
    json.dump({"mode": a.mode, "model": a.model, "rows": rows, "hit": hit, "det": det, "fa": fa, "n": len(ok)}, open(DATA / f"eval_jaf_{a.mode}.json", "w"), ensure_ascii=False, indent=1, default=str)


if __name__ == "__main__":
    main()
