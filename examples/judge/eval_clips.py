"""試験動画 (manifest.json) を judge で回し、項目ごとの正解率と温度校正を出す。

  CUDA_VISIBLE_DEVICES=4 .venv/bin/python examples/judge/eval_clips.py           # 判定 → state/judge/eval.json, calibration.json
出力: 項目 × (問題あり / なし) の判定結果、窓単位の AUROC 相当 (事象窓 vs 通常窓の p(yes))、温度 T (NLL 最小)。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.judge.checks import BY_KEY, CHECKS  # noqa: E402
from examples.judge.server import summarize, vq  # noqa: E402
from openvons.vision.video_judge import VideoJudge  # noqa: E402

DATA = Path(os.environ.get("JUDGE_DATA", str(Path(__file__).resolve().parents[2] / "state" / "judge")))
CLIPS = DATA / "clips"


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """3 択の raw logit と正解 index から温度 T を NLL 最小で求める (グリッド)。"""
    best = (1e9, 1.0)
    for T in np.exp(np.linspace(np.log(0.3), np.log(10), 60)):
        z = logits / T; z = z - z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
        nll = -np.log(p[np.arange(len(labels)), labels] + 1e-9).mean()
        if nll < best[0]:
            best = (nll, float(T))
    return best[1]


def main():
    man = json.load(open(CLIPS / "manifest.json"))
    judge = VideoJudge(os.environ.get("JUDGE_MODEL", "Qwen/Qwen3-VL-4B-Instruct"))
    judge.T = {}  # 生の確率で回し、後で校正
    rows, win_logit, win_label = [], {c.key: [] for c in CHECKS}, {c.key: [] for c in CHECKS}
    for m in man:
        c = BY_KEY[m["check"]]
        res = judge.judge(str(CLIPS / m["file"]), [vq(x) for x in CHECKS], state="Fixed camera footage.", max_s=60)
        summ = summarize(res)
        it = next(i for i in summ["items"] if i["key"] == c.key)
        flagged = it["action"] in ("execute", "confirm")
        # 他項目の誤検出 (false alarm) も数える
        others = [i for i in summ["items"] if i["key"] != c.key and i["action"] in ("execute", "confirm")]
        rows.append({"file": m["file"], "check": c.key, "label": m["label"], "p_max": it["p_max"], "t_max": it["t_max"], "action": it["action"],
                     "correct": flagged == (m["label"] == 1), "false_alarms": [o["key"] for o in others]})
        # 窓単位: 事象区間に重なる窓 = 1、それ以外 = 0 (問題なし動画は全部 0)
        for w in res["questions"][c.key]["series"]:
            p = np.array(w["p"]); lg = np.log(p + 1e-9)
            ev = m["label"] == 1 and m["event_start"] is not None and w["t1"] > m["event_start"] and w["t0"] < m["event_end"]
            win_logit[c.key].append(lg); win_label[c.key].append(0 if ev else 1)   # 0 = yes, 1 = no
        print(f"{m['file']:32s} label={m['label']} -> {it['action']:8s} p_max={it['p_max']:.2f} t={it['t_max']:.0f}s {'OK' if rows[-1]['correct'] else 'NG'} false_alarms={rows[-1]['false_alarms']}", flush=True)
    # 集計
    per = {}
    for c in CHECKS:
        rs = [r for r in rows if r["check"] == c.key]
        if not rs:
            continue
        pos = [r for r in rs if r["label"] == 1]; neg = [r for r in rs if r["label"] == 0]
        lg = np.array(win_logit[c.key]); lb = np.array(win_label[c.key])
        auc = None
        if lg.size and (lb == 0).any() and (lb == 1).any():
            py = np.exp(lg[:, 0]) ; pos_s = py[lb == 0]; neg_s = py[lb == 1]
            auc = float(np.mean([[1.0 if a > b else 0.5 if a == b else 0.0 for b in neg_s] for a in pos_s]))
        per[c.key] = {"title": c.title, "n": len(rs), "recall": float(np.mean([r["correct"] for r in pos])) if pos else None,
                      "specificity": float(np.mean([r["correct"] for r in neg])) if neg else None,
                      "window_auroc": auc, "T": fit_temperature(lg, lb) if lg.size else 1.0}
    fa = sum(len(r["false_alarms"]) for r in rows) / max(len(rows), 1)
    out = {"rows": rows, "per_check": per, "false_alarms_per_clip": fa, "accuracy": float(np.mean([r["correct"] for r in rows]))}
    json.dump(out, open(DATA / "eval.json", "w"), ensure_ascii=False, indent=1)
    json.dump({"temperature": {k: v["T"] for k, v in per.items()} | {"_default": float(np.median([v["T"] for v in per.values()]))}}, open(DATA / "calibration.json", "w"), indent=1)
    print(json.dumps({k: {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()} for k, v in per.items()}, ensure_ascii=False, indent=1))
    print("accuracy %.3f  false alarms/clip %.2f" % (out["accuracy"], fa))


if __name__ == "__main__":
    main()
