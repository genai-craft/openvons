"""fit_engine_calibration.py の結果 (範囲ごと・エンジンごとの校正値と精度) を各デモの担当範囲 (state/<app>/scopes.json) に書き込む.

    scripts/serve_demo.sh stop ... (書き込みはデモを止めてから。動いているデモは自分の持つ範囲で上書きする)
    .venv/bin/python scripts/apply_engine_calibration.py --report docs/kana_engines_eval.json

komimi 系は常に書く。kana-whisper はその範囲にまだ校正が無いときだけ書く (画面の「事前学習」で採った読みの追加つきの校正を壊さない)。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=str(ROOT / "docs/kana_engines_eval.json"))
    ap.add_argument("--state", default=str(ROOT / "state"))
    args = ap.parse_args()
    rep = json.loads(Path(args.report).read_text(encoding="utf-8"))
    by_app: dict[str, list[dict]] = {}
    for r in rep["results"]:
        by_app.setdefault(r["app"], []).append(r)
    for app, rows in by_app.items():
        p = Path(args.state) / app / "scopes.json"
        if not p.exists():
            print(f"{app}: {p} が無い (デモを一度起動すると作られる)"); continue
        scopes = json.loads(p.read_text(encoding="utf-8"))
        n = 0
        for r in rows:
            sc = next((s for s in scopes if s["id"] == r["scope"]), None)
            if sc is None:
                continue
            prof = sc.setdefault("profile", {}) or {}
            sc["profile"] = prof
            has_kw = bool((prof.get("calibrations") or {}).get("kana-whisper") or prof.get("calibration"))
            if r["engine"] == "kana-whisper" and has_kw:
                continue
            prof.setdefault("calibrations", {})[r["engine"]] = r["calibration"]
            prof.setdefault("engines", {})[r["engine"]] = {"accuracy": r["accuracy_default"], "accuracy_after": r["accuracy_after"],
                                                         "none_recall": r["none_recall"], "false_accept": r["false_accept"], "ece_after": r["ece_after"],
                                                         "n_utts": r["n_utts"], "trained_at": rep.get("date"), "source": "fit_engine_calibration"}
            n += 1
        p.write_text(json.dumps(scopes, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{app}: {n} 件書き込み ({p})")


if __name__ == "__main__":
    main()
