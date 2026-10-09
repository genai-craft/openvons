"""kana 入力エンジンごとの既定の校正値を fit し、3 デモでの精度を比べる.

    .venv/bin/python scripts/fit_engine_calibration.py --asr http://127.0.0.1:8630 --out openvons/voice/engines_calibration.json

3 デモ (指令・駅・河川) の既定範囲で、デモの「事前学習」と同じ手順 (TTS で実体名・他状態の命令・文法外を合成し、
雑音 SNR 20/10 dB と背景音を混ぜ、校正を fit) をエンジンごとに回す。範囲を事前学習していないときに使う既定値は
3 デモの校正サンプルをまとめて 1 つ fit した値 (範囲ごとの fit は境界に張り付くことがあり、パラメータごとの中央値は
互いに合わない組み合わせになるため)。まとめた値で各デモのサンプルを判断し直した精度も記録する。読みの追加 (add_readings) は切る。

TTS は各デモの state/<app>/tts_cache を使う (デモと同じ音声、2 回目からは合成しない)。
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from openvons.core.decision import Thresholds  # noqa: E402
from openvons.core.none_calibration import Calibration, ece, fit  # noqa: E402
from openvons.voice.engine import Recognizer  # noqa: E402
from openvons.voice.engines import ENGINES, default_calibration, load_local  # noqa: E402
from openvons.voice.lexicon import Lexicon  # noqa: E402
from openvons.voice.synth import PretrainConfig, Pretrainer, TTSClient  # noqa: E402

APPS = {"road_cameras": "examples.road_cameras.app", "stations": "examples.stations.app", "kasen": "examples.kasen.app"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asr", default="http://127.0.0.1:8630", help="共有 ASR サーバー (local でこのプロセスに読む)")
    ap.add_argument("--engines", default="kana-whisper,komimi-v12,komimi-v12a,komimi-v12m,komimi-v12s")
    ap.add_argument("--apps", default="road_cameras,stations,kasen")
    ap.add_argument("--tts", default="voicevox://127.0.0.1:50021")
    ap.add_argument("--out", default=str(ROOT / "openvons/voice/engines_calibration.json"))
    ap.add_argument("--report", default=str(ROOT / "docs/kana_engines_eval.json"))
    args = ap.parse_args()
    old_rep = json.loads(Path(args.report).read_text(encoding="utf-8")) if Path(args.report).exists() else {}
    report: dict = {"date": time.strftime("%Y-%m-%d %H:%M"), "results": [], "pooled": old_rep.get("pooled", {})}
    keep = [r for r in old_rep.get("results", []) if r["engine"] not in args.engines.split(",") or r["app"] not in args.apps.split(",")]
    report["results"] = keep
    pooled: dict[str, dict[str, list]] = {}
    INIT = Calibration(2.5, 4.0, 1.4, 1.0)
    for app_name in args.apps.split(","):
        APP = importlib.import_module(APPS[app_name])
        app_dir = Path(APP.__file__).resolve().parent
        state_dir = ROOT / "state" / app_name
        lex_path = state_dir / "entities.json" if (state_dir / "entities.json").exists() else app_dir / APP.DATA_FILE
        tts = TTSClient(args.tts, cache_dir=state_dir / "tts_cache")
        for eng in args.engines.split(","):
            lex = Lexicon.load(lex_path)                    # 読み込み直す (前のエンジンの試験で辞書が変わらないように)
            scope = APP.default_scopes(lex)[0]
            inst = APP.AppClass(lex, scope)
            if args.asr in ("", "local"):
                asr = load_local(eng)
            else:
                from openvons.voice.remote import RemoteASR
                asr = RemoteASR(args.asr, eng)
            rec = Recognizer(asr, Calibration.from_dict(default_calibration(eng)) if eng == "kana-whisper" else INIT, Thresholds())
            cfg = PretrainConfig(carriers=["{%s}" % APP.SLOT, "{%s}を表示" % APP.SLOT], n_out_of_grammar=12, extra_states=APP.CALIBRATION_STATES, add_readings=False)
            t0 = time.time()
            pt = Pretrainer(tts, rec, inst.grammar, lex)
            res = pt.run(scope, [APP.SELECT_INTENT], slot=APP.SLOT, cfg=cfg, entities=inst.entities())
            pooled.setdefault(eng, {})[app_name] = list(getattr(pt, "last_samples", []))
            row = {"app": app_name, "scope": scope.id, "engine": eng, "n_utts": res.n_utts, "accuracy_default": round(res.accuracy, 4),
                   "accuracy_after": round(res.accuracy_after, 4), "accuracy_by_snr": {k: round(v, 4) for k, v in res.accuracy_by_snr.items()},
                   "none_recall": round(res.none_recall, 4), "false_accept": round(res.false_accept, 4),
                   "ece_before": round(res.ece_before, 4), "ece_after": round(res.ece_after, 4), "calibration": res.calibration,
                   "latency_ms": {k: round(v, 1) for k, v in res.latency_ms.items()}, "seconds": round(time.time() - t0, 1)}
            report["results"].append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    out = {}
    for eng, by_app in pooled.items():
        if eng == "kana-whisper":
            continue                     # kana-whisper の既定値は Calibration() のまま (首都国道で fit した事前分布)
        allS = [x for v in by_app.values() for x in v]
        cal = fit(allS, INIT)
        out[eng] = {k: round(v, 3) for k, v in cal.to_dict().items()}
        ev = {}
        for app_name, S in by_app.items():
            probs = [cal.probs(cs_, fs_, l_, nf_) for cs_, fs_, y, l_, nf_ in S]
            ys = [y for _, _, y, _, _ in S]; nones = [len(cs_) for cs_, *_ in S]
            pred = [int(p.argmax()) for p in probs]
            neg = [i for i, (y, n) in enumerate(zip(ys, nones)) if y == n]
            conf = np.array([p.max() for p in probs]); corr = np.array([int(a == b) for a, b in zip(pred, ys)])
            from openvons.voice.synth import _ece_top
            ev[app_name] = {"n": len(S), "accuracy": round(float(corr.mean()), 4),
                            "none_recall": round(float(np.mean([pred[i] == nones[i] for i in neg])), 4) if neg else None,
                            "false_accept_argmax": round(float(np.mean([pred[i] != nones[i] for i in neg])), 4) if neg else None,
                            "ece": round(float(_ece_top(conf, corr)), 4)}
        report["pooled"][eng] = {"calibration": out[eng], "n": len(allS), "by_app": ev}
        print(eng, "pooled", out[eng], json.dumps(ev, ensure_ascii=False), flush=True)
    old = json.loads(Path(args.out).read_text()) if Path(args.out).exists() else {}
    old.update(out)
    Path(args.out).write_text(json.dumps(old, ensure_ascii=False, indent=1), encoding="utf-8")
    report["defaults"] = old
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print("defaults:", json.dumps(old, ensure_ascii=False))


if __name__ == "__main__":
    main()
