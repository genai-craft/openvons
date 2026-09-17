"""合成音声による定量評価 (docs/evaluation.md の数字を出す).

条件:
  A. 自由認識 + カナ最近傍 (リスコアなし)            … 従来のキーワード照合相当
  B. 自由認識 -> 絞り込み -> 教師強制リスコア (本方式)
  範囲: 首都国道事務所 (100 台) / 千葉県 (数百台) / 全国 (2,932 台)
  雑音: なし / SNR 20 / 10 / 5 dB、隣の声の重ね合わせ
  文法外発話の棄却率、遅延の内訳

  CUDA_VISIBLE_DEVICES=4 .venv/bin/python scripts/eval_synthetic.py --n 60 --out experiments/eval_synth.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples.road_cameras.app import INTENTS  # noqa: E402
from jev.voice import kana as K  # noqa: E402
from jev.voice.asr import KanaASR  # noqa: E402
from jev.voice.calibration import Calibration, fit  # noqa: E402
from jev.voice.engine import Recognizer  # noqa: E402
from jev.voice.grammar import Grammar  # noqa: E402
from jev.voice.lexicon import Lexicon, Scope  # noqa: E402
from jev.voice.synth import OUT_OF_GRAMMAR, TTSClient, add_noise, mix_background, pad_silence, tts_text  # noqa: E402

CARRIERS = ["{camera}", "{camera}を表示", "{camera}出して", "{camera}に切り替え"]
PTZ = {"pan_right": ["もっと右に向けて", "右に振って"], "pan_left": ["左に向けて"], "tilt_up": ["上に向いて"], "tilt_down": ["もっと下に向けて"],
       "zoom_in": ["もっと寄って", "ズームイン"], "zoom_out": ["もっと引いて", "ズームアウト"], "home": ["ホームに戻して"], "back": ["一覧に戻る", "戻る"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="範囲ごとのカメラ発話数")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--out", default=str(ROOT / "experiments" / "eval_synth.json"))
    ap.add_argument("--tts", default="voicevox://127.0.0.1:50021")
    ap.add_argument("--asr-model", default="sbintuitions/kana-whisper", help="kana-whisper か、蒸留した小型モデルのディレクトリ")
    args = ap.parse_args()
    rng = random.Random(0)
    seeds = [int(x) for x in args.seeds.split(",")]

    lex = Lexicon.load(ROOT / "examples" / "road_cameras" / "cameras.json")
    scopes = {
        "首都国道 (100台)": Scope("shuto", "首都国道", filters={"office": ["首都国道事務所"]}),
        "千葉県 (全事務所)": Scope("chiba", "千葉県", filters={"pref": ["千葉県"]}),
        "全国 (2,932台)": Scope("all", "全国", filters={"bureau": lex.values_of("bureau")}),
    }
    asr = KanaASR(args.asr_model)
    print(f"asr {args.asr_model}: {asr.n_params / 1e6:.0f}M params", flush=True)
    rec = Recognizer(asr)
    g = Grammar(INTENTS)
    tts = TTSClient(args.tts, cache_dir="/data/openjev/state/tts_cache")
    assert tts.ok(), "TTS unavailable"

    # 評価に使うカメラは首都国道から (全範囲で同じ発話を使い、選択肢の増加だけの効果を見る)
    shuto = lex.in_scope(scopes["首都国道 (100台)"])
    picks = rng.sample(shuto, min(args.n, len(shuto)))
    utts = []   # (wav, entity_id, text, condition)
    print(f"synthesizing {len(picks) * len(seeds)} camera utterances ...", flush=True)
    bg = [tts.synth(t, seed=1) for t in OUT_OF_GRAMMAR[:4]]
    conds = ["clean", "snr20", "snr10", "snr5", "babble"]
    for e in picks:
        for sd in seeds:
            carrier = rng.choice(CARRIERS)
            text = tts_text(e, carrier, "camera")
            wav = tts.synth(text, seed=sd)
            cond = rng.choice(conds)
            w = pad_silence(wav, rng)
            if cond.startswith("snr"):
                w = add_noise(w, float(cond[3:]), rng)
            elif cond == "babble":
                w = mix_background(w, rng.choice(bg), -10.0, rng)
            utts.append((w, e.id, carrier.replace("{camera}", e.label), cond))
    oog = [(pad_silence(tts.synth(t, seed=seeds[i % len(seeds)]), rng), None, t, "oog") for i, t in enumerate(OUT_OF_GRAMMAR)]
    ptz = []
    for intent, texts in PTZ.items():
        for t in texts:
            for sd in seeds[:2]:
                ptz.append((pad_silence(tts.synth(t, seed=sd), rng), intent, t, "ptz"))

    results = {"asr_model": args.asr_model, "asr_params_m": round(asr.n_params / 1e6), "n_camera_utts": len(utts), "n_oog": len(oog), "n_ptz": len(ptz), "scopes": {}}
    for sname, sc in scopes.items():
        ents = lex.in_scope(sc)
        t0 = time.time(); cs = g.compile(["select_camera", "help"], ents, "WALL"); t_compile = time.time() - t0
        row = {"n_cameras": len(ents), "n_hypotheses": len(cs), "compile_s": round(t_compile, 2), "A_nearest": {}, "B_rescore": {}, "latency_ms": {}}
        hitA = {c: [0, 0] for c in conds}; hitB = {c: [0, 0] for c in conds}
        lat = {}
        confB = []; corrB = []
        samples = []
        dump = []
        for w, eid, text, cond in utts:
            d = rec.recognize(w, cs)
            for k, v in d.timings.items():
                lat.setdefault(k, []).append(v)
            # A: 最近傍 (自由認識カナに最も近い仮説)
            idx = cs.shortlist(d.free_kana, 1)
            a_ok = bool(idx) and cs.hyps[idx[0]].slots.get("camera") == eid
            hitA[cond][0] += int(a_ok); hitA[cond][1] += 1
            b_ok = d.action in ("execute", "confirm") and d.top is not None and d.top.hypothesis.slots.get("camera") == eid
            hitB[cond][0] += int(b_ok); hitB[cond][1] += 1
            confB.append(d.top.prob if d.top else 0.0); corrB.append(int(b_ok))
            # 校正サンプル (実行時と同じ analyze 経路)
            a = rec.analyze(w, cs)
            y = a.gold_index(lambda h: h.slots.get("camera") == eid)
            samples.append(a.sample(lambda h: h.slots.get("camera") == eid))
            dump.append({"scores": a.scores.tolist(), "free_score": a.free_score, "y": y, "cands": [h.kana for h in a.cands], "free": a.free_kana, "lens": a.lens.tolist(), "n_free": a.n_free,
                         "cond": cond, "text": text, "sims": [K.mora_similarity(a.free_kana, h.kana) for h in a.cands]})
        row["A_nearest"] = {c: round(h[0] / max(h[1], 1), 3) for c, h in hitA.items()}
        row["A_nearest"]["all"] = round(sum(h[0] for h in hitA.values()) / len(utts), 3)
        row["B_rescore"] = {c: round(h[0] / max(h[1], 1), 3) for c, h in hitB.items()}
        row["B_rescore"]["all"] = round(sum(h[0] for h in hitB.values()) / len(utts), 3)
        row["latency_ms"] = {k: round(float(np.median(v)), 1) for k, v in lat.items()}
        row["latency_p95_ms"] = round(float(np.percentile(lat["total"], 95)), 1)
        # 文法外
        none_ok = 0; false_acc = 0
        for w, _, text, _ in oog:
            d = rec.recognize(w, cs)
            none_ok += int(d.action in ("none", "reject")); false_acc += int(d.action == "execute")
            a = rec.analyze(w, cs)
            if a.cands:
                samples.append(a.sample(lambda h: False))
                dump.append({"scores": a.scores.tolist(), "free_score": a.free_score, "y": len(a.cands), "cands": [h.kana for h in a.cands], "free": a.free_kana, "lens": a.lens.tolist(), "n_free": a.n_free,
                             "cond": "oog", "text": text, "sims": [K.mora_similarity(a.free_kana, h.kana) for h in a.cands]})
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(str(args.out).replace(".json", f"_samples_{sc.id}.json")).write_text(json.dumps(dump, ensure_ascii=False), encoding="utf-8")
        row["oog_rejected"] = round(none_ok / len(oog), 3); row["oog_false_execute"] = round(false_acc / len(oog), 3)
        # 校正
        cal = fit(samples)
        row["calibration"] = cal.to_dict()
        def top_ece(cal_):
            conf = []; corr = []
            for cs_, fs_, y, lens_, nf_ in samples:
                p = cal_.probs(cs_, fs_, lens_, nf_); conf.append(p.max()); corr.append(int(p.argmax() == y))
            conf = np.array(conf); corr = np.array(corr)
            bins = np.linspace(0, 1, 11); e = 0.0
            for lo, hi in zip(bins[:-1], bins[1:]):
                m = (conf > lo) & (conf <= hi)
                if m.any(): e += m.mean() * abs(corr[m].mean() - conf[m].mean())
            return round(float(e), 4), round(float(corr.mean()), 4)
        row["ece_before"], row["acc_argmax_before"] = top_ece(Calibration())
        row["ece_after"], row["acc_argmax_after"] = top_ece(cal)
        # 確信度別 (高確信のみ自動実行した場合の網羅率と精度)
        confB = np.array(confB); corrB = np.array(corrB)
        row["conf_gate"] = {}
        for th in (0.5, 0.85, 0.95):
            m = confB >= th
            row["conf_gate"][str(th)] = {"coverage": round(float(m.mean()), 3), "precision": round(float(corrB[m].mean()) if m.any() else 0.0, 3)}
        results["scopes"][sname] = row
        print(sname, json.dumps(row, ensure_ascii=False), flush=True)

    # PTZ (FOCUS 状態) と CONFIRM 状態
    ents = lex.in_scope(scopes["首都国道 (100台)"])
    cs_focus = g.compile(["pan_right", "pan_left", "tilt_up", "tilt_down", "zoom_in", "zoom_out", "home", "back", "save_preset", "select_camera", "help"], ents, "FOCUS")
    ok = 0
    detail = []
    for w, intent, text, _ in ptz:
        d = rec.recognize(w, cs_focus)
        good = d.action == "execute" and d.top and d.top.hypothesis.intent == intent
        ok += int(good)
        if not good:
            detail.append({"text": text, "free": d.free_kana, "action": d.action, "top": d.top.hypothesis.text if d.top else None, "p": round(d.top.prob, 3) if d.top else 0})
    results["ptz_focus"] = {"acc": round(ok / len(ptz), 3), "n": len(ptz), "n_hypotheses": len(cs_focus), "misses": detail}
    cs_conf = g.compile(["yes", "no"], [], "CONFIRM")
    yn = [("はい", "yes"), ("いいえ", "no"), ("そうです", "yes"), ("違います", "no"), ("お願いします", "yes"), ("キャンセル", "no")]
    ok = 0; n = 0
    for t, intent in yn:
        for sd in seeds:
            w = pad_silence(tts.synth(t, seed=sd), rng)
            w = add_noise(w, 10.0, rng)
            d = rec.recognize(w, cs_conf)
            ok += int(d.action == "execute" and d.top and d.top.hypothesis.intent == intent); n += 1
    results["confirm_state"] = {"acc": round(ok / n, 3), "n": n, "n_hypotheses": len(cs_conf), "snr_db": 10}
    # 確認待ちで電話相手に言った「はい、〜」を拾わないか (該当なし/棄却が正解)
    phone = ["はい、お世話になっております", "はいはい、大丈夫です", "いいえ、こちらこそ", "はい、少々お待ちください", "そうですね、確認します"]
    ok = 0; n = 0; detail = []
    for t in phone:
        for sd in seeds[:2]:
            w = pad_silence(tts.synth(t, seed=sd), rng)
            d = rec.recognize(w, cs_conf)
            good = d.action in ("none", "reject"); ok += int(good); n += 1
            if not good:
                detail.append({"text": t, "free": d.free_kana, "top": d.top.hypothesis.text if d.top else None, "p": round(d.top.prob, 3) if d.top else 0})
    results["confirm_phone_reject"] = {"rate": round(ok / n, 3), "n": n, "false_accepts": detail}
    print("ptz", results["ptz_focus"], "confirm", results["confirm_state"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved", args.out)


if __name__ == "__main__":
    main()
