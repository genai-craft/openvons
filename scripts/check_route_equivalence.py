"""C (WebAssembly と同じソース) の振り分けが Python の Recognizer と同じ判断を出すかを確かめる.

    .venv/bin/python scripts/check_route_equivalence.py --engine komimi-v12a --n 40

3 デモ (指令・駅・河川) の既定範囲の全状態について、TTS (VOICEVOX) で
  実体名 / 状態ごとの命令 / 文法外の発話 を合成し (半分は SNR 10 dB の雑音付き)、
同じ log-softmax 行列を Python (openvons.voice.engine.Recognizer + KomimiASR) と C (ov_route.c、ovk_route_logprobs) に通して
  自由認識・絞り込みの集合・最尤の意味・該当なしの確率・判断 を比べる。音声から通す経路 (ovk_route) も別に比べる。
"""
from __future__ import annotations

import argparse
import ctypes
import importlib
import json
import random
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from openvons.core.decision import Thresholds  # noqa: E402
from openvons.core.none_calibration import Calibration  # noqa: E402
from openvons.voice.engine import Recognizer  # noqa: E402
from openvons.voice.engines import ENGINES, default_calibration  # noqa: E402
from openvons.voice.komimi_asr import KOMIMI_HOME, KomimiASR  # noqa: E402
from openvons.voice.lexicon import Lexicon  # noqa: E402
from openvons.voice.synth import add_noise  # noqa: E402

RISK = {"low": 0, "medium": 1, "high": 2}


def pack_commands(cs) -> tuple[bytes, list[int]]:
    """仮説を C の ovk_set_commands の形に詰める (ブラウザの kana.js と同じ形)。意味 id は初出順。"""
    mid: dict = {}
    out = [struct.pack("<I", len(cs.hyps))]
    means = []
    for h in cs.hyps:
        m = mid.setdefault(h.meaning, len(mid)); means.append(m)
        kb = h.kana.encode("utf-8")
        flags = (1 if h.allow_embed else 0) | (2 if h.confirmable else 0) | (4 if h.positive else 0)
        out.append(struct.pack("<IBBH", m, RISK.get(h.risk, 0), flags, len(kb)) + kb)
    return b"".join(out), means


def params(cal: Calibration, th: Thresholds, rec: Recognizer) -> list[float]:
    return [cal.temperature, cal.none_bias, cal.len_bonus, cal.residual_penalty, th.execute, th.confirm, th.execute_medium,
            th.clear_min, th.clear_ratio, th.clear_none_max, th.answer_yes, th.answer_no, rec.shortlist_k, 1.0 if rec.embed else 0.0,
            rec.embed_min_ratio, rec.embed_max_residual, rec.embed_min_morae, rec.embed_penalty]


class CRoute:
    def __init__(self, lib_path: Path, kmm: Path, vocab: Path):
        self.lib = ctypes.CDLL(str(lib_path))
        L = self.lib
        L.ovk_new.restype = ctypes.c_void_p; L.ovk_new.argtypes = [ctypes.c_char_p, ctypes.c_int]
        L.ovk_set_vocab_scores.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        L.ovk_set_commands.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
        L.ovk_set_params.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        L.ovk_route_logprobs.restype = ctypes.c_char_p; L.ovk_route_logprobs.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        L.ovk_route.restype = ctypes.c_char_p; L.ovk_route.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        b = kmm.read_bytes()
        self.h = L.ovk_new(b, len(b))
        sc = np.asarray(json.loads(vocab.read_text(encoding="utf-8"))["scores"], dtype=np.float32)
        L.ovk_set_vocab_scores(self.h, sc.ctypes.data, len(sc))

    def set_commands(self, blob: bytes):
        return self.lib.ovk_set_commands(self.h, blob, len(blob))

    def set_params(self, p: list[float]):
        a = np.asarray(p, dtype=np.float64); self.lib.ovk_set_params(self.h, a.ctypes.data, len(a))

    def route_lp(self, lp: np.ndarray) -> dict:
        lp = np.ascontiguousarray(lp, dtype=np.float32)
        return json.loads(self.lib.ovk_route_logprobs(self.h, lp.ctypes.data, lp.shape[0]).decode())

    def route(self, wav: np.ndarray) -> dict:
        w = np.ascontiguousarray(wav, dtype=np.float32)
        return json.loads(self.lib.ovk_route(self.h, w.ctypes.data, len(w)).decode())


def utterances(app_mod, inst, state: str, rng: random.Random, n: int) -> list[tuple[str, str]]:
    """(合成する読み, 種類)。実体名は登録読みで、命令は例文の読み (例文の漢字は pyopenjtalk で読ませる)。"""
    from openvons.voice.kana import g2p
    cs = inst.command_set()
    picks = rng.sample(cs.hyps, min(n, len(cs.hyps)))
    out = [(h.kana, "grammar") for h in picks]
    for t in ["ハイ、オセワニナッテオリマス", "キョーワイーテンキデスネ", "ソレジャア、マタアシタ", "エートナンダッケ"]:
        out.append((t, "oog"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="komimi-v12a")
    ap.add_argument("--n", type=int, default=24, help="状態ごとの文法内の発話数")
    ap.add_argument("--apps", default="examples.road_cameras.app,examples.stations.app,examples.kasen.app")
    ap.add_argument("--lib", default=str(ROOT / "openvons/voice/wasm/libovroute.so"))
    ap.add_argument("--tts", default="voicevox://127.0.0.1:50021")
    args = ap.parse_args()
    from openvons.tts import get_backend
    tts = get_backend(args.tts)
    spec = ENGINES[args.engine]
    asr = KomimiASR(spec.file)
    cal = Calibration.from_dict(default_calibration(args.engine)); th = Thresholds()
    rec = Recognizer(asr, cal, th)
    cr = CRoute(Path(args.lib), KOMIMI_HOME / "models" / spec.file, KOMIMI_HOME / "models" / "ja1024_vocab.json")
    cr.set_params(params(cal, th, rec))
    rng = random.Random(0)
    tot = {"n": 0, "free": 0, "short": 0, "top": 0, "action": 0, "pdiff": 0.0, "wav_action": 0, "wav_top": 0}
    bad = []
    for mod in args.apps.split(","):
        APP = importlib.import_module(mod)
        app_dir = Path(APP.__file__).resolve().parent
        lex = Lexicon.load(app_dir / APP.DATA_FILE)
        inst = APP.AppClass(lex, APP.default_scopes(lex)[0])
        for state in APP.STATES:
            inst.sm.state = state
            cs = inst.command_set()
            if not len(cs):
                continue
            blob, means = pack_commands(cs)
            assert cr.set_commands(blob) == len(cs.hyps)
            for text, kind in utterances(APP, inst, state, rng, args.n):
                wav = tts.synth(text, seed=1 + rng.randrange(3))
                if rng.random() < 0.5:
                    wav = add_noise(wav, 10.0, rng)
                enc = asr.encode(wav)
                d = rec.recognize(wav, cs)
                c = cr.route_lp(enc.hidden)
                cw = cr.route(wav)
                tot["n"] += 1
                py_short = sorted(cs.hyps.index(x.hypothesis) for x in d.candidates) if False else None
                a = rec.analyze(wav, cs)
                py_set = sorted(cs.hyps.index(h) for h in a.cands)
                c_set = sorted(x[0] for x in c["cands"])
                py_top = means[cs.hyps.index(d.top.hypothesis)] if d.top else -1
                c_top = means[c["meanings"][0][0]] if c["meanings"] else -1
                cw_top = means[cw["meanings"][0][0]] if cw["meanings"] else -1
                tot["free"] += d.free_kana == c["free_kana"]
                tot["short"] += py_set == c_set
                tot["top"] += py_top == c_top
                tot["action"] += d.action == c["action"]
                tot["wav_action"] += d.action == cw["action"]
                tot["wav_top"] += py_top == cw_top
                pd = abs((d.top.prob if d.top else 0) - (c["meanings"][0][1] if c["meanings"] else 0)) + abs(d.none_prob - c["none_prob"])
                tot["pdiff"] = max(tot["pdiff"], pd)
                if d.action != c["action"] or py_top != c_top or py_set != c_set or pd > 1e-3:
                    bad.append({"app": app_dir.name, "state": state, "text": text, "kind": kind, "py": [d.free_kana, d.action, py_top, round(d.none_prob, 4)],
                                "c": [c["free_kana"], c["action"], c_top, round(c["none_prob"], 4)], "short_eq": py_set == c_set, "pd": round(pd, 5),
                                "py_only": sorted(set(py_set) - set(c_set))[:5], "c_only": sorted(set(c_set) - set(py_set))[:5]})
            print(f"{app_dir.name:13s} {state:10s} hyps {len(cs):6d}  累計 {tot['n']} 件  不一致 {len(bad)}", flush=True)
    n = tot["n"]
    print(json.dumps({k: (round(v / n, 4) if k not in ("n", "pdiff") else v) for k, v in tot.items()}, ensure_ascii=False))
    for b in bad[:15]:
        print(json.dumps(b, ensure_ascii=False))


if __name__ == "__main__":
    main()
