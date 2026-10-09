"""C の振り分け (openvons/voice/wasm/ov_route.c、ブラウザの WebAssembly と同じソース) == Python の Recognizer.

音声は使わず、CTC の log-softmax 行列を作って両方に渡す (「ある候補の piece 列をなぞる行列」+ 雑音、文法外っぽい行列)。
komimi (KOMIMI_HOME) の語彙と共有ライブラリ、openvons/voice/wasm/libovroute.so (openvons/voice/wasm/build.sh) が無ければ skip。
TTS を使う大きな確認は scripts/check_route_equivalence.py。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LIB = ROOT / "openvons/voice/wasm/libovroute.so"


def _deps():
    from openvons.voice.komimi_asr import KOMIMI_HOME
    kmm = KOMIMI_HOME / "models" / "ja_v12s_i8_c16.kmm"
    vocab = KOMIMI_HOME / "models" / "ja1024_vocab.json"
    if not (LIB.exists() and kmm.exists() and vocab.exists()):
        pytest.skip("libovroute.so / komimi のモデルが無い")
    sys.path.insert(0, str(KOMIMI_HOME))
    return kmm, vocab


class LpASR:
    """encode() が与えた行列をそのまま返す ASR (komimi の greedy / CTC 採点 / トークナイザを使う)。"""
    def __init__(self, vocab):
        from komimi.engine import Tokenizer, ctc_forward
        self.tok = Tokenizer(vocab); self.blank = len(self.tok.pieces); self._ctc = ctc_forward
        self.lp = None

    def encode(self, wav):
        from openvons.voice.asr_types import Encoded
        return Encoded(self.lp, 1.0, 0.0)

    def transcribe(self, enc, max_new_tokens=64):
        from openvons.voice.asr_types import Transcript
        best = enc.hidden.argmax(-1); ids, prev = [], self.blank
        for b in best.tolist():
            if b != self.blank and b != prev:
                ids.append(b)
            prev = b
        lp = float(self._ctc(enc.hidden, [ids], self.blank)[0]) if ids else float(enc.hidden[:, self.blank].sum())
        return Transcript(self.tok.decode(ids), lp, len(ids), 0.0, ids)

    def tokenize(self, k):
        return self.tok.encode(k)

    def score_tokens(self, enc, token_lists, batch_size=0):
        return np.asarray(self._ctc(enc.hidden, [list(t) for t in token_lists], self.blank), dtype=np.float64), np.array([len(t) for t in token_lists])


def _lp_for(ids, V, rng, T_per=3, noise=1.0):
    """ids をなぞる (各 piece を T_per フレーム、間に blank) 行列 + 雑音。"""
    frames = []
    for i in ids:
        frames += [i] * T_per + [V - 1]
    frames = [V - 1] * 2 + frames + [V - 1] * 2
    x = rng.standard_normal((len(frames), V)).astype(np.float32) * noise
    for t, f in enumerate(frames):
        x[t, f] += 8.0
    x -= x.max(1, keepdims=True)
    return (x - np.log(np.exp(x).sum(1, keepdims=True))).astype(np.float32)


def test_c_router_matches_python():
    kmm, vocab = _deps()
    sys.path.insert(0, str(ROOT / "scripts"))
    from check_route_equivalence import CRoute, pack_commands, params
    from openvons.core.decision import Thresholds
    from openvons.core.none_calibration import Calibration
    from openvons.voice.engine import Recognizer
    from openvons.voice.grammar import CommandSet, Hypothesis
    names = ["トーキョー", "シンジュク", "シブヤ", "イケブクロ", "ウエノ", "シナガワ", "メグロ", "エビス", "オーサキ", "タバタ",
             "ニッポリ", "ウグイスダニ", "オカチマチ", "アキハバラ", "カンダ", "ユーラクチョー", "シンバシ", "ハママツチョー", "タマチ", "ゴタンダ"]
    hyps = []
    for n in names:
        for carrier, risk in (("{}", "low"), ("{}オヒョージ", "low"), ("{}ニイキタイ", "medium")):
            hyps.append(Hypothesis("select", carrier.format(n), carrier.format(n), {"st": n}, risk=risk))
    hyps += [Hypothesis("yes", "はい", "ハイ", {}, confirmable=False, allow_embed=False), Hypothesis("no", "いいえ", "イイエ", {}, confirmable=False, positive=False, allow_embed=False)]
    cs = CommandSet(hyps, "S")
    asr = LpASR(vocab)
    cal = Calibration(2.3, 3.0, 1.8, 1.2)
    rec = Recognizer(asr, cal, Thresholds())
    cr = CRoute(LIB, kmm, vocab); cr.set_params(params(cal, Thresholds(), rec))
    blob, means = pack_commands(cs); assert cr.set_commands(blob) == len(hyps)
    rng = np.random.default_rng(0)
    utts = [h.kana for h in hyps[::3]] + ["エートトーキョーオネガイ", "ハイオセワニナッテオリマス", "キョーワイーテンキ", "シブヤシブヤシブヤ", "ハイハイハイハイハイハイハイ"]
    n_ok = 0
    for i, text in enumerate(utts):
        asr.lp = _lp_for(asr.tok.encode(text), 1025, rng, noise=0.5 + (i % 3))
        d = rec.recognize(None, cs)
        c = cr.route_lp(asr.lp)
        assert d.free_kana == c["free_kana"], text
        assert d.action == c["action"], (text, d.action, c)
        py_top = means[cs.hyps.index(d.top.hypothesis)] if d.top else -1
        c_top = means[c["meanings"][0][0]] if c["meanings"] else -1
        assert py_top == c_top, text
        assert abs(d.none_prob - c["none_prob"]) < 1e-3, text
        n_ok += 1
    assert n_ok == len(utts)
