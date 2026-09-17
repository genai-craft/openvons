"""GPU 不要の単体テスト: カナ正規化・文法展開・状態機械・判断ポリシー・校正."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openvons.voice import kana as K
from openvons.core.none_calibration import Calibration, fit
from openvons.core.decision import Thresholds
from openvons.voice.engine import Candidate, Recognizer
from openvons.voice.grammar import Grammar, Hypothesis, Intent, expand_template
from openvons.voice.lexicon import Entity, Lexicon, Scope
from openvons.voice.state import StateDef, StateMachine


def test_normalize_conventions():
    assert K.normalize("ﾌﾅﾊﾞｼ") == "フナバシ"
    assert K.normalize("トウキョウ") == "トーキョー"
    assert K.normalize("チュウオウク") == "チューオーク"
    assert K.normalize("センセイ") == "センセー"
    assert K.normalize("ヲ") == "オ"
    assert K.normalize("ふなばし みなみ") == "フナバシミナミ"


def test_mora():
    assert K.mora_split("キャクダリー") == ["キャ", "ク", "ダ", "リ", "ー"]
    assert K.mora_distance("ナラシノ", "ナシラノ") == 2
    assert K.digits_to_kana(14) == "ジューヨン"
    assert K.digits_to_kana(1) == "イチ"


def test_expand_template():
    seqs = expand_template("{camera}[を](表示|出して)")
    texts = {"".join(v if k == "lit" else "{" + v + "}" for k, v in s) for s in seqs}
    assert texts == {"{camera}を表示", "{camera}を出して", "{camera}表示", "{camera}出して"}


def test_grammar_compile_and_shortlist():
    g = Grammar([Intent("select", ["{camera}[を]表示", "{camera}"]), Intent("right", ["右"], params={"dir": "right"})])
    ents = [Entity("a", "船橋南1上り", readings=["フナバシミナミイチノボリ"]), Entity("b", "谷津2下り", readings=["ヤツニクダリ"])]
    cs = g.compile(["select", "right"], ents)
    assert cs.n_meanings == 3
    kanas = set(cs.kanas)
    assert "フナバシミナミイチノボリオヒョージ" in kanas and "ヤツニクダリ" in kanas and "ミギ" in kanas
    top = cs.hyps[cs.shortlist("ヤツニクダリオヒョージ", 1)[0]]
    assert top.slots["camera"] == "b"


def test_scope_filters():
    lex = Lexicon([Entity("a", "A", attrs={"office": "X", "route": 1}), Entity("b", "B", attrs={"office": "Y", "route": 1}), Entity("c", "C", attrs={"office": "X", "route": 2})])
    assert {e.id for e in lex.in_scope(Scope("s", "s", filters={"office": ["X"]}))} == {"a", "c"}
    assert {e.id for e in lex.in_scope(Scope("s", "s", filters={"office": ["X"], "route": [2]}))} == {"c"}
    assert {e.id for e in lex.in_scope(Scope("s", "s", filters={"office": ["X"]}, ids=["b"], exclude_ids=["a"]))} == {"b", "c"}


def test_state_machine():
    sm = StateMachine({"A": StateDef("A", ["x"]), "B": StateDef("B", ["y"])}, "A")
    assert sm.allowed_intents() == ["x"]
    sm.goto("B", cam="1")
    assert sm.state == "B" and sm.context["cam"] == "1"
    sm.back()
    assert sm.state == "A" and "cam" not in sm.context


def _cand(prob, risk="low"):
    return Candidate(Hypothesis("i", "t", "k", {}, {}, risk), prob, -1.0)


def test_decision_policy():
    r = Recognizer.__new__(Recognizer)
    r.thresholds = Thresholds()
    assert r._decide(_cand(0.9), 0.05)[0] == "execute"
    assert r._decide(_cand(0.6), 0.05)[0] == "confirm"
    assert r._decide(_cand(0.2), 0.05)[0] == "reject"
    assert r._decide(_cand(0.3), 0.6)[0] == "none"
    assert r._decide(_cand(0.99, "high"), 0.0)[0] == "confirm"
    assert r._decide(_cand(0.9, "medium"), 0.0)[0] == "confirm"
    assert r._decide(_cand(0.97, "medium"), 0.0)[0] == "execute"


def test_calibration_none_option():
    cal = Calibration(1.0, 3.0)
    # 候補が自由認識と同じ尤度 -> 候補が e^3 倍有利
    p = cal.probs(np.array([-10.0]), -10.0)
    assert p[0] > 0.95
    # 候補が自由認識より 10 nats 悪い -> 該当なし
    p = cal.probs(np.array([-20.0]), -10.0)
    assert p[-1] > 0.99


def test_calibration_fit_recovers_bias():
    rng = np.random.default_rng(0)
    samples = []
    for _ in range(200):
        free = -rng.uniform(5, 30)
        cs = np.array([free - rng.uniform(0, 1.0), free - rng.uniform(8, 20)])
        samples.append((cs, free, 0))            # 文法内: 正解は近い候補
    for _ in range(100):
        free = -rng.uniform(5, 30)
        cs = np.array([free - rng.uniform(6, 20), free - rng.uniform(10, 25)])
        samples.append((cs, free, 2))            # 文法外: 該当なし
    cal = fit(samples)
    assert 1.0 < cal.none_bias < 6.0
    acc = np.mean([cal.probs(cs, f).argmax() == y for cs, f, y in samples])
    assert acc > 0.95


def test_residual_penalty_blocks_short_candidate_in_long_utterance():
    """電話相手への「はい、お世話になっております」(13 トークン) に対し、候補「はい」(2 トークン) が
    尤度 -5 程度で済んでも、説明しない残り 11 トークンの罰則で該当なしが勝つこと。"""
    cal = Calibration()
    p = cal.probs(np.array([-5.0]), -0.1, np.array([2.0]), 13)
    assert p[-1] > 0.95
    # 逆に、雑音で 14 nats 落ちた長い候補 (12 トークン、残り 0) は救われる
    p = cal.probs(np.array([-14.0]), -0.2, np.array([12.0]), 12)
    assert p[0] > 0.9


def test_fit_four_params_smoke():
    rng = np.random.default_rng(1)
    S = []
    for _ in range(60):
        S.append((np.array([-1.0 * rng.uniform(1, 12)]), -0.2, 0, np.array([10.0]), 10))    # 文法内: 長い候補
        S.append((np.array([-5.0]), -0.1, 1, np.array([2.0]), 12))                            # 電話の「はい」: 該当なし
    cal = fit(S)
    assert cal.residual_penalty > 0.5 and cal.len_bonus > 0.5
    assert cal.probs(np.array([-5.0]), -0.1, np.array([2.0]), 12)[-1] > 0.9
