"""認識パイプラインと判断.

recognize(音声, コマンド集合):
  1. encoder 1 回 -> 自由認識 (greedy) でカナ列と、その対数尤度 s_free
  2. カナ列に近い仮説を K 個に絞る (文字 Levenshtein)
  3. K 仮説を教師強制で採点 s_i = log p(仮説カナ | 音声)
  4. 校正 (温度 T、該当なしバイアス beta) -> [候補..., 該当なし] の確率
  5. 同じ意味 (意図 + スロット) の表層形は確率を足す
  6. 3 段閾値 (実行 / 確認 / 棄却) に意図の危険度を掛け合わせて判断

「該当なし」= 自由認識そのものを 1 つの選択肢として並べる。文法外の発話 (電話の相手に
話している等) では自由認識の尤度が候補を大きく上回るので、ここに確率が集まる。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from openvons.core.decision import Thresholds, decide
from openvons.core.none_calibration import Calibration

from . import kana as K
from .asr import KanaASR
from .grammar import CommandSet, Hypothesis
from rapidfuzz import fuzz


@dataclass
class Candidate:
    hypothesis: Hypothesis
    prob: float                 # 意味単位で足した確率
    score: float                # 最良表層形の log p
    surface_probs: dict[str, float] = field(default_factory=dict)   # 表層形カナ -> 確率

    def to_dict(self) -> dict[str, Any]:
        h = self.hypothesis
        return {"intent": h.intent, "text": h.text, "kana": h.kana, "slots": h.slots, "params": h.params,
                "risk": h.risk, "prob": round(self.prob, 4), "score": round(self.score, 2)}


@dataclass
class Decision:
    action: str                 # execute / confirm / reject / none
    top: Candidate | None
    candidates: list[Candidate]
    none_prob: float
    free_kana: str
    free_score: float
    timings: dict[str, float]
    state: str = ""
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"action": self.action, "top": self.top.to_dict() if self.top else None,
                "candidates": [c.to_dict() for c in self.candidates[:8]], "none_prob": round(self.none_prob, 4),
                "free_kana": self.free_kana, "free_score": round(self.free_score, 2),
                "timings_ms": {k: round(v, 1) for k, v in self.timings.items()}, "state": self.state, "reason": self.reason}


def _is_degenerate(kana: str, max_repeat: int = 6) -> bool:
    """同じ 1〜2 モーラが max_repeat 回以上連続したら反復ハルシネーションとみなす。"""
    m = K.mora_split(kana)
    for n in (1, 2):
        run = 1
        for i in range(n, len(m)):
            if m[i - n:i] == m[i:i + n] if n == 1 else m[i - n:i] == m[i:i + n]:
                run += 1
                if run >= max_repeat:
                    return True
            else:
                run = 1
    return False


@dataclass
class Analysis:
    """analyze() の結果。校正サンプル = (scores, free_score, 正解 index, lens, n_free)。"""
    cs: CommandSet
    free_kana: str
    free_score: float
    n_free: int
    cands: list[Hypothesis]
    scores: np.ndarray
    lens: np.ndarray
    timings: dict[str, float]
    degenerate: bool = False

    def gold_index(self, pred) -> int:
        """pred(h) が真の候補のうち最良スコアの index。無ければ該当なし (= len(cands))。"""
        g = [i for i, h in enumerate(self.cands) if pred(h)]
        return int(g[int(np.argmax(self.scores[g]))]) if g else len(self.cands)

    def sample(self, pred) -> tuple:
        return (self.scores, self.free_score, self.gold_index(pred), self.lens, self.n_free)


class Recognizer:
    def __init__(self, asr: KanaASR, calibration: Calibration | None = None, thresholds: Thresholds | None = None, shortlist_k: int = 16,
                 embed: bool = True, embed_min_ratio: float = 0.3, embed_max_residual: int = 14, embed_min_morae: int = 4,
                 embed_penalty: float = 0.5):
        self.asr = asr
        self.calibration = calibration or Calibration()
        self.thresholds = thresholds or Thresholds()
        self.shortlist_k = shortlist_k
        #: 埋め込み採点: 候補を自由認識の中に置き (前後の余計な語を自由認識のまま残す)、その文も採点する。
        #: 「えー、そうゆう、船橋南1上り、確認します」や隣の声の重なりで前後に語が付いても候補が負けなくなる。
        #: ただし候補が発話の embed_min_ratio 未満しか占めない (「はい、お世話になっております」の「はい」) 場合は
        #: 埋め込みを許さない。残り (前後の語) も embed_max_residual モーラまで。
        self.embed = embed
        self.embed_min_ratio = embed_min_ratio
        self.embed_max_residual = embed_max_residual
        self.embed_min_morae = embed_min_morae
        #: 埋め込み文は自由認識から借りたトークンの分だけ減点する (logit/トークン、温度の外で効く)。これが無いと「東京」だけの候補が
        #: 「新宿から東京まで」の全文を借りて説明でき、全文を自分で説明する経路候補と同点になって確率が割れる
        self.embed_penalty = embed_penalty

    def _embedded(self, free_kana: str, cand_kana: str) -> str | None:
        """候補を自由認識の最も似た区間に置き換えた文。条件を満たさなければ None。"""
        if not free_kana or cand_kana == free_kana:
            return None
        n_free = len(K.mora_split(free_kana)); n_cand = len(K.mora_split(cand_kana))
        if n_cand < self.embed_min_morae or n_cand < self.embed_min_ratio * n_free or n_free - n_cand > self.embed_max_residual:
            return None
        al = fuzz.partial_ratio_alignment(cand_kana, free_kana)
        if al is None or al.score < 55:
            return None
        emb = free_kana[:al.dest_start] + cand_kana + free_kana[al.dest_end:]
        return emb if emb != cand_kana else None

    # ------------------------------------------------------------------
    def analyze(self, wav: np.ndarray, cs: CommandSet, cal: Calibration | None = None) -> "Analysis":
        """認識の前半: 自由認識 → 絞り込み → (埋め込み込みの) 採点。校正サンプルの採取と recognize() が
        同じ経路を通るための入口 (校正時と実行時で採点の中身が違うと fit が壊れる)。"""
        cal = cal or self.calibration
        t_all = time.perf_counter()
        enc = self.asr.encode(wav)
        free = self.asr.transcribe(enc)
        free_kana = K.normalize(free.kana)
        timings = {"encode": enc.ms, "transcribe": free.ms}
        n_free = len(free.tokens or self.asr.tokenize(free_kana))
        a = Analysis(cs, free_kana, free.logprob, n_free, [], np.zeros(0), np.zeros(0), timings)
        if len(cs) == 0 or not free_kana or _is_degenerate(free_kana):
            a.degenerate = bool(free_kana) and _is_degenerate(free_kana)
            timings["total"] = (time.perf_counter() - t_all) * 1000
            return a
        t0 = time.perf_counter()
        idx = cs.shortlist(free_kana, self.shortlist_k)
        timings["shortlist"] = (time.perf_counter() - t0) * 1000
        cands = [cs.hyps[i] for i in idx]
        t0 = time.perf_counter()
        seqs = [self.asr.tokenize(h.kana) for h in cands]
        emb_index: list[int] = []
        if self.embed:
            for i, h in enumerate(cands):
                if not h.allow_embed:
                    continue
                e = self._embedded(free_kana, h.kana)
                if e is not None:
                    emb_index.append(i); seqs.append(self.asr.tokenize(e))
        all_scores, _ = self.asr.score_tokens(enc, seqs)
        scores = all_scores[:len(cands)].copy()
        lens = np.array([len(t) for t in seqs[:len(cands)]], dtype=np.float64)
        for j, i in enumerate(emb_index):
            # 埋め込み文 (前後の語を自由認識から取り込んだもの) と素の候補のうち、校正後 logit が大きい方を採る。
            # 埋め込み文は残り (説明しないトークン) が 0 になるので γ の罰則を受けない
            n_e = float(len(seqs[len(cands) + j]))
            borrowed = max(0.0, n_e - lens[i])
            # 借用罰則は構造的な事前分布なので温度の外 (logit 空間) で効かせる。score 側に入れると温度 3 の校正では
            # 8 トークン借りても 2.4 logit しか差がつかず、全文を自分で説明する経路候補と割れる
            s_e = all_scores[len(cands) + j] - self.embed_penalty * borrowed * cal.temperature
            z_bare = cal.logits(np.array([scores[i]]), free.logprob, np.array([lens[i]]), n_free)[0]
            z_emb = cal.logits(np.array([s_e]), free.logprob, np.array([n_e]), n_free)[0]
            if z_emb > z_bare:
                scores[i] = s_e; lens[i] = n_e
        timings["score"] = (time.perf_counter() - t0) * 1000
        timings["total"] = (time.perf_counter() - t_all) * 1000
        a.cands, a.scores, a.lens = cands, scores, lens
        return a

    def recognize(self, wav: np.ndarray, cs: CommandSet, calibration: Calibration | None = None) -> Decision:
        cal = calibration or self.calibration
        a = self.analyze(wav, cs, cal)
        if not a.cands:
            reason = "反復ハルシネーション" if a.degenerate else "empty"
            return Decision("none", None, [], 1.0, a.free_kana, a.free_score, a.timings, cs.state, reason)
        probs = cal.probs(a.scores, a.free_score, a.lens, a.n_free)
        none_prob = float(probs[-1])
        # 意味単位で確率を集約
        agg: dict[tuple, Candidate] = {}
        for h, s, p in zip(a.cands, a.scores, probs[:-1]):
            c = agg.get(h.meaning)
            if c is None:
                agg[h.meaning] = Candidate(h, float(p), float(s), {h.kana: float(p)})
            else:
                c.prob += float(p)
                c.surface_probs[h.kana] = float(p)
                if s > c.score:
                    c.score = float(s); c.hypothesis = h
        ranked = sorted(agg.values(), key=lambda c: -c.prob)
        top = ranked[0] if ranked else None
        action, reason = self._decide(top, none_prob)
        return Decision(action, top, ranked, none_prob, a.free_kana, a.free_score, a.timings, cs.state, reason)

    # ------------------------------------------------------------------
    def _decide(self, top: Candidate | None, none_prob: float) -> tuple[str, str]:
        if top is None:
            return "none", "no candidates"
        h = top.hypothesis
        return decide(top.prob, none_prob, h.risk, self.thresholds,
                      confirmable=getattr(h, "confirmable", True), positive=getattr(h, "positive", True))
