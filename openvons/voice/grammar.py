"""意図 (Intent) とテンプレートから、状態ごとの「コマンド集合」をコンパイルする.

テンプレート記法:
    {camera}                スロット。lexicon の kind="camera" の実体すべてに展開される
    [を]                    省略可能な文字列
    [して|の画像]           省略可能な択一 (どれか 1 つ、または無し)
    (表示|出して|見せて)    択一 (必ずどれか 1 つ)
例: "{camera}[を](表示|出して|見せて|お願い)" -> camera 実体 × 4 × 2 = 8 通り/読み

コンパイル結果 CommandSet は仮説 (Hypothesis) の列。仮説 = (意図, スロット充填, 表示文, ASR 形カナ)。
同じ意味 (意図+スロット) の仮説が複数の表層形を持つのは正常で、engine は確率を意味単位で足し合わせる。
"""
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein

from . import kana as K
from .lexicon import Entity

_TOKEN_RE = re.compile(r"\{(\w+)\}|\[([^\]]+)\]|\(([^)]+)\)")


@dataclass
class Intent:
    name: str
    patterns: list[str]
    params: dict[str, Any] = field(default_factory=dict)
    risk: str = "low"                 # low / medium / high
    description: str = ""
    slots: dict[str, str] = field(default_factory=dict)   # スロット名 -> 実体 kind (省略時は同名)
    #: 埋め込み採点 (前後に余計な語が付いても候補を認める) を許すか。「はい/いいえ」のような確認応答は
    #: 単独で言わせる (False)。「そうですね、確認します」の頭の「そうです」を肯定と取らないため
    allow_embed: bool = True
    #: スロット展開で主読みだけを使う (2 スロットの意図は n² に膨らむので別名・「〜駅」読みを落とす)
    primary_only: bool = False
    #: この意図自体を「〜でよろしいですか」と確認できるか。はい / いいえ は False
    #: (確認への返事をさらに確認すると無限に聞き返すことになる)
    confirmable: bool = True
    #: 確認への返事のとき、実行側 (はい) なら True、取り消し側 (いいえ) なら False
    positive: bool = True

    def slot_kind(self, slot: str) -> str:
        return self.slots.get(slot, slot)


@dataclass
class Hypothesis:
    intent: str
    text: str
    kana: str
    slots: dict[str, str] = field(default_factory=dict)   # スロット名 -> 実体 id
    params: dict[str, Any] = field(default_factory=dict)
    risk: str = "low"
    allow_embed: bool = True
    confirmable: bool = True
    positive: bool = True

    @property
    def meaning(self) -> tuple:
        """意味の同一性 = 意図 + スロット充填。"""
        return (self.intent, tuple(sorted(self.slots.items())))


def example_of(pattern: str, slot_labels: dict[str, str] | None = None) -> str:
    """テンプレートから、画面に出す 1 つの例文を作る。

    択一 (a|b) も省略可能 [a|b] も最初のものを採り、{slot} は差し込み文字に置き換える。
    (省略可能を落とすと「一覧戻って」のような読みにくい例文になるので、付けたまま出す)
    画面の「いま言えること」に出すので、記号が残らないようにする。
    """
    labels = slot_labels or {}
    out, pos = [], 0
    for m in _TOKEN_RE.finditer(pattern):
        if m.start() > pos:
            out.append(pattern[pos:m.start()])
        if m.group(1):
            out.append(labels.get(m.group(1), f"<{m.group(1)}>"))
        elif m.group(2):
            out.append(m.group(2).split("|")[0])
        elif m.group(3):
            out.append(m.group(3).split("|")[0])
        pos = m.end()
    if pos < len(pattern):
        out.append(pattern[pos:])
    return "".join(out)


def expand_template(pattern: str) -> list[list[tuple[str, str]]]:
    """テンプレートを [(kind, value)...] の列に展開。kind は 'lit' か 'slot'。省略可能・択一を全部展開する。"""
    parts: list[list[tuple[str, str] | None]] = []   # 各位置の選択肢
    pos = 0
    for m in _TOKEN_RE.finditer(pattern):
        if m.start() > pos:
            parts.append([("lit", pattern[pos:m.start()])])
        if m.group(1):
            parts.append([("slot", m.group(1))])
        elif m.group(2):
            # [a] は「a か、無し」。[a|b] は「a か b か、無し」(| を書けないと [して|の画像] が literal になる)
            parts.append([("lit", alt) for alt in m.group(2).split("|")] + [None])
        else:
            parts.append([("lit", alt) for alt in m.group(3).split("|")])
        pos = m.end()
    if pos < len(pattern):
        parts.append([("lit", pattern[pos:])])
    out: list[list[tuple[str, str]]] = []
    for combo in itertools.product(*parts):
        seq = [c for c in combo if c is not None]
        # 連続する literal は結合 (G2P を文として通すため)
        merged: list[tuple[str, str]] = []
        for kind, val in seq:
            if kind == "lit" and merged and merged[-1][0] == "lit":
                merged[-1] = ("lit", merged[-1][1] + val)
            else:
                merged.append((kind, val))
        out.append(merged)
    return out


#: 数字の読み方。コードは 1 桁ずつ読む (C06 = シーゼロロク)。ゼロは マル / レイ とも言う
_DIGIT_KANA = {
    "0": ["ゼロ", "マル", "レイ"], "1": ["イチ"], "2": ["ニ"], "3": ["サン"], "4": ["ヨン", "シ"],
    "5": ["ゴ"], "6": ["ロク"], "7": ["ナナ", "シチ"], "8": ["ハチ"], "9": ["キュー", "ク"],
}
#: 英字の読み方 (コードの接頭辞に使うぶんだけ)
_LETTER_KANA = {
    "A": ["エー"], "B": ["ビー"], "C": ["シー"], "D": ["ディー"], "E": ["イー"], "F": ["エフ"],
    "G": ["ジー"], "H": ["エイチ"], "K": ["ケー"], "L": ["エル"], "M": ["エム"], "N": ["エヌ"],
    "P": ["ピー"], "R": ["アール"], "S": ["エス"], "T": ["ティー"], "V": ["ブイ"], "W": ["ダブリュー"],
}


def code_readings(code: str, max_forms: int = 6) -> list[str]:
    """「C06」→ [シーゼロロク, シーマルロク, シーレイロク, シーロク]。

    人は同じコードをいろいろに読む (ゼロ / マル / レイ、先頭の 0 を飛ばす) ので、
    ありうる読みを並べて候補にする。読みが 1 つだけだと言い方が違うだけで外れる。
    """
    parts: list[list[str]] = []
    for ch in code.upper():
        if ch.isdigit():
            parts.append(_DIGIT_KANA[ch])
        elif ch in _LETTER_KANA:
            parts.append(_LETTER_KANA[ch])
        else:
            continue
    out: list[str] = []
    for combo in itertools.product(*parts):
        out.append("".join(combo))
        if len(out) >= max_forms:
            break
    m = re.match(r"^([A-Za-z]+)(\d+)$", code)
    if m:
        head, num = m.group(1), m.group(2)
        head_kana = _LETTER_KANA.get(head.upper(), [head])[0]
        # 先頭の 0 を飛ばした言い方 (C06 → シーロク)
        if num.lstrip("0") != num and num.lstrip("0"):
            out.extend(r for r in code_readings(head + num.lstrip("0"), max_forms=2) if r not in out)
        # 数としての言い方 (C10 → シージュー、C37 → シーサンジューナナ)
        whole = K.normalize(K.g2p(str(int(num))))
        if whole and head_kana + whole not in out:
            out.append(head_kana + whole)
    return out


def code_hypotheses(intent: str, slot: str, coded: Iterable[tuple[str, Entity]],
                    patterns: Iterable[str] | None = None, risk: str = "low") -> list[Hypothesis]:
    """「C06」のような整理番号でも選べるようにする仮説。

    固有名を全部覚えて言うのは現場では無理があるので、画面に出ている短いコードで呼べる道を用意する。
    数字だけ (「6 番」) より、英字を頭に付けたコードの方が音として長く、雑談とぶつかりにくい。
    意味 (意図 + スロット) は名前で呼んだときと同じなので、確率は自然に足し合わされる。
    """
    pats = list(patterns or ["{c}", "{c}[を](表示|出して|見せて|お願い)", "{c}[に](切り替え|して)"])
    out: list[Hypothesis] = []
    for code, e in coded:
        for kana_code in code_readings(code):
            for pat in pats:
                for seq in expand_template(pat.replace("{c}", "\x00")):
                    tail = "".join(v for _, v in seq).replace("\x00", "")
                    text = f"{code}{tail}"
                    kana = K.normalize(kana_code + (K.g2p(tail) if tail else ""))
                    if kana:
                        out.append(Hypothesis(intent, f"{text} ({e.label})", kana, {slot: e.id}, {}, risk, True, True, True))
    return out


def number_hypotheses(intent: str, slot: str, entities: Iterable[Entity], patterns: Iterable[str] | None = None,
                      risk: str = "low", start: int = 1) -> list[Hypothesis]:
    """一覧の並び順に番号を振り、「12 番」でも選べるようにする仮説を作る。

    固有名を全部覚えて言うのは現場では無理があるので、画面に出ている番号で呼べる道を用意する。
    意味 (意図 + スロット) は名前で呼んだときと同じなので、確率は自然に足し合わされる。
    番号は範囲 (スコープ) ごとの並び順に依存するため、語彙ではなくコマンド集合を作るときに付ける。
    """
    pats = list(patterns or ["{n}番", "{n}番[を](表示|出して|見せて|お願い)", "番号{n}", "{n}番[に](切り替え|して)"])
    out: list[Hypothesis] = []
    for i, e in enumerate(entities, start):
        for pat in pats:
            surface = pat.replace("{n}", str(i))
            for seq in expand_template(surface):
                text = "".join(v for _, v in seq)
                kana = K.normalize(K.g2p(text))
                if not kana:
                    continue
                out.append(Hypothesis(intent, f"{text} ({e.label})", kana, {slot: e.id}, {}, risk, True, True, True))
    return out


class CommandSet:
    def __init__(self, hyps: list[Hypothesis], state: str = ""):
        self.state = state
        self.hyps = hyps
        self.kanas = [h.kana for h in hyps]
        self._meaning_index: dict[tuple, list[int]] = {}
        for i, h in enumerate(hyps):
            self._meaning_index.setdefault(h.meaning, []).append(i)
        # 意味ごとに短い方から 3 表層形 (素の名前・別名・最短の担体付き)。部分一致の絞り込みはこれだけを見る
        # (13 万仮説 → 1 万弱に縮む)。最短 1 つだけだと別名 (「谷津2」= ヤツニ) が選ばれて本名が漏れる
        self.bare_index: list[int] = []
        for ix in self._meaning_index.values():
            self.bare_index.extend(sorted(ix, key=lambda i: len(hyps[i].kana))[:3])
        self._bare_kanas = [hyps[i].kana for i in self.bare_index]

    def __len__(self) -> int:
        return len(self.hyps)

    @property
    def n_meanings(self) -> int:
        return len(self._meaning_index)

    def shortlist(self, query_kana: str, k: int = 16) -> list[int]:
        """自由認識のカナに近い仮説の index を k 個。
        (a) 文字 Levenshtein の正規化類似度 (全文が近いもの) と (b) 部分一致 partial_ratio (前置き・後置きに
        包まれた短い仮説: 「えー、そうゆう、谷津2下り、確認します」の中の「ヤツニクダリ」) の上位を合わせる。
        (b) は埋め込み採点の入口。C++ 実装で 13 万仮説でも 20ms 級。
        同じ意味の仮説が上位を独占しないよう、意味ごとに最良の 2 表層形までに制限する。"""
        if len(self.hyps) <= k:
            return list(range(len(self.hyps)))
        full = process.extract(query_kana, self.kanas, scorer=Levenshtein.normalized_similarity, limit=k * 3)
        part = process.extract(query_kana, self._bare_kanas, scorer=fuzz.partial_ratio, limit=k * 2, score_cutoff=80)
        ranked: list[tuple[float, int]] = [(sc, idx) for _, sc, idx in full]
        # 部分一致は候補が短すぎる (2 モーラ) と何にでも当たるので、query の 30% 以上の長さを要求する
        min_len = max(3, int(0.3 * len(query_kana)))
        ranked += [(0.9 * sc / 100.0, self.bare_index[j]) for _, sc, j in part if len(self._bare_kanas[j]) >= min_len]
        ranked.sort(key=lambda t: -t[0])
        out: list[int] = []
        seen: set[int] = set()
        per_meaning: dict[tuple, int] = {}
        for _, idx in ranked:
            if idx in seen:
                continue
            m = self.hyps[idx].meaning
            if per_meaning.get(m, 0) >= 2:
                continue
            per_meaning[m] = per_meaning.get(m, 0) + 1
            seen.add(idx)
            out.append(idx)
            if len(out) >= k:
                break
        return out

    def intents(self) -> list[str]:
        seen: list[str] = []
        for h in self.hyps:
            if h.intent not in seen:
                seen.append(h.intent)
        return seen


class Grammar:
    def __init__(self, intents: Iterable[Intent]):
        self.intents: dict[str, Intent] = {i.name: i for i in intents}

    def compile(self, intent_names: Iterable[str], entities: Iterable[Entity], state: str = "",
                extra_patterns: dict[str, list[str]] | None = None) -> CommandSet:
        by_kind: dict[str, list[Entity]] = {}
        for e in entities:
            by_kind.setdefault(e.kind, []).append(e)
        # 範囲内で複数の実体に当たる読み (別名の「船橋南」= 上り/下り両方、「幕張2」= 上り/下り) は
        # 単独では選べないので落とす。埋め込み採点と組むと同点の実体が並んで確率が割れる (v6 の退行の原因)。
        # 主読みが衝突する場合 (同名カメラ) はそのまま残し、確率が割れて確認に回るのが正しい挙動。
        self._readings_in_scope: dict[str, dict[str, list[str]]] = {}
        for kind, ents in by_kind.items():
            owner: dict[str, set[str]] = {}
            for e in ents:
                for r in e.all_readings():
                    owner.setdefault(r, set()).add(e.id)
            table: dict[str, list[str]] = {}
            for e in ents:
                rs = list(e.primary_readings()) + [r for r in e.alias_readings() if len(owner.get(r, ())) == 1]
                table[e.id] = rs
            self._readings_in_scope[kind] = table
        hyps: list[Hypothesis] = []
        seen: set[tuple[str, str, tuple]] = set()
        for name in intent_names:
            it = self.intents[name]
            patterns = list(it.patterns) + list((extra_patterns or {}).get(name, []))
            for pat in patterns:
                for seq in expand_template(pat):
                    for h in self._fill(it, seq, by_kind):
                        key = (h.intent, h.kana, tuple(sorted(h.slots.items())))
                        if key in seen or not h.kana:
                            continue
                        seen.add(key)
                        hyps.append(h)
        return CommandSet(hyps, state)

    def _fill(self, it: Intent, seq: list[tuple[str, str]], by_kind: dict[str, list[Entity]]) -> Iterable[Hypothesis]:
        slot_positions = [i for i, (k, _) in enumerate(seq) if k == "slot"]
        lit_kana = {i: K.g2p(v) for i, (k, v) in enumerate(seq) if k == "lit"}
        if not slot_positions:
            text = "".join(v for _, v in seq)
            for kv in K.variants("".join(lit_kana[i] for i in range(len(seq)))):
                yield Hypothesis(it.name, text, kv, {}, dict(it.params), it.risk, it.allow_embed,
                                 it.confirmable, it.positive)
            return
        # スロットごとの (entity, reading) の候補
        options: list[list[tuple[str, Entity, str]]] = []
        for i in slot_positions:
            slot = seq[i][1]
            kind = it.slot_kind(slot)
            ents = by_kind.get(kind, [])
            table = self._readings_in_scope.get(kind, {})
            if it.primary_only:
                opts = [(slot, e, e.primary_readings()[0]) for e in ents]
            else:
                opts = [(slot, e, r) for e in ents for r in table.get(e.id, e.all_readings())]
            options.append(opts)
        for combo in itertools.product(*options):
            fill = {slot: e for slot, e, _ in combo}
            reading = {slot: r for slot, _, r in combo}
            text_parts: list[str] = []
            kana_parts: list[str] = []
            for i, (k, v) in enumerate(seq):
                if k == "lit":
                    text_parts.append(v); kana_parts.append(lit_kana[i])
                else:
                    text_parts.append(fill[v].label); kana_parts.append(reading[v])
            for kv in K.variants(K.normalize("".join(kana_parts))):
                yield Hypothesis(it.name, "".join(text_parts), kv, {s: e.id for s, e in fill.items()},
                                 dict(it.params), it.risk, it.allow_embed, it.confirmable, it.positive)
