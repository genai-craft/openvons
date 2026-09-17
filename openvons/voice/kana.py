"""カナの正規化・読み生成 (G2P)・モーラ単位の編集距離.

ASR (kana-whisper) の書き方に合わせた「ASR 形」を正規形とする:
  - 全角カタカナ、長音は「ー」(トウキョウ -> トーキョー、センセイ -> センセー)
  - ヲ->オ、ヂ->ジ、ヅ->ズ、ヰ->イ、ヱ->エ
  - 空白・句読点・記号は除去
候補側 (郵便番号データ・G2P・人手入力) をすべてこの形に寄せて比較する。
"""
from __future__ import annotations

import re
from functools import lru_cache

import jaconv
from rapidfuzz.distance import Levenshtein

_A_ROW = set("アカサタナハマヤラワガザダバパァャヮ")
_I_ROW = set("イキシチニヒミリギジヂビピィ")
_U_ROW = set("ウクスツヌフムユルグズヅブプゥュ")
_E_ROW = set("エケセテネヘメレゲゼデベペェ")
_O_ROW = set("オコソトノホモヨロヲゴゾドボポォョ")
_SMALL = set("ァィゥェォャュョヮ")
_PUNCT_RE = re.compile(r"[\s、。,.!?！？「」『』・~〜〔〕（）()\[\]【】\"'’‘\-–—:;/]+")

_REPLACE = str.maketrans({"ヲ": "オ", "ヂ": "ジ", "ヅ": "ズ", "ヰ": "イ", "ヱ": "エ", "ヵ": "カ", "ヶ": "ケ", "ゝ": "", "ゞ": ""})


def to_katakana(s: str) -> str:
    """半角カナ・ひらがな・全角英数を全角カタカナ寄りに寄せる (読み以外は残す)。"""
    s = jaconv.h2z(s, kana=True, ascii=True, digit=True)
    s = jaconv.hira2kata(s)
    return s


def long_vowelize(kana: str) -> str:
    """オ段+ウ -> オ段+ー、エ段+イ -> エ段+ー (pyopenjtalk / kana-whisper の表記に合わせる)。"""
    out: list[str] = []
    for ch in kana:
        if out:
            prev = out[-1]
            if ch == "ウ" and (prev in _O_ROW or prev in _U_ROW):
                out.append("ー"); continue
            if ch == "イ" and prev in _E_ROW:
                out.append("ー"); continue
            if ch == "ー" and prev == "ー":
                continue
    # ↑ 直前が「ー」の連続は 1 つに (エーー のような二重長音を防ぐ)
        out.append(ch)
    return "".join(out)


def normalize(s: str) -> str:
    """任意の読み表記を ASR 形に正規化する。"""
    s = to_katakana(s)
    s = _PUNCT_RE.sub("", s)
    s = s.translate(_REPLACE)
    s = long_vowelize(s)
    # 先頭の長音・促音は発音できないので落とす
    s = s.lstrip("ーッ")
    return s


@lru_cache(maxsize=65536)
def g2p(text: str) -> str:
    """漢字かな交じり文 -> ASR 形カナ。pyopenjtalk (naist-jdic) に依存するので固有名詞は誤読しうる。
    そのため lexicon では人手の読み・郵便番号データの読み・TTS 往復で採れた読みを優先し、これはフォールバック。"""
    import pyopenjtalk  # 遅延 import (辞書の展開が初回に走る)

    k = pyopenjtalk.g2p(text, kana=True)
    return normalize(k)


def variants(kana: str) -> list[str]:
    """ASR の長音表記の揺れに備えた表層バリアント。kana-whisper は「引いて」を ヒーテ とも ヒイテ とも書き、
    「お母さん」は オカーサン/オカアサン の両方がありうる。イ段+イ、ア段+ア を「ー」に置き換えた形を追加する
    (エ段+イ、オ段+ウ は normalize() で既に ー にしている)。返り値は元の形を先頭に含む重複なしのリスト。"""
    out = [kana]
    alt: list[str] = []
    for ch in kana:
        if alt and ((ch == "イ" and alt[-1] in _I_ROW) or (ch == "ア" and alt[-1] in _A_ROW)):
            alt.append("ー")
        else:
            alt.append(ch)
    a = "".join(alt)
    if a != kana:
        out.append(a)
    return out


def mora_split(kana: str) -> list[str]:
    """カナ列をモーラ列に分割する。拗音 (キャ) は 1 モーラ、長音「ー」促音「ッ」撥音「ン」は 1 モーラ。"""
    morae: list[str] = []
    for ch in kana:
        if ch in _SMALL and morae:
            morae[-1] += ch
        else:
            morae.append(ch)
    return morae


def mora_distance(a: str, b: str) -> int:
    return Levenshtein.distance(mora_split(a), mora_split(b))


def mora_similarity(a: str, b: str) -> float:
    """0..1 の正規化類似度 (モーラ Levenshtein)。"""
    return Levenshtein.normalized_similarity(mora_split(a), mora_split(b))


def digits_to_kana(n: int) -> str:
    """1..999 程度の整数を読みに (カメラ番号「1上り」「12下り」用)。pyopenjtalk でも読めるが、
    テンプレート展開時に高速に済ませたいので自前で持つ。"""
    ones = ["", "イチ", "ニ", "サン", "ヨン", "ゴ", "ロク", "ナナ", "ハチ", "キュー"]
    if n < 10:
        return ones[n] if n else "ゼロ"
    if n < 100:
        t, o = divmod(n, 10)
        return ("ジュー" if t == 1 else ones[t] + "ジュー") + ones[o]
    h, r = divmod(n, 100)
    head = {1: "ヒャク", 3: "サンビャク", 6: "ロッピャク", 8: "ハッピャク"}.get(h, ones[h] + "ヒャク")
    return head + (digits_to_kana(r) if r else "")
