"""どのアプリでも使い回す意図。いまのところ「地図を動かす」。

方向 8 つ × 量 3 段を 1 つずつ書くと 24 個になるので、組み合わせで作る。
params の dir (向き) と amount (画面の何割動かすか) を画面側がそのまま地図に渡す。
"""
from __future__ import annotations

from .grammar import Intent

#: (キー, 表記, 言い換え)
PAN_DIRS = [
    ("up", "上", ["北"]), ("down", "下", ["南"]), ("left", "左", ["西"]), ("right", "右", ["東"]),
    ("upleft", "左上", []), ("upright", "右上", []), ("downleft", "左下", []), ("downright", "右下", []),
]
#: (キー, 言い方, 画面の何割)
PAN_MAGS = [("small", ["ちょっと", "少し"], 0.25), ("normal", [""], 0.6), ("large", ["大きく", "ぐっと", "もっと"], 1.2)]

PAN_INTENT_NAMES = [f"pan_{d}_{m}" for d, _, _ in PAN_DIRS for m, _, _ in PAN_MAGS]


def pan_intents() -> list[Intent]:
    out: list[Intent] = []
    for dkey, dword, alts in PAN_DIRS:
        words = [dword, *alts]
        for mkey, mwords, amount in PAN_MAGS:
            pats: list[str] = []
            for mag in mwords:
                for w in words:
                    if mag:
                        # 「ちょっと右」だけでも通す (5 モーラあるので雑音には強い)
                        pats.append(f"{mag}{w}[に|へ][動かして|ずらして|移動して|寄せて]")
                    else:
                        # 量を言わないときは動詞を必須にする (「右」単独は短すぎて雑音に弱い)
                        pats.append(f"{w}(に|へ)(動かして|ずらして|移動して|寄せて)")
                        pats.append(f"{w}(の方|のほう)[に|へ][動かして|ずらして]")
            out.append(Intent(f"pan_{dkey}_{mkey}", pats, params={"dir": dkey, "amount": amount},
                              description=f"地図を{mwords[0]}{dword}へ"))
    return out
