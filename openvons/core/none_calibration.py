"""該当なし付き校正 (音声など、候補の尤度 + 自由仮説の尤度から分布を作る場面で使う)。

engine は候補の対数尤度 s_i と自由認識の対数尤度 s_free から
    logits = [s_1/T, ..., s_K/T, (s_free - beta)/T]
の softmax を「候補 + 該当なし」の分布とする。T と beta を検証データ (事前学習の合成音声) で
負の対数尤度最小化により求める。typesafe の Decision Model と同じ発想:
「選択肢外」を明示的な選択肢にすると、劣化時に誤答でなく棄却へ倒れる。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass
class Calibration:
    """候補 i の logit:  s_i + β1·min(n_i, n_free) − γ·max(0, n_free − n_i)
       該当なしの logit:  s_free − β0
       すべて温度 T で割って softmax。

    - s_i: 教師強制の log p(候補 (または埋め込み文) | 音声)、n_i: その採点列のトークン数、
      s_free / n_free: 自由認識 (greedy) の対数尤度とトークン数
    - β0 (none_bias): 自由認識との差の許容量の定数部
    - β1 (len_bonus): 候補が説明したトークン数に比例する加点 (ASR の挿入ボーナス)。雑音で崩れた自由認識に
      対して、長い候補は 1 トークンあたり 1〜2 nats ずつ負けるので、長さに比例した許容量が要る。
    - γ (residual_penalty): 候補が説明していない自由認識の残りトークンへの罰則。kana-whisper は途中で EOT を
      出すことに大きな罰を与えない (「ハイ」+EOT が「ハイオセワニナッテオリマス」の音声に対して −5 nats 程度)
      ので、尤度だけでは「はい」が電話の「はい、お世話になっております」に勝ってしまう。埋め込み採点で前後の
      語を取り込んだ候補は残りが 0 になり罰則を受けない。
    既定値は合成音声 (全状態: カメラ名・PTZ・はい/いいえ・文法外) での校正結果に合わせた事前分布で、
    事前学習で範囲ごとに上書きされる。
    """
    temperature: float = 2.5
    none_bias: float = 4.0
    len_bonus: float = 1.4
    residual_penalty: float = 1.0
    # 2026-09-17 首都国道 100 台の事前学習 (カメラ名 400 + 他状態 31 + 文法外 12 発話) の fit:
    # T 2.88 / β0 3.95 / β1 1.32 / γ 0.82 を丸めたもの

    def logits(self, cand_scores: np.ndarray, free_score: float, cand_lens: np.ndarray | None = None, n_free: int | None = None) -> np.ndarray:
        cs = np.asarray(cand_scores, dtype=np.float64).copy()
        if cand_lens is not None:
            n = np.asarray(cand_lens, dtype=np.float64)
            if n_free is None:
                cs += self.len_bonus * n
            else:
                cs += self.len_bonus * np.minimum(n, float(n_free)) - self.residual_penalty * np.maximum(0.0, float(n_free) - n)
        z = np.concatenate([cs, [free_score - self.none_bias]])
        return z / max(self.temperature, 1e-3)

    def probs(self, cand_scores: np.ndarray, free_score: float, cand_lens: np.ndarray | None = None, n_free: int | None = None) -> np.ndarray:
        z = self.logits(cand_scores, free_score, cand_lens, n_free)
        z = z - z.max()
        p = np.exp(z)
        return p / p.sum()

    def to_dict(self) -> dict:
        return {"temperature": self.temperature, "none_bias": self.none_bias, "len_bonus": self.len_bonus, "residual_penalty": self.residual_penalty}

    @classmethod
    def from_dict(cls, d: dict | None) -> "Calibration":
        if not d:
            return cls()
        c = cls()
        return cls(float(d.get("temperature", c.temperature)), float(d.get("none_bias", c.none_bias)),
                   float(d.get("len_bonus", c.len_bonus)), float(d.get("residual_penalty", c.residual_penalty)))


Sample = tuple  # (cand_scores, free_score, y, cand_lens, n_free)


def fit(samples: list[tuple], init: Calibration | None = None, fit_len_bonus: bool = True) -> Calibration:
    """samples: (候補スコア, 自由スコア, 正解 index[, 候補トークン数[, 自由トークン数]])。該当なし = len(候補)。
    NLL を T, β0, β1, γ で最小化。初期値へ弱く正則化 (少数サンプルの暴走防止)。"""
    init = init or Calibration()
    S = []
    for smp in samples:
        cs, fs, y = np.asarray(smp[0], dtype=np.float64), float(smp[1]), int(smp[2])
        lens = np.asarray(smp[3], dtype=np.float64) if len(smp) > 3 and smp[3] is not None else None
        nf = int(smp[4]) if len(smp) > 4 and smp[4] is not None else None
        S.append((cs, fs, y, lens, nf))

    # 正例 (文法内) と負例 (該当なし) を 1:1 に重み付け。負例が少ないと β0 が正例に引かれて伸び、
    # 短い命令の状態で電話の「はい」を拾うようになる (山手線 30 駅で β0 が 24 に飛んだ)
    n_neg = sum(1 for cs, _, y, _, _ in S if y == len(cs)); n_pos = len(S) - n_neg
    w_neg = (n_pos / n_neg) if n_neg and n_pos else 1.0

    def nll(x):
        T = np.exp(x[0]); b0 = x[1]; b1 = x[2] if fit_len_bonus else init.len_bonus; g = x[3] if fit_len_bonus else init.residual_penalty
        cal = Calibration(T, b0, b1, g)
        tot = 0.0
        for cs, fs, y, lens, nf in S:
            z = cal.logits(cs, fs, lens, nf)
            z = z - z.max()
            tot += (w_neg if y == len(cs) else 1.0) * (np.log(np.exp(z).sum()) - z[y])
        # 正則化はサンプルが少ないほど強く (200 件で 0.02、50 件で 0.08)。少数サンプルでの退化 (β0 が 17 に飛ぶ等) を防ぐ
        lam = 0.02 * max(1.0, 200.0 / max(len(S), 1))
        tot += lam * ((x[0] - np.log(init.temperature)) ** 2 + 0.1 * (b0 - init.none_bias) ** 2 + (b1 - init.len_bonus) ** 2 + (g - init.residual_penalty) ** 2)
        return tot / max(len(S), 1)

    res = minimize(nll, x0=[np.log(init.temperature), init.none_bias, init.len_bonus, init.residual_penalty], method="Nelder-Mead",
                   options={"xatol": 1e-3, "fatol": 1e-7, "maxiter": 6000})
    # 事前分布の範囲 (合成音声での校正の経験値)。外れる値は退化なので切る
    T = float(np.clip(np.exp(res.x[0]), 0.5, 6.0))
    b0 = float(np.clip(res.x[1], -2.0, 12.0))
    b1 = float(np.clip(res.x[2], 0.5, 4.0)) if fit_len_bonus else init.len_bonus
    g = float(np.clip(res.x[3], 0.5, 4.0)) if fit_len_bonus else init.residual_penalty
    return Calibration(T, b0, b1, g)


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    probs = np.asarray(probs); labels = np.asarray(labels)
    conf = probs.max(1); pred = probs.argmax(1)
    correct = (pred == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)
