"""TTS を使った事前学習 (担当範囲の登録時に走らせる).

やること:
  1. 範囲内の実体ごとに、名前だけ・テンプレート付きの発話を複数の声で合成する (Irodori TTS)
  2. それを認識器に通し、(a) 正解率と混同対、(b) ASR が実際に書く読み (実現読み)、(c) 校正 (T, beta) を得る
  3. 実現読みが既存の読みと違えば読みとして登録する (「谷津」を G2P が「タニツ」と誤読しても、
     TTS が「ヤツ」と読み ASR が「ヤツ」と書くならそれが登録される。逆に TTS も誤読すれば
     人手修正が必要と分かる = 混同・誤読の可視化)
  4. 文法外の発話 (電話の応対など) も合成し「該当なし」に倒れることを確認、beta を合わせる

雑音付加: ホワイト/ピンクノイズ (SNR 指定)、別発話の重ね合わせ (隣の席の声)、ゲイン変動。
"""
from __future__ import annotations

import io
import json
import os
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import httpx
import numpy as np
import soundfile as sf

from . import kana as K
from jev.core.none_calibration import Calibration, ece, fit
from .engine import Recognizer
from .grammar import CommandSet, Grammar
from .lexicon import Entity, Lexicon, Scope

SR = 16000

#: 文法外の発話 (電話応対・独り言)。「該当なし」の学習に使う
OUT_OF_GRAMMAR = [
    "はい、お世話になっております", "少々お待ちください", "ええ、そうですね、確認します",
    "それでは午後の会議でお願いします", "今ちょっと手が離せないので後でかけ直します",
    "了解しました、確認して折り返します", "資料は共有フォルダに入れておきました",
    "おつかれさまです、渋滞の件ですが", "雨が強くなってきましたね", "課長、こちらの画面ご覧ください",
    "はいはい、大丈夫です", "ちょっと待ってね", "うーん、どうしようかな", "お先に失礼します",
    "この後の点検はいつになりますか", "通行止めの解除は十五時の予定です",
]


from jev.tts import to16k  # noqa: E402,F401


from jev.tts import TTSBackend, get_backend  # noqa: E402


def TTSClient(url: str, cache_dir=None, **_) -> TTSBackend:   # 後方互換の名前
    return get_backend(url, cache_dir=cache_dir)


# ---------------------------------------------------------------- 雑音付加
def add_noise(wav: np.ndarray, snr_db: float, rng: random.Random, pink: bool = True) -> np.ndarray:
    n = np.random.default_rng(rng.randrange(1 << 30)).standard_normal(len(wav)).astype(np.float32)
    if pink:
        # 1/f 近似: 累積和をハイパスした簡易ピンク
        n = np.cumsum(n); n = n - np.convolve(n, np.ones(64) / 64, mode="same")
    n = n / (np.sqrt((n ** 2).mean()) + 1e-8)
    p_sig = np.sqrt((wav ** 2).mean()) + 1e-8
    n = n * p_sig / (10 ** (snr_db / 20))
    return np.clip(wav + n, -1, 1)


def _active_rms(x: np.ndarray) -> float:
    """有声部 (ピークの 5% 以上) だけの RMS。無音を含む切片を全体 RMS で正規化すると、無音が多いほど
    有声部が過大に増幅される (事前学習で背景の「日吉倉」が前景を覆った事故の原因)。"""
    a = np.abs(x)
    m = a > 0.05 * (a.max() + 1e-8)
    if m.sum() < 160:
        return float(np.sqrt((x ** 2).mean()) + 1e-8)
    return float(np.sqrt((x[m] ** 2).mean()) + 1e-8)


def mix_background(wav: np.ndarray, other: np.ndarray, level_db: float, rng: random.Random) -> np.ndarray:
    """別の発話を level_db (負値、有声部 RMS 比) で重ねる。隣の席・電話相手の声の模擬。"""
    if len(other) < len(wav):
        other = np.tile(other, int(np.ceil(len(wav) / len(other))))
    start = rng.randrange(0, len(other) - len(wav) + 1)
    o = other[start:start + len(wav)]
    gain = _active_rms(wav) / _active_rms(o) * 10 ** (level_db / 20)
    return np.clip(wav + o * gain, -1, 1)


def pad_silence(wav: np.ndarray, rng: random.Random, max_sec: float = 0.4) -> np.ndarray:
    a = int(rng.uniform(0.05, max_sec) * SR); b = int(rng.uniform(0.05, max_sec) * SR)
    return np.concatenate([np.zeros(a, np.float32), wav, np.zeros(b, np.float32)])


# ---------------------------------------------------------------- TTS 入力の作り方
def tts_text(entity: Entity, carrier: str, slot: str, use_reading: bool = True) -> str:
    """TTS に渡す文。既定では実体の読み (カナ) を埋め込む。
    表示名 (漢字+数字) をそのまま渡すと TTS 側の G2P が「2下り」を「ニグダリ」、「谷津」を「タニズ」と
    誤読し、ASR ではなく TTS の誤りを測ってしまう。読みカナなら TTS はほぼ正確に読むので、
    往復で残る差分は ASR 側の書き方 (実現読み) だけになる。"""
    name = entity.all_readings()[0] if use_reading else entity.label
    return carrier.replace("{" + slot + "}", name)


@dataclass
class PretrainConfig:
    seeds: list[int] = field(default_factory=lambda: [1, 2, 3])
    carriers: list[str] = field(default_factory=lambda: ["{camera}", "{camera}を表示", "{camera}出して"])
    snr_db: list[float | None] = field(default_factory=lambda: [None, 20.0, 10.0])
    background_level_db: float | None = -12.0     # None で無効
    n_out_of_grammar: int = 12
    add_readings: bool = True
    min_utts: int = 320                         # 校正サンプルの下限 (足りなければ声・言い方を自動で増やす)
    reading_max_mora_distance: int = 4          # 実現読みを追加する上限 (これ以上離れていたら別物として警告)
    check_label_reading: bool = True            # 表示名 (漢字) でも合成し、登録読みと TTS の読みが食い違う実体を警告する
    #: 他の状態の校正サンプル: (状態名, 受理する意図, [(発話, 正解意図 or None=該当なし)])。アプリが自分の文法に合わせて渡す
    extra_states: list[tuple[str, list[str], list[tuple[str, str | None]]]] = field(default_factory=list)


@dataclass
class PretrainResult:
    n_utts: int = 0
    accuracy: float = 0.0                        # 校正前 (実行時の校正値) の判断での正解率
    accuracy_after: float = 0.0                  # 校正後に argmax をやり直した正解率 (該当なし含む)
    rejected_correct_top: int = 0                # 最尤候補は正しいのに該当なし/棄却になった数
    accuracy_by_snr: dict[str, float] = field(default_factory=dict)
    none_recall: float = 0.0                     # 文法外発話が該当なしに倒れた率
    false_accept: float = 0.0                    # 文法外発話を実行してしまった率
    ece_before: float = 0.0
    ece_after: float = 0.0
    calibration: dict[str, float] = field(default_factory=dict)
    confusions: list[dict[str, Any]] = field(default_factory=list)      # 混同対 (上位)
    added_readings: list[dict[str, str]] = field(default_factory=list)   # 追加した実現読み
    warnings: list[str] = field(default_factory=list)
    latency_ms: dict[str, float] = field(default_factory=dict)
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)


class Pretrainer:
    def _add_sample(self, w: np.ndarray, cs: CommandSet, gold_intent: str | None, samples: list, probs_before: list, labels: list) -> None:
        """1 発話を校正サンプルにする (実行時と同じ analyze 経路)。gold_intent が None なら該当なしが正解。"""
        a = self.rec.analyze(w, cs)
        if not a.cands:
            return
        pred = (lambda h: h.intent == gold_intent) if gold_intent is not None else (lambda h: False)
        samples.append(a.sample(pred))
        probs_before.append(self.rec.calibration.probs(a.scores, a.free_score, a.lens, a.n_free)); labels.append(a.gold_index(pred))

    def __init__(self, tts: TTSBackend, recognizer: Recognizer, grammar: Grammar, lexicon: Lexicon):
        self.tts = tts
        self.rec = recognizer
        self.grammar = grammar
        self.lexicon = lexicon

    def run(self, scope: Scope, intent_names: list[str], slot: str = "camera", cfg: PretrainConfig | None = None,
            progress: Callable[[dict[str, Any]], None] | None = None, seed: int = 0, entities: list[Entity] | None = None) -> PretrainResult:
        cfg = cfg or PretrainConfig()
        rng = random.Random(seed)
        t_start = time.time()
        ents = entities if entities is not None else self.lexicon.in_scope(scope)
        res = PretrainResult()
        if not ents:
            res.warnings.append("範囲に実体がありません")
            return res
        # 範囲が小さいと校正サンプルが足りず fit が退化する (山手線 30 駅 × 2 言い方 × 3 声 = 180 発話で β0 が 17 に飛んだ)。
        # 最低 cfg.min_utts 発話になるまで声 (seed) と言い方を増やす
        extra_carriers = ["{%s}を表示" % slot, "{%s}に行きたい" % slot, "{%s}お願い" % slot, "{%s}出して" % slot]
        seeds, carriers = list(cfg.seeds), list(cfg.carriers)
        while len(ents) * len(seeds) * len(carriers) < cfg.min_utts and (len(seeds) < 6 or len(carriers) < 6):
            if len(seeds) < 6:
                seeds.append(max(seeds) + 1)
            elif extra_carriers:
                c = extra_carriers.pop(0)
                if c not in carriers:
                    carriers.append(c)
            else:
                break
        if (seeds, carriers) != (list(cfg.seeds), list(cfg.carriers)):
            res.warnings.append(f"範囲が小さいため声 {len(seeds)} 種・言い方 {len(carriers)} 種に増やしました ({len(ents) * len(seeds) * len(carriers)} 発話)")
        cfg = PretrainConfig(**{**cfg.__dict__, "seeds": seeds, "carriers": carriers})
        rep = lambda **kw: progress(kw) if progress else None

        # --- 1. 名前だけを合成し、実現読みを採る (複数の声で一致したものだけ登録)
        realized: dict[str, list[str]] = {}
        total_steps = len(ents) * len(cfg.seeds) * len(cfg.carriers) + cfg.n_out_of_grammar
        step = 0
        cs_full = self.grammar.compile(intent_names, ents, "PRETRAIN")
        samples: list[tuple[np.ndarray, float, int]] = []     # 校正用 (候補スコア, 自由スコア, 正解 index)
        probs_before: list[np.ndarray] = []
        labels: list[int] = []
        hits = 0; n = 0; n_rejected = 0
        by_snr: dict[str, list[int]] = {}
        confusion: dict[tuple[str, str], int] = {}
        lat: dict[str, list[float]] = {}
        bg_pool: list[np.ndarray] = []
        for e in ents:
            for carrier in cfg.carriers:
                text = tts_text(e, carrier, slot)
                for sd in cfg.seeds:
                    step += 1
                    try:
                        wav = self.tts.synth(text, seed=sd)
                    except Exception as ex:   # TTS 落ち等は記録して続行
                        res.warnings.append(f"TTS 失敗: {text} ({ex})")
                        continue
                    if len(bg_pool) < 8:
                        bg_pool.append(wav)
                    snr = rng.choice(cfg.snr_db)
                    w = pad_silence(wav, rng)
                    if snr is not None:
                        w = add_noise(w, snr, rng)
                    if cfg.background_level_db is not None and len(bg_pool) > 2 and rng.random() < 0.3:
                        w = mix_background(w, rng.choice(bg_pool), cfg.background_level_db, rng)
                    a = self.rec.analyze(w, cs_full)
                    d = self.rec.recognize(w, cs_full)      # 判断 (校正前の実行時挙動) — analyze を 2 回呼ぶが GPU コストは小さい
                    for k, v in d.timings.items():
                        lat.setdefault(k, []).append(v)
                    if carrier == "{" + slot + "}":
                        realized.setdefault(e.id, []).append(d.free_kana)
                    if a.cands:
                        y = a.gold_index(lambda h: h.slots.get(slot) == e.id)
                        if y == len(a.cands):
                            res.warnings.append(f"絞り込み漏れ: {e.label} <- {d.free_kana}")
                        samples.append(a.sample(lambda h: h.slots.get(slot) == e.id))
                        probs_before.append(self.rec.calibration.probs(a.scores, a.free_score, a.lens, a.n_free)); labels.append(y)
                    n += 1
                    ok = d.top is not None and d.top.hypothesis.slots.get(slot) == e.id and d.action != "none"
                    hits += int(ok)
                    key = "clean" if snr is None else f"snr{int(snr)}"
                    by_snr.setdefault(key, []).append(int(ok))
                    if not ok and d.top is not None:
                        other = d.top.hypothesis.slots.get(slot, d.top.hypothesis.intent)
                        if other != e.id:
                            confusion[(e.id, other)] = confusion.get((e.id, other), 0) + 1
                        else:
                            n_rejected += 1          # 候補は正しいが該当なし/棄却になった (校正で直る類)
                    rep(step=step, total=total_steps, label=e.label, text=text, ok=ok, free=d.free_kana, acc=hits / max(n, 1))

        # --- 2. 文法外発話
        none_ok = 0; false_acc = 0; n_oog = 0
        for i, text in enumerate(rng.sample(OUT_OF_GRAMMAR, min(cfg.n_out_of_grammar, len(OUT_OF_GRAMMAR)))):
            step += 1
            try:
                wav = self.tts.synth(text, seed=cfg.seeds[i % len(cfg.seeds)])
            except Exception as ex:
                res.warnings.append(f"TTS 失敗: {text} ({ex})"); continue
            w = pad_silence(wav, rng)
            d = self.rec.recognize(w, cs_full)
            self._add_sample(w, cs_full, None, samples, probs_before, labels)
            n_oog += 1
            none_ok += int(d.action in ("none", "reject"))
            false_acc += int(d.action == "execute")
            rep(step=step, total=total_steps, label="(文法外)", text=text, ok=d.action in ("none", "reject"), free=d.free_kana, acc=hits / max(n, 1))

        # --- 2.5 他の状態の校正サンプル (短い命令・はい/いいえ・それらに対する文法外)。
        # カメラ名 (長い) だけで校正すると短い候補に緩くなり、確認待ちで電話の「はい」を拾う。
        if cfg.extra_states:
            for st_name, intents, phrases in cfg.extra_states:
                cs_st = self.grammar.compile(intents, ents, st_name)
                if len(cs_st) == 0:
                    continue
                for text, gold_intent in phrases:
                    step += 1
                    try:
                        wav = self.tts.synth(text, seed=cfg.seeds[step % len(cfg.seeds)])
                    except Exception as ex:
                        res.warnings.append(f"TTS 失敗: {text} ({ex})"); continue
                    w = pad_silence(wav, rng)
                    if rng.random() < 0.5:
                        w = add_noise(w, rng.choice([20.0, 10.0]), rng)
                    self._add_sample(w, cs_st, gold_intent, samples, probs_before, labels)
                    rep(step=step, total=total_steps, label=f"({st_name})", text=text, ok=True, free="", acc=hits / max(n, 1))

        # --- 3. 校正
        cal = fit(samples, self.rec.calibration)
        res.calibration = cal.to_dict()
        # ECE は可変長の分布なので「正解に置いた確率 vs 正解/不正解」で top-label ECE を計算
        def top_ece(cal_: Calibration) -> float:
            confs = []; corr = []
            for cs_, fs_, y, lens_, nf_ in samples:
                p = cal_.probs(cs_, fs_, lens_, nf_)
                confs.append(p.max()); corr.append(int(p.argmax() == y))
            return _ece_top(np.array(confs), np.array(corr))
        res.ece_before = top_ece(self.rec.calibration)
        res.ece_after = top_ece(cal)
        # 校正後に判断をやり直したときの精度 (該当なしも 1 クラス)。UI で「事前学習後の精度」として見せる値
        res.accuracy_after = float(np.mean([cal.probs(cs_, fs_, lens_, nf_).argmax() == y for cs_, fs_, y, lens_, nf_ in samples])) if samples else 0.0
        res.rejected_correct_top = n_rejected

        # --- 3.5 表示名 (漢字) の読み照合: TTS の G2P と登録読みが食い違えば人手確認を促す
        if cfg.check_label_reading:
            for e in ents:
                try:
                    wav = self.tts.synth(e.label, seed=cfg.seeds[0])
                except Exception:
                    continue
                d = self.rec.recognize(pad_silence(wav, rng), cs_full)
                best = max(K.mora_similarity(d.free_kana, r) for r in e.all_readings())
                if best < 0.6:
                    res.warnings.append(f"表示名の読み要確認: {e.label} 登録 {e.all_readings()[0]} / TTS→ASR {d.free_kana}")
                rep(step=step, total=total_steps, label=e.label, text=e.label, ok=best >= 0.6, free=d.free_kana, acc=hits / max(n, 1))

        # --- 4. 実現読みの登録
        if cfg.add_readings:
            for eid, kanas in realized.items():
                e = self.lexicon.get(eid)
                if e is None:
                    continue
                counts: dict[str, int] = {}
                for k in kanas:
                    counts[k] = counts.get(k, 0) + 1
                for k, c in counts.items():
                    if c < 2 or not k:
                        continue
                    existing = e.all_readings()
                    dist = min(K.mora_distance(k, r) for r in existing)
                    if dist == 0:
                        continue
                    if dist <= cfg.reading_max_mora_distance:
                        if self.lexicon.add_reading(eid, k):
                            res.added_readings.append({"id": eid, "label": e.label, "reading": k, "from": existing[0]})
                    else:
                        res.warnings.append(f"読みが大きく違う: {e.label} 想定 {existing[0]} / ASR {k} (距離 {dist}) — 読みの人手確認を推奨")

        # --- 5. 集計
        res.n_utts = n
        res.accuracy = hits / max(n, 1)
        res.accuracy_by_snr = {k: sum(v) / len(v) for k, v in by_snr.items()}
        res.none_recall = none_ok / max(n_oog, 1)
        res.false_accept = false_acc / max(n_oog, 1)
        top_conf = sorted(confusion.items(), key=lambda kv: -kv[1])[:10]
        for (a, b), c in top_conf:
            ea = self.lexicon.get(a); eb = self.lexicon.get(b)
            res.confusions.append({"from": ea.label if ea else a, "to": eb.label if eb else b, "count": c})
        res.latency_ms = {k: float(np.median(v)) for k, v in lat.items()}
        res.seconds = time.time() - t_start
        return res


def _ece_top(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)
