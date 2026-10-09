"""kana 入力エンジンの一覧と読み込み.

どのデモも同じ一覧から選ぶ。サーバー側は共有 ASR サーバー (openvons.voice.asr_server、既定 :8630) に 1 つずつ載せ、
各デモはそこへ HTTP で問い合わせる (RemoteASR)。kana-whisper (809M、GPU) をデモの数だけ読まなくて済む。

  engine id      中身                                        大きさ    どこで
  kana-whisper   sbintuitions/kana-whisper (whisper 系、MIT)   809M     サーバー GPU
  komimi-v12     komimi ja_v12  (Conformer-CTC 16 層 d512)    100M     サーバー CPU / ブラウザ (106 MB)
  komimi-v12a    komimi ja_v12a (16 層 d256)                   30M     サーバー CPU / ブラウザ (28 MB)
  komimi-v12m    komimi ja_v12m (16 層 d176、P4 向け)           13M     サーバー CPU / ブラウザ (13 MB)
  komimi-v12s    komimi ja_v12s (8 層、S3 向け)                6.5M     サーバー CPU / ブラウザ (7 MB)

komimi は CTC なので、候補の採点 (振り分け) を音声側の行列 1 枚の上の動的計画法だけで済ませられる。
ブラウザでは komimi の WebAssembly と振り分けの WebAssembly (openvons/voice/wasm) を同じモジュールで動かす。
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass

DEFAULT_ENGINE = os.environ.get("OPENVONS_KANA_ENGINE", "kana-whisper")


@dataclass
class EngineSpec:
    id: str
    label: str
    kind: str              # "whisper" (decoder で教師強制採点) | "ctc" (CTC forward 採点)
    params: str
    file: str = ""         # komimi の .kmm (KOMIMI_HOME/models)
    browser: bool = False  # ブラウザ (WebAssembly) で動かせるか
    mb: float = 0.0        # ブラウザが取得する大きさ
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


ENGINES: dict[str, EngineSpec] = {e.id: e for e in [
    EngineSpec("kana-whisper", "kana-whisper (809M、サーバー GPU)", "whisper", "809M", note="教師。最も正確。候補ごとに decoder を回す"),
    EngineSpec("komimi-v12", "komimi v12 (100M)", "ctc", "100M", "ja_v12_i8_c32.kmm", True, 106.0, "komimi 最大。dev CER 16.8% (全文脈)"),
    EngineSpec("komimi-v12a", "komimi v12a (30M)", "ctc", "30M", "ja_v12a_i8_c32.kmm", True, 28.1, "PC・ブラウザ向け。17.4%"),
    EngineSpec("komimi-v12m", "komimi v12m (13M)", "ctc", "13M", "ja_v12m_i8_c32.kmm", True, 14.0, "ESP32-P4 向け。18.8%"),
    EngineSpec("komimi-v12s", "komimi v12s (6.5M)", "ctc", "6.5M", "ja_v12s_i8_c16.kmm", True, 7.1, "ESP32-S3 向け。20.2%"),
]}

#: 事前学習 (合成音声の校正) をしていない範囲で使う既定の校正値。エンジンごとに尤度の尺度が違うので別々に持つ。
#: kana-whisper は Calibration() の既定値。komimi は scripts/fit_engine_calibration.py で 3 デモの既定範囲を合わせて fit した値
DEFAULT_CALIBRATION: dict[str, dict] = {
    "kana-whisper": {"temperature": 2.5, "none_bias": 4.0, "len_bonus": 1.4, "residual_penalty": 1.0},
}


def default_calibration(engine: str) -> dict:
    return DEFAULT_CALIBRATION.get(engine) or DEFAULT_CALIBRATION.get("komimi-v12a" if engine.startswith("komimi") else "kana-whisper") \
        or DEFAULT_CALIBRATION["kana-whisper"]


def load_local(engine: str):
    """このプロセスにエンジンを読む (共有 ASR サーバーと、サーバーを使わない単独起動のデモが使う)。"""
    spec = ENGINES[engine]
    if spec.kind == "whisper":
        from .asr import KanaASR
        return KanaASR()
    from .komimi_asr import KomimiASR
    return KomimiASR(spec.file)


def _load_fitted() -> None:
    """fit 済みの既定校正 (engines_calibration.json) があれば読む。"""
    import json
    from pathlib import Path
    p = Path(__file__).with_name("engines_calibration.json")
    if p.exists():
        for k, v in json.loads(p.read_text(encoding="utf-8")).items():
            DEFAULT_CALIBRATION[k] = v


_load_fitted()
