"""TTS バックエンド (事前学習の合成音声用)。URL のスキームで選ぶ:

    voicevox://127.0.0.1:50021?speaker=3      VOICEVOX ENGINE (無償・ローカル・複数話者)  ← 既定
    irodori://172.20.0.3:8093                  Irodori TTS の HTTP ワーカー (Aunvox 同梱)
    openai://host:port/v1?model=tts-1&voice=alloy   OpenAI 互換 /v1/audio/speech
どれも synth(text, seed) -> 16kHz float32 を返す。seed は「声の違い」に写像する (VOICEVOX は話者 id の巡回)。
"""
from __future__ import annotations

from .base import TTSBackend, get_backend, to16k  # noqa: F401
