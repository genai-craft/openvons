"""VOICEVOX ENGINE (https://voicevox.hiroshiba.jp/, docker voicevox/voicevox_engine:cpu-latest, :50021).
audio_query が読み (アクセント付きカナ) も返すので、G2P の照合にも使える (reading())。"""
from __future__ import annotations

import json

from .base import TTSBackend


class VoicevoxTTS(TTSBackend):
    name = "voicevox"

    def __init__(self, base_url: str, speakers: list[int], **kw):
        super().__init__(**kw)
        self.base_url = base_url.rstrip("/")
        self.speakers = speakers

    def ok(self) -> bool:
        try:
            return self.client.get(self.base_url + "/version").status_code == 200
        except Exception:
            return False

    def speaker_for(self, seed: int) -> int:
        return self.speakers[seed % len(self.speakers)]

    def query(self, text: str, seed: int = 0) -> dict:
        r = self.client.post(self.base_url + "/audio_query", params={"speaker": self.speaker_for(seed), "text": text})
        r.raise_for_status()
        return r.json()

    def reading(self, text: str) -> str:
        """VOICEVOX の読み (アクセント記号を除いたカタカナ)。"""
        k = self.query(text).get("kana", "")
        return k.replace("'", "").replace("/", "").replace("、", "").replace("_", "").replace("？", "")

    def _request(self, text: str, seed: int) -> bytes:
        q = self.query(text, seed)
        q["outputSamplingRate"] = 24000
        r = self.client.post(self.base_url + "/synthesis", params={"speaker": self.speaker_for(seed)}, content=json.dumps(q), headers={"Content-Type": "application/json"})
        r.raise_for_status()
        return r.content
