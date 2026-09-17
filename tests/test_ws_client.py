"""ブラウザの代わりに WebSocket へ PCM を流す統合テスト (サーバー起動中に実行).

    .venv/bin/python tests/test_ws_client.py --url ws://localhost:8600/ws --tts voicevox://127.0.0.1:50021

TTS で作った発話を 4096 サンプルずつ 実時間の 1/4 のペースで送り、前後に無音を入れて VAD の切り出しと
result イベントを確認する。pytest からも呼べる (サーバーが無ければ skip)。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


async def run(url: str, tts_url: str, texts: list[str]) -> list[dict]:
    import websockets
    from openvons.voice.synth import TTSClient
    tts = TTSClient(tts_url, cache_dir="/data/sashizu/state/tts_cache")
    results = []
    async with websockets.connect(url + "?session=wstest", max_size=None) as ws:
        hello = json.loads(await ws.recv()); assert hello["type"] == "hello"
        await ws.send(json.dumps({"type": "reset"})); await ws.recv()
        for text in texts:
            wav = tts.synth(text, seed=1)
            pcm = np.concatenate([np.zeros(16000, np.float32), wav, np.zeros(16000, np.float32)])
            pcm16 = (np.clip(pcm, -1, 1) * 32767).astype(np.int16)
            t0 = time.time()
            for i in range(0, len(pcm16), 4096):
                await ws.send(pcm16[i:i + 4096].tobytes())
                await asyncio.sleep(4096 / 16000 / 4)
            # result を待つ
            got = None
            deadline = time.time() + 10
            while time.time() < deadline:
                try:
                    m = json.loads(await asyncio.wait_for(ws.recv(), timeout=deadline - time.time()))
                except asyncio.TimeoutError:
                    break
                if m["type"] == "result":
                    got = m; break
            d = got["decision"] if got else None
            results.append({"text": text, "result": d, "wall_s": round(time.time() - t0, 2)})
            print(f"'{text}' -> {d['free_kana'] if d else 'NO RESULT'} => {d['action'] if d else '-'} {d['top']['text'] if d and d['top'] else ''} p={d['top']['prob'] if d and d['top'] else 0} state={got['state']['state'] if got else '-'}")
    return results


def test_ws_roundtrip():
    import pytest
    import httpx
    try:
        httpx.get("http://localhost:8600/api/state", timeout=2)
    except Exception:
        pytest.skip("demo server not running")
    res = asyncio.run(run("ws://localhost:8600/ws", "voicevox://127.0.0.1:50021", ["船橋南1上りを表示", "もっと右に向けて", "一覧に戻る"]))
    assert all(r["result"] is not None for r in res)
    assert res[0]["result"]["action"] == "execute" and res[0]["result"]["top"]["text"].startswith("船橋南1上り")
    assert res[1]["result"]["top"]["intent"] == "pan_right"
    assert res[2]["result"]["top"]["intent"] == "back"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://localhost:8600/ws")
    ap.add_argument("--tts", default="voicevox://127.0.0.1:50021")
    ap.add_argument("texts", nargs="*", default=["船橋南1上りを表示", "もっと右に向けて", "この位置を保存して", "はい", "一覧に戻る", "はい、お世話になっております"])
    a = ap.parse_args()
    asyncio.run(run(a.url, a.tts, a.texts))
