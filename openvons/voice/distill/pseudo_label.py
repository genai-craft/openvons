"""ReazonSpeech (HF parquet) に kana-whisper で疑似ラベル (カナ) を付ける.

    CUDA_VISIBLE_DEVICES=5 .venv/bin/python -m openvons.voice.distill.pseudo_label \
        --parquet-dir /data/lychee_ja/reazon/hf/small/small --out /data/openjev/distill/reazon_small_kana.jsonl

出力 jsonl: {"id", "kana", "text" (元の書き起こし), "duration", "logprob", "parquet", "row"}
音声そのものは書き出さない (parquet から都度読む)。ReazonSpeech は再配布不可 (docs/licensing.md)。
"""
from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch

from openvons.tts import to16k
from openvons.voice.asr import KanaASR


def decode_audio(cell) -> tuple[np.ndarray, int]:
    """HF audio 列 ({bytes, path}) と、ReazonSpeech parquet の素の bytes 列の両方に対応。"""
    if isinstance(cell, (bytes, bytearray)):
        a, sr = sf.read(io.BytesIO(bytes(cell)))
    elif isinstance(cell, dict) and cell.get("bytes"):
        a, sr = sf.read(io.BytesIO(cell["bytes"]))
    else:
        a, sr = sf.read(cell["path"])
    return a, sr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--max-sec", type=float, default=30.0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in open(out, encoding="utf-8"):
            done.add(json.loads(line)["id"])
    asr = KanaASR()
    proc = asr.processor
    files = sorted(Path(args.parquet_dir).glob("*.parquet"))
    n = 0; t0 = time.time()
    with open(out, "a", encoding="utf-8") as fo:
        for pf in files:
            table = pq.ParquetFile(pf)
            cols = table.schema_arrow.names          # 入れ子 (audio 構造体) は葉ではなく最上位の列名で見る
            audio_col = "audio" if "audio" in cols else ("bytes" if "bytes" in cols else cols[0])
            text_col = "transcription" if "transcription" in cols else next((c for c in cols if "trans" in c or "text" in c), None)
            for rg in range(table.num_row_groups):
                df = table.read_row_group(rg).to_pandas()
                batch_wavs, batch_meta = [], []
                for i, row in df.iterrows():
                    rid = f"{pf.name}:{rg}:{i}"
                    if rid in done:
                        continue
                    try:
                        a, sr = decode_audio(row[audio_col])
                    except Exception:
                        continue
                    wav = to16k(a, sr)
                    dur = len(wav) / 16000
                    if dur > args.max_sec or dur < 0.3:
                        continue
                    batch_wavs.append(wav); batch_meta.append((rid, str(row[text_col]) if text_col else "", dur, pf.name, int(i)))
                    if len(batch_wavs) >= args.batch:
                        n += flush(asr, proc, batch_wavs, batch_meta, fo); batch_wavs, batch_meta = [], []
                        if n % (args.batch * 20) == 0:
                            print(f"{n} utts  {n / (time.time() - t0):.1f}/s", flush=True)
                    if args.limit and n >= args.limit:
                        return
                if batch_wavs:
                    n += flush(asr, proc, batch_wavs, batch_meta, fo)
    print("done", n, "utts in", round(time.time() - t0), "s")


@torch.no_grad()
def flush(asr: KanaASR, proc, wavs, meta, fo) -> int:
    feats = proc(wavs, sampling_rate=16000, return_tensors="pt").input_features.to(asr.device, asr.dtype)
    out = asr.model.generate(feats, max_new_tokens=128, num_beams=1, do_sample=False, suppress_tokens=asr.suppress_ids, language="ja", task="transcribe")
    for seq, (rid, text, dur, pf, row) in zip(out.tolist(), meta):
        content = [t for t in seq if t < asr.eot]
        kana = asr.tok.decode(content, skip_special_tokens=True).strip()
        fo.write(json.dumps({"id": rid, "kana": kana, "text": text, "duration": round(dur, 2), "parquet": pf, "row": row}, ensure_ascii=False) + "\n")
    fo.flush()
    return len(meta)


if __name__ == "__main__":
    main()
