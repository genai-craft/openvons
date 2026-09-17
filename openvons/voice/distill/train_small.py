"""whisper-small (または base) をカナ出力に蒸留する (スマホ搭載用の小型 kana モデル).

    CUDA_VISIBLE_DEVICES=5 .venv/bin/python -m openvons.voice.distill.train_small \
        --labels state/distill/reazon_small_kana.jsonl --parquet-dir <reazonspeech parquet dir> \
        --student openai/whisper-small --decoder-layers 2 --out state/distill/kana-whisper-small-2l

レシピ (distil-whisper / kotoba-whisper / ふりがな Whisper の組み合わせ):
  - 教師 = kana-whisper の疑似ラベル (カナ)。学生は whisper-small の重みで初期化し、decoder を先頭 1 層 + 最終 1 層の 2 層に切る
  - encoder は最初の数エポック凍結 (--freeze-encoder-steps)、その後全体を学習
  - 出力語彙をカナに限定: 学習は CE のみ (KL 蒸留は教師の語彙分布を取り直す必要があり、まず CE で)。推論時は suppress_tokens で非カナを抑制
  - 学習中の評価: 検証 1,000 発話の kana CER と、examples/road_cameras の合成評価 (eval_synthetic を --asr-model で)
"""
from __future__ import annotations

import argparse
import io
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset

from openvons.tts import to16k


class KanaSet(Dataset):
    def __init__(self, rows: list[dict], parquet_dir: Path, processor, max_len: int = 96):
        self.rows = rows; self.dir = parquet_dir; self.proc = processor; self.max_len = max_len
        self._pf: dict[str, pq.ParquetFile] = {}
        self._cache: dict[tuple[str, int], object] = {}

    def __len__(self):
        return len(self.rows)

    def _row_group(self, pf: str, rg: int):
        key = (pf, rg)
        if key not in self._cache:
            if len(self._cache) > 6:
                self._cache.pop(next(iter(self._cache)))
            f = self._pf.setdefault(pf, pq.ParquetFile(self.dir / pf))
            self._cache[key] = f.read_row_group(rg).to_pandas()
        return self._cache[key]

    def __getitem__(self, i):
        r = self.rows[i]
        pf, rg, idx = r["id"].split(":")
        df = self._row_group(pf, int(rg))
        cell = df.loc[int(idx), "audio"]
        a, sr = sf.read(io.BytesIO(cell["bytes"]))
        wav = to16k(a, sr)
        feats = self.proc.feature_extractor(wav, sampling_rate=16000, return_tensors="np").input_features[0]
        ids = self.proc.tokenizer(r["kana"], add_special_tokens=False).input_ids[: self.max_len]
        return torch.from_numpy(feats), ids


def collate(batch, prefix: list[int], eot: int):
    feats = torch.stack([b[0] for b in batch])
    seqs = [prefix + ids + [eot] for _, ids in batch]
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), eot, dtype=torch.long)
    labels = torch.full((len(seqs), L), -100, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
        labels[i, : len(s)] = torch.tensor(s)
    labels[:, : len(prefix)] = -100         # prefix は学習しない
    return feats, ids[:, :-1], labels[:, 1:]


def cer(ref: str, hyp: str) -> float:
    import rapidfuzz.distance.Levenshtein as L
    return L.distance(ref, hyp) / max(len(ref), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--parquet-dir", required=True)
    ap.add_argument("--student", default="openai/whisper-small")
    ap.add_argument("--decoder-layers", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--freeze-encoder-steps", type=int, default=1500)
    ap.add_argument("--max-sec", type=float, default=20.0)
    ap.add_argument("--min-chars", type=int, default=2)
    ap.add_argument("--valid", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    rows = [json.loads(l) for l in open(args.labels, encoding="utf-8")]
    rows = [r for r in rows if r["duration"] <= args.max_sec and len(r["kana"]) >= args.min_chars and "ハハハハハ" not in r["kana"]]
    random.Random(0).shuffle(rows)
    valid, train = rows[: args.valid], rows[args.valid:]
    print(f"train {len(train)}  valid {len(valid)}  hours {sum(r['duration'] for r in train) / 3600:.1f}", flush=True)

    proc = WhisperProcessor.from_pretrained(args.student)
    model = WhisperForConditionalGeneration.from_pretrained(args.student, dtype=torch.float32)
    # decoder を切る: 先頭層 + 最終層 (distil-whisper の初期化)
    dec = model.model.decoder
    if args.decoder_layers < len(dec.layers):
        keep = [0] + list(range(len(dec.layers) - (args.decoder_layers - 1), len(dec.layers))) if args.decoder_layers > 1 else [0]
        dec.layers = torch.nn.ModuleList([dec.layers[i] for i in keep])
        model.config.decoder_layers = len(dec.layers)
        # 残した層は元の layer_idx (例: 11) を持ったままなので、KV cache の添字が範囲外になる → 新しい位置に振り直す
        for new_idx, layer in enumerate(dec.layers):
            for attr in ("self_attn", "encoder_attn"):
                a = getattr(layer, attr, None)
                if a is not None and hasattr(a, "layer_idx"):
                    a.layer_idx = new_idx
            if hasattr(layer, "layer_idx"):
                layer.layer_idx = new_idx
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    tok = proc.tokenizer
    prefix = tok.convert_tokens_to_ids(["<|startoftranscript|>", "<|ja|>", "<|transcribe|>", "<|notimestamps|>"])
    eot = tok.eos_token_id
    device = torch.device("cuda")
    model.to(device)
    # 非カナのトークンを抑制するリスト (推論用、保存する)
    kana_ok = set("ァ-ヶー")
    import re
    suppress = [t for t in range(tok.eos_token_id) if not re.fullmatch(r"[ァ-ヶー]+", tok.decode([t], skip_special_tokens=True) or "")]
    ds = KanaSet(train, Path(args.parquet_dir), proc)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, collate_fn=lambda b: collate(b, prefix, eot), drop_last=True, persistent_workers=True)
    vds = KanaSet(valid, Path(args.parquet_dir), proc)
    steps_total = int(len(dl) * args.epochs)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 200) * max(0.05, 1 - s / max(steps_total, 1)))
    scaler = torch.amp.GradScaler()
    for p in model.model.encoder.parameters():
        p.requires_grad = False
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    step = 0; t0 = time.time(); best = 9.9
    model.train()
    while step < steps_total:
        for feats, dec_in, labels in dl:
            if step == args.freeze_encoder_steps:
                for p in model.model.encoder.parameters():
                    p.requires_grad = True
                print("encoder unfrozen", flush=True)
            feats, dec_in, labels = feats.to(device), dec_in.to(device), labels.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_features=feats, decoder_input_ids=dec_in, use_cache=False).logits
                loss = torch.nn.functional.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            if step % 50 == 0:
                print(f"step {step}/{steps_total} loss {loss.item():.3f} lr {sched.get_last_lr()[0]:.2e} {(time.time() - t0) / 60:.1f}min", flush=True)
            if step % args.eval_every == 0 or step >= steps_total:
                c = evaluate(model, vds, proc, prefix, eot, suppress, device)
                print(f"== step {step} valid kana CER {c:.4f}", flush=True)
                last = out.parent / (out.name + "-last")
                model.save_pretrained(last); proc.save_pretrained(last)
                json.dump({"suppress_tokens_kana_only": suppress, "valid_cer": c, "step": step, "student": args.student, "decoder_layers": args.decoder_layers}, open(last / "distill_info.json", "w"))
                if c < best:
                    best = c
                    model.save_pretrained(out); proc.save_pretrained(out)
                    json.dump({"suppress_tokens_kana_only": suppress, "valid_cer": c, "step": step, "student": args.student, "decoder_layers": args.decoder_layers}, open(out / "distill_info.json", "w"))
                    print("saved", out, flush=True)
                model.train()
            if step >= steps_total:
                break
    print("done best CER", best)


@torch.no_grad()
def evaluate(model, vds, proc, prefix, eot, suppress, device, n: int = 300, bs: int = 24) -> float:
    model.eval()
    tot = 0.0; cnt = 0
    idx = list(range(min(n, len(vds))))
    for s in range(0, len(idx), bs):
        items = [vds[i] for i in idx[s: s + bs]]
        feats = torch.stack([it[0] for it in items]).to(device)
        # 生成は fp32 で (bf16 autocast の generate はカナがほぼ出ず CER 0.98 になった。教師強制の loss 0.37 とは矛盾していた)
        out = model.generate(input_features=feats, decoder_input_ids=torch.tensor([prefix] * len(items), device=device), max_new_tokens=96, num_beams=1, do_sample=False, suppress_tokens=suppress)
        for seq, (_, ids) in zip(out.tolist(), items):
            hyp = proc.tokenizer.decode([t for t in seq if t < eot], skip_special_tokens=True)
            ref = proc.tokenizer.decode(ids, skip_special_tokens=True)
            tot += cer(ref, hyp); cnt += 1
    return tot / max(cnt, 1)


if __name__ == "__main__":
    main()
