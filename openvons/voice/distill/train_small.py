"""小型のカナ ASR を蒸留する (端末内推論用).

    # 1 GPU
    CUDA_VISIBLE_DEVICES=4 .venv/bin/python -m openvons.voice.distill.train_small \
        --labels state/distill/reazon_medium_kana.jsonl --parquet-dir <reazonspeech parquet dir> \
        --student openai/whisper-small --decoder-layers 4 --teacher sbintuitions/kana-whisper --kl-weight 0.7 \
        --out state/distill/kana-whisper-small-4l

    # 4 GPU (DDP)
    CUDA_VISIBLE_DEVICES=4,5,6,7 .venv/bin/torchrun --nproc_per_node=4 -m openvons.voice.distill.train_small ... --batch 24

レシピ (distil-whisper / kotoba-whisper / ふりがな Whisper の組み合わせ):
  - 教師 = kana-whisper (809M) の疑似ラベル (カナ)。学生は whisper-small の重みで、decoder を先頭 2 層 + 最終 2 層に切る
  - 損失 = CE(疑似ラベル) + KL(教師の分布 || 学生の分布)。KL は語彙の共通部分 (内容トークン 0..50256 + EOT) だけで取る
    (教師 large-v3 系は語彙 51866、学生 small は 51865。内容トークンの id は同じで、特殊トークンだけずれている)
  - encoder は最初の数千 step 凍結 → その後まとめて学習
  - 推論は非カナトークンを抑制 (distill_info.json に同梱)

データ読みは parquet の行グループ単位でまとめて読む (ブロック単位でシャッフル)。完全ランダムだと毎回読み直しになって遅い。
"""
from __future__ import annotations

import argparse
import io
import json
import os
import random
import re
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, Sampler

from openvons.tts import to16k

CONTENT_VOCAB = 50258          # 0..50256 が内容トークン、50257 が EOT。ここまでを教師と学生で共有する


class KanaSet(Dataset):
    def __init__(self, rows: list[dict], parquet_dir: Path, processor, max_len: int = 96):
        self.rows = rows
        self.dir = Path(parquet_dir)
        self.proc = processor
        self.max_len = max_len
        self._pf: dict[str, pq.ParquetFile] = {}
        self._cache: OrderedDict = OrderedDict()

    def __len__(self) -> int:
        return len(self.rows)

    def blocks(self) -> dict[tuple[str, int], list[int]]:
        out: dict[tuple[str, int], list[int]] = {}
        for i, r in enumerate(self.rows):
            pf, rg, _ = r["id"].split(":")
            out.setdefault((pf, int(rg)), []).append(i)
        return out

    def _row_group(self, pf: str, rg: int):
        key = (pf, rg)
        if key not in self._cache:
            if len(self._cache) > 2:
                self._cache.popitem(last=False)
            f = self._pf.get(pf)
            if f is None:
                f = self._pf[pf] = pq.ParquetFile(self.dir / pf)
            self._cache[key] = f.read_row_group(rg).to_pandas()
        return self._cache[key]

    def __getitem__(self, i: int):
        r = self.rows[i]
        pf, rg, idx = r["id"].split(":")
        df = self._row_group(pf, int(rg))
        cell = df.loc[int(idx), "audio"]
        a, sr = sf.read(io.BytesIO(cell["bytes"]))
        feats = self.proc.feature_extractor(to16k(a, sr), sampling_rate=16000, return_tensors="np").input_features[0]
        ids = self.proc.tokenizer(r["kana"], add_special_tokens=False).input_ids[: self.max_len]
        return torch.from_numpy(feats), ids


class BlockSampler(Sampler):
    """同じ行グループの行をまとめて出す (毎 epoch、ブロック順と中身をシャッフル)。DDP では rank ごとにブロックを分ける。"""

    def __init__(self, ds: KanaSet, rank: int = 0, world: int = 1, seed: int = 0):
        self.blocks = list(ds.blocks().values())
        self.rank, self.world, self.seed, self.epoch = rank, world, seed, 0
        self.n = sum(len(b) for b in self.blocks[rank::world])

    def set_epoch(self, e: int) -> None:
        self.epoch = e

    def __len__(self) -> int:
        return self.n

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        blocks = self.blocks[self.rank::self.world]
        rng.shuffle(blocks)
        for b in blocks:
            b = list(b)
            rng.shuffle(b)
            yield from b


def collate(batch, prefix: list[int], eot: int):
    feats = torch.stack([b[0] for b in batch])
    seqs = [prefix + ids + [eot] for _, ids in batch]
    L = max(len(x) for x in seqs)
    ids = torch.full((len(seqs), L), eot, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.bool)
    for i, x in enumerate(seqs):
        ids[i, : len(x)] = torch.tensor(x)
        mask[i, : len(x)] = True
    mask[:, : len(prefix)] = False              # 強制 prefix は学習しない
    return feats, ids, mask


def cut_decoder(model, n_layers: int):
    dec = model.model.decoder
    if n_layers >= len(dec.layers):
        return
    keep = list(range(n_layers // 2)) + list(range(len(dec.layers) - (n_layers - n_layers // 2), len(dec.layers)))
    dec.layers = torch.nn.ModuleList([dec.layers[i] for i in keep])
    model.config.decoder_layers = len(dec.layers)
    # 残した層は元の layer_idx を持ったままなので振り直す (KV cache の添字が範囲外になる)
    for new_idx, layer in enumerate(dec.layers):
        for attr in ("self_attn", "encoder_attn"):
            a = getattr(layer, attr, None)
            if a is not None and hasattr(a, "layer_idx"):
                a.layer_idx = new_idx
        if hasattr(layer, "layer_idx"):
            layer.layer_idx = new_idx


def cer(ref: str, hyp: str) -> float:
    from rapidfuzz.distance import Levenshtein
    return Levenshtein.distance(ref, hyp) / max(len(ref), 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--parquet-dir", required=True)
    ap.add_argument("--student", default="openai/whisper-small")
    ap.add_argument("--decoder-layers", type=int, default=4)
    ap.add_argument("--teacher", default="sbintuitions/kana-whisper")
    ap.add_argument("--kl-weight", type=float, default=0.7, help="0 で CE のみ")
    ap.add_argument("--kl-temp", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--freeze-encoder-steps", type=int, default=2000)
    ap.add_argument("--max-sec", type=float, default=20.0)
    ap.add_argument("--min-chars", type=int, default=2)
    ap.add_argument("--valid", type=int, default=600)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--resume", default="")
    args = ap.parse_args()
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world > 1
    if ddp:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    rank = dist.get_rank() if ddp else 0
    log = (lambda *a: print(*a, flush=True)) if rank == 0 else (lambda *a: None)

    rows = [json.loads(l) for l in open(args.labels, encoding="utf-8")]
    rows = [r for r in rows if r["duration"] <= args.max_sec and len(r["kana"]) >= args.min_chars
            and not re.search(r"(.{1,3}?)\1{5,}", r["kana"])]          # 反復ハルシネーションを落とす
    random.Random(0).shuffle(rows)
    valid, train = rows[: args.valid], rows[args.valid:]
    log(f"train {len(train)}  valid {len(valid)}  hours {sum(r['duration'] for r in train) / 3600:.1f}  world {world}")

    proc = WhisperProcessor.from_pretrained(args.student)
    model = WhisperForConditionalGeneration.from_pretrained(args.resume or args.student, dtype=torch.float32)
    if not args.resume:
        cut_decoder(model, args.decoder_layers)
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    model.to(device)
    tok = proc.tokenizer
    prefix = tok.convert_tokens_to_ids(["<|startoftranscript|>", "<|ja|>", "<|transcribe|>", "<|notimestamps|>"])
    eot = tok.eos_token_id
    suppress = [t for t in range(eot) if not re.fullmatch(r"[ァ-ヶー]+", tok.decode([t], skip_special_tokens=True) or "")]

    teacher = None
    if args.kl_weight > 0:
        teacher = WhisperForConditionalGeneration.from_pretrained(args.teacher, dtype=torch.bfloat16).to(device).eval()
        for p in teacher.parameters():
            p.requires_grad = False
        t_tok = WhisperProcessor.from_pretrained(args.teacher).tokenizer
        t_prefix = t_tok.convert_tokens_to_ids(["<|startoftranscript|>", "<|ja|>", "<|transcribe|>", "<|notimestamps|>"])
        log(f"teacher {args.teacher} (KL {args.kl_weight}, 共通語彙 {CONTENT_VOCAB})")
    else:
        t_prefix = prefix

    ds = KanaSet(train, Path(args.parquet_dir), proc)
    sampler = BlockSampler(ds, rank, world)
    dl = DataLoader(ds, batch_size=args.batch, sampler=sampler, num_workers=args.workers,
                    collate_fn=lambda b: collate(b, prefix, eot), drop_last=True, persistent_workers=args.workers > 0,
                    prefetch_factor=4 if args.workers else None)
    vds = KanaSet(valid, Path(args.parquet_dir), proc)

    steps_total = int(len(dl) * args.epochs)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 500) * max(0.05, 1 - s / max(steps_total, 1)))
    for p in model.model.encoder.parameters():
        p.requires_grad = False
    net = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True) if ddp else model

    out = Path(args.out)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    step, best, t0 = 0, 9.9, time.time()
    log(f"steps/epoch {len(dl)}  total {steps_total}")
    model.train()
    ep = 0
    while step < steps_total:
        sampler.set_epoch(ep); ep += 1
        for feats, ids, mask in dl:
            if step == args.freeze_encoder_steps:
                for p in model.model.encoder.parameters():
                    p.requires_grad = True
                log("encoder unfrozen")
            feats, ids, mask = feats.to(device, non_blocking=True), ids.to(device), mask.to(device)
            dec_in, labels_mask = ids[:, :-1], mask[:, 1:]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = net(input_features=feats, decoder_input_ids=dec_in, use_cache=False).logits
            logits = logits.float()
            tgt = ids[:, 1:].clone()
            tgt[~labels_mask] = -100
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), ignore_index=-100)
            if teacher is not None:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    t_in = dec_in.clone()
                    t_in[:, : len(prefix)] = torch.tensor(t_prefix, device=device)      # prefix の id だけ教師側に合わせる
                    t_logits = teacher(input_features=feats, decoder_input_ids=t_in, use_cache=False).logits.float()
                T = args.kl_temp
                s_lp = torch.log_softmax(logits[..., :CONTENT_VOCAB] / T, -1)
                t_p = torch.softmax(t_logits[..., :CONTENT_VOCAB] / T, -1)
                kl = (t_p * (torch.log(t_p.clamp_min(1e-9)) - s_lp)).sum(-1)
                kl = (kl * labels_mask).sum() / labels_mask.sum().clamp_min(1)
                loss = (1 - args.kl_weight) * loss + args.kl_weight * (T * T) * kl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            if step % 100 == 0:
                log(f"step {step}/{steps_total} loss {loss.item():.3f} lr {sched.get_last_lr()[0]:.2e} {(time.time() - t0) / 60:.1f}min")
            if step % args.eval_every == 0 or step >= steps_total:
                if rank == 0:
                    c = evaluate(model, vds, proc, prefix, eot, suppress, device)
                    log(f"== step {step} valid kana CER {c:.4f}")
                    info = {"suppress_tokens_kana_only": suppress, "valid_cer": c, "step": step, "student": args.student,
                            "decoder_layers": model.config.decoder_layers, "teacher": args.teacher if teacher else None,
                            "kl_weight": args.kl_weight, "train_hours": round(sum(r["duration"] for r in train) / 3600, 1)}
                    last = out.parent / (out.name + "-last")
                    model.save_pretrained(last); proc.save_pretrained(last); json.dump(info, open(last / "distill_info.json", "w"))
                    if c < best:
                        best = c
                        model.save_pretrained(out); proc.save_pretrained(out); json.dump(info, open(out / "distill_info.json", "w"))
                        log(f"saved {out} (CER {c:.4f})")
                    model.train()
                if ddp:
                    dist.barrier()
            if step >= steps_total:
                break
    log(f"done best CER {best}")
    if ddp:
        dist.destroy_process_group()


@torch.no_grad()
def evaluate(model, vds, proc, prefix, eot, suppress, device, n: int = 300, bs: int = 24) -> float:
    model.eval()
    tot, cnt = 0.0, 0
    for s in range(0, min(n, len(vds)), bs):
        items = [vds[i] for i in range(s, min(s + bs, n, len(vds)))]
        feats = torch.stack([it[0] for it in items]).to(device)
        out = model.generate(input_features=feats, decoder_input_ids=torch.tensor([prefix] * len(items), device=device),
                             max_new_tokens=96, num_beams=1, do_sample=False, suppress_tokens=suppress)
        for seq, (_, ids) in zip(out.tolist(), items):
            hyp = proc.tokenizer.decode([t for t in seq if t < eot], skip_special_tokens=True)
            ref = proc.tokenizer.decode(ids, skip_special_tokens=True)
            tot += cer(ref, hyp); cnt += 1
    return tot / max(cnt, 1)


if __name__ == "__main__":
    main()
