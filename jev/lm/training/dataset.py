from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from jev.core.formats import Sample, read_jsonl
from jev.lm.models.encoding import DecisionTokenizer

ROOT = Path(__file__).resolve().parents[1]


def task_path(task: str, split: str, soft: str | None = None) -> Path:
    """soft = '<teacher>_<mode>' subdirectory under data/store/generated/<task>/ (soft labels) or None (hard)."""
    if soft:
        p = ROOT / "data/store/generated" / task / soft / f"{split}.jsonl"
        if p.exists():
            return p
    return ROOT / "data/store/processed" / task / f"{split}.jsonl"


def load_split(task: str, split: str, soft: str | None = None, limit: int | None = None) -> list[Sample]:
    return read_jsonl(str(task_path(task, split, soft)), limit)


class DecisionDataset(Dataset):
    def __init__(self, samples: list[Sample], dtok: DecisionTokenizer, target: str = "soft"):
        self.samples = samples
        self.dtok = dtok
        self.target = target  # soft | hard

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        enc = self.dtok.encode(s.state, s.question)
        hard = [1.0 if j == s.label else 0.0 for j in range(s.question.n)]
        if self.target == "soft":
            t = s.targets()
        elif self.target == "mix":            # 0.5 * gold one-hot + 0.5 * teacher distribution
            t = [0.5 * h + 0.5 * p for h, p in zip(hard, s.targets())]
        else:
            t = hard
        return enc, t, s.label if s.label is not None else -1

    def collate(self, items):
        encs, ts, labels = zip(*items)
        b = self.dtok.collate(list(encs))
        N = b["option_mask"].shape[1]
        tgt = torch.zeros((len(ts), N))
        for i, t in enumerate(ts):
            tgt[i, : len(t)] = torch.tensor(t)
        b["targets"] = tgt
        b["labels"] = torch.tensor(labels)
        return b
