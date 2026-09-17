"""TASK-008: generate soft-label datasets.

usage: python scripts/generate_teacher_data.py --task massive_scenario_en --teacher qwen3.8-27b --port 8301 \
           --splits train valid --mode logprob [--limit N]
output: data/store/generated/<task>/<teacher>_<mode>/<split>.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jev.core.formats import read_jsonl, write_jsonl  # noqa: E402
from jev.lm.backends.llm_backend import LLMBackend  # noqa: E402
from jev.lm.teacher.local_llm import label_samples  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--teacher", default="qwen3.8-27b")
    ap.add_argument("--port", type=int, default=8301)
    ap.add_argument("--mode", default="logprob", choices=["logprob", "sample"])
    ap.add_argument("--n_samples", type=int, default=20)
    ap.add_argument("--splits", nargs="+", default=["train", "valid"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=128)
    a = ap.parse_args()
    be = LLMBackend(f"http://127.0.0.1:{a.port}/v1", a.teacher, mode=a.mode, n_samples=a.n_samples, concurrency=a.concurrency)
    for split in a.splits:
        src = ROOT / "data/store/processed" / a.task / f"{split}.jsonl"
        samples = read_jsonl(str(src), a.limit)
        labelled = await label_samples(be, samples, a.mode, desc=f"{a.task}/{split}")
        dst = ROOT / "data/store/generated" / a.task / f"{a.teacher}_{a.mode}"
        dst.mkdir(parents=True, exist_ok=True)
        n = write_jsonl(str(dst / f"{split}.jsonl"), labelled)
        acc = sum(1 for s in labelled if s.label is not None and max(range(s.question.n), key=lambda i: s.target_probs[i]) == s.label) / max(1, len(labelled))
        print(f"wrote {n} -> {dst / (split + '.jsonl')}  teacher-acc={acc:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
