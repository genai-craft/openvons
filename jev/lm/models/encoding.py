"""Token layout for the decision model.

  [state segment]                                   -> shared between questions
  Question: ...\nOptions:\n
  A. id: description\n        <- option vector = hidden state at the LAST token of each option line
  B. ...\n
  Decision:<|decision|>       <- query vector = hidden at <|decision|> (pooling=decision) or last token

Segments are tokenised separately and concatenated so that positions are exact.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from jev.core.primitives import Question
from jev.core.formats import option_labels

DECISION_TOKEN = "<|decision|>"


@dataclass
class Encoded:
    ids: list[int]
    option_pos: list[int]   # absolute positions
    decision_pos: int
    state_len: int


class DecisionTokenizer:
    def __init__(self, tok, use_decision_token: bool, max_state_tokens: int = 1024):
        self.tok = tok
        self.use_decision_token = use_decision_token
        self.max_state_tokens = max_state_tokens
        self.decision_id = None
        if use_decision_token:
            if DECISION_TOKEN not in tok.get_vocab():
                tok.add_special_tokens({"additional_special_tokens": [DECISION_TOKEN]})
            self.decision_id = tok.convert_tokens_to_ids(DECISION_TOKEN)
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        self._cache: dict[str, list[int]] = {}

    def _enc(self, s: str) -> list[int]:
        r = self._cache.get(s)
        if r is None:
            r = self.tok.encode(s, add_special_tokens=False)
            if len(self._cache) < 200000:
                self._cache[s] = r
        return r

    def encode_state(self, state: str) -> list[int]:
        ids = self._enc("State:\n" + state)[: self.max_state_tokens]
        return ids + self._enc("\n\n")

    def encode_question(self, q: Question, offset: int = 0) -> tuple[list[int], list[int], int]:
        ids = list(self._enc(f"Question: {q.text}\nOptions:\n"))
        pos = []
        nl = self._enc("\n")
        for lab, o in zip(option_labels(q.n), q.options):
            ids += self._enc(f"{lab}. {o.label()}")
            pos.append(offset + len(ids) - 1)
            ids += nl
        ids += self._enc("Decision:")
        if self.use_decision_token:
            ids.append(self.decision_id)
        return ids, pos, offset + len(ids) - 1

    def encode(self, state: str, q: Question) -> Encoded:
        s = self.encode_state(state)
        qids, pos, dpos = self.encode_question(q, offset=len(s))
        return Encoded(s + qids, pos, dpos, len(s))

    def collate(self, encs: list[Encoded], device=None) -> dict[str, torch.Tensor]:
        B = len(encs)
        L = max(len(e.ids) for e in encs)
        N = max(len(e.option_pos) for e in encs)
        ids = torch.full((B, L), self.pad_id, dtype=torch.long)
        am = torch.zeros((B, L), dtype=torch.long)
        opos = torch.zeros((B, N), dtype=torch.long)
        omask = torch.zeros((B, N), dtype=torch.bool)
        dpos = torch.zeros((B,), dtype=torch.long)
        for i, e in enumerate(encs):
            ids[i, : len(e.ids)] = torch.tensor(e.ids)
            am[i, : len(e.ids)] = 1
            opos[i, : len(e.option_pos)] = torch.tensor(e.option_pos)
            omask[i, : len(e.option_pos)] = True
            dpos[i] = e.decision_pos
        b = {"input_ids": ids, "attention_mask": am, "option_pos": opos, "option_mask": omask, "decision_pos": dpos}
        if device is not None:
            b = {k: v.to(device) for k, v in b.items()}
        return b
