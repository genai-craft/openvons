"""Decision-model backend with the four multi-question strategies of spec Phase 6.

  naive      : one forward per question  (state + Q_k)
  batched    : all questions padded into one batch (v3)
  kv_shared  : prefill the state once, expand its KV cache, run the questions as a batch (v2 + v3)
  block_diag : one sequence [state | Q1 | Q2 | ...] with a block-diagonal attention mask so that every
               question attends to the state and to itself only (v4)
"""
from __future__ import annotations

import time

import torch

from openvons.core.primitives import Question
from openvons.lm.models.decision_model import DecisionModel
from .base import Decision, DecisionBackend

MODES = ["naive", "batched", "kv_shared", "block_diag"]


class ModelBackend(DecisionBackend):
    name = "model"

    def __init__(self, model: DecisionModel, mode: str = "kv_shared"):
        self.model = model
        self.mode = mode
        self.dtok = model.dtok
        self.device = model.device

    @torch.no_grad()
    def logits(self, state: str, qs: list[Question], mode: str | None = None) -> list[torch.Tensor]:
        mode = mode or self.mode
        m = self.model
        if mode == "naive":
            outs = []
            for q in qs:
                b = self.dtok.collate([self.dtok.encode(state, q)], self.device)
                outs.append(m(b, q.type)[0, : q.n])
            return outs
        if mode == "batched":
            encs = [self.dtok.encode(state, q) for q in qs]
            b = self.dtok.collate(encs, self.device)
            lg = m(b, qs[0].type)
            return [lg[i, : q.n] for i, q in enumerate(qs)]
        if mode == "kv_shared":
            return self._kv_shared(state, qs)
        if mode == "block_diag":
            return self._block_diag(state, qs)
        raise ValueError(mode)

    def _kv_shared(self, state: str, qs: list[Question]) -> list[torch.Tensor]:
        m = self.model
        s_ids = self.dtok.encode_state(state)
        S = len(s_ids)
        s = torch.tensor([s_ids], device=self.device)
        out = m.backbone(input_ids=s, use_cache=True)
        cache = out.past_key_values
        B = len(qs)
        cache.batch_repeat_interleave(B)
        encs = []
        for q in qs:
            ids, pos, dpos = self.dtok.encode_question(q, offset=0)
            encs.append((ids, pos, dpos))
        L = max(len(e[0]) for e in encs)
        N = max(len(e[1]) for e in encs)
        ids = torch.full((B, L), self.dtok.pad_id, dtype=torch.long)
        am = torch.zeros((B, S + L), dtype=torch.long)
        am[:, :S] = 1
        opos = torch.zeros((B, N), dtype=torch.long)
        omask = torch.zeros((B, N), dtype=torch.bool)
        dpos = torch.zeros((B,), dtype=torch.long)
        for i, (qids, pos, dp) in enumerate(encs):
            ids[i, : len(qids)] = torch.tensor(qids)
            am[i, S : S + len(qids)] = 1
            opos[i, : len(pos)] = torch.tensor(pos)
            omask[i, : len(pos)] = True
            dpos[i] = dp
        ids, am, opos, omask, dpos = (t.to(self.device) for t in (ids, am, opos, omask, dpos))
        position_ids = (S + torch.arange(L, device=self.device)).unsqueeze(0).expand(B, -1)
        cache_position = S + torch.arange(L, device=self.device)
        h = m.backbone(input_ids=ids, attention_mask=am, past_key_values=cache, position_ids=position_ids,
                       cache_position=cache_position, use_cache=True).last_hidden_state
        batch = {"attention_mask": am[:, S:], "decision_pos": dpos, "option_pos": opos, "option_mask": omask}
        pooled = self._pool(h, batch)
        opt = h[torch.arange(B, device=self.device).unsqueeze(1), opos]
        lg = m.head(pooled.to(m.head_dtype), opt.to(m.head_dtype), omask, qs[0].type)
        return [lg[i, : q.n] for i, q in enumerate(qs)]

    def _block_diag(self, state: str, qs: list[Question]) -> list[torch.Tensor]:
        m = self.model
        s_ids = self.dtok.encode_state(state)
        S = len(s_ids)
        ids = list(s_ids)
        pos_ids = list(range(S))
        seg = [0] * S            # 0 = state, k = question k
        opt_pos, dec_pos = [], []
        for k, q in enumerate(qs, start=1):
            qids, pos, dp = self.dtok.encode_question(q, offset=len(ids))
            ids += qids
            pos_ids += list(range(S, S + len(qids)))
            seg += [k] * len(qids)
            opt_pos.append(pos)
            dec_pos.append(dp)
        L = len(ids)
        seg_t = torch.tensor(seg, device=self.device)
        i = torch.arange(L, device=self.device)
        causal = i.unsqueeze(1) >= i.unsqueeze(0)
        same = seg_t.unsqueeze(1) == seg_t.unsqueeze(0)
        to_state = (seg_t.unsqueeze(0) == 0).expand(L, L)
        mask = causal & (same | to_state)                                    # (L, L) True = attend
        mask4d = mask.unsqueeze(0).unsqueeze(0)
        ids_t = torch.tensor([ids], device=self.device)
        pid = torch.tensor([pos_ids], device=self.device)
        h = m.backbone(input_ids=ids_t, attention_mask=mask4d, position_ids=pid).last_hidden_state[0]  # (L, H)
        outs = []
        for q, pos, dp in zip(qs, opt_pos, dec_pos):
            pooled = h[dp].unsqueeze(0)
            opt = h[torch.tensor(pos, device=self.device)].unsqueeze(0)
            omask = torch.ones((1, q.n), dtype=torch.bool, device=self.device)
            outs.append(m.head(pooled.to(m.head_dtype), opt.to(m.head_dtype), omask, q.type)[0])
        return outs

    def _pool(self, h, batch):
        from openvons.lm.models.pooling import pool
        return pool(self.model.cfg.pooling, h, batch["attention_mask"], batch["decision_pos"], self.model.attn_pool)

    def decide(self, state: str, questions: list[Question]) -> list[Decision]:
        t0 = time.perf_counter()
        lgs = self.logits(state, questions)
        probs = [self.model.probs(lg).tolist() for lg in lgs]
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000
        return [Decision(p, dt, {"mode": self.mode, "n_questions": len(questions)}) for p in probs]
