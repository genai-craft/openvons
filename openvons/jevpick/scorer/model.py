"""OpenVons Block Scorer (§7)。

入力: target hidden h (選択 layer), anchor token x の embedding, 候補 token 列の embedding (target の embedding を流用, 凍結),
      source one-hot, prior 特徴。
出力: 各候補・各 prefix 長 k について logit of P(match >= k) (S3: prefix-position 表現)。
      score = Σ_k P(match >= k) = 期待受理長。
候補 encoder は S1 (prefix mean pooling) と S2 (小型 causal Transformer) を切替。
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

SOURCES = ("ngram", "grammar", "corpus", "schema", "macro", "repo", "dflash", "mtp")
N_PRIOR = 3


class BlockScorer(nn.Module):
    def __init__(self, hidden: int, d: int = 512, max_len: int = 16, encoder: str = "pool", n_layers: int = 2):
        super().__init__()
        self.encoder = encoder
        self.h_proj = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, d))
        self.x_proj = nn.Linear(hidden, d)
        self.c_proj = nn.Linear(hidden, d)
        self.pos = nn.Embedding(max_len, d)
        self.src = nn.Embedding(len(SOURCES) + 1, d)
        self.prior = nn.Linear(N_PRIOR, d)
        if encoder == "transformer":
            layer = nn.TransformerEncoderLayer(d, 8, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.ctx = nn.Sequential(nn.Linear(2 * d, d), nn.GELU())

    def forward(self, h, x_emb, c_emb, c_mask, src_id, prior):
        """h: (B, H)  x_emb: (B, H)  c_emb: (B, C, L, H)  c_mask: (B, C, L) bool  src_id: (B, C)  prior: (B, C, 3)
        戻り値: logits (B, C, L) = logit P(match >= k+1) (padding 位置は -inf 相当で埋めない; mask で扱う)"""
        B, C, L, _ = c_emb.shape
        ctx = self.ctx(torch.cat([self.h_proj(h), self.x_proj(x_emb)], -1))  # (B, d)
        c = self.c_proj(c_emb) + self.pos.weight[:L]  # (B, C, L, d)
        if self.encoder == "pool":
            # prefix mean pooling: 位置 k の表現 = c[:k+1] の平均
            cs = torch.cumsum(c * c_mask[..., None], dim=2)
            cnt = torch.cumsum(c_mask.float(), dim=2).clamp(min=1)[..., None]
            rep = cs / cnt
        else:
            flat = c.view(B * C, L, -1)
            causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=c.device), 1)
            rep = self.enc(flat, mask=causal, src_key_padding_mask=~c_mask.view(B * C, L)).view(B, C, L, -1)
            rep = torch.nan_to_num(rep)
        meta = self.src(src_id) + self.prior(prior)  # (B, C, d)
        z = torch.cat([ctx[:, None, None].expand(B, C, L, -1), rep, meta[:, :, None].expand(B, C, L, -1)], -1)
        return self.head(z).squeeze(-1)

    @staticmethod
    def expected_len(logits, c_mask):
        p = torch.sigmoid(logits) * c_mask
        # P(match>=k) は単調非増加のはず。累積最小で整合させる
        p = torch.cummin(p + (~c_mask) * 0.0, dim=-1).values * c_mask
        return p.sum(-1)  # (B, C)


def scorer_loss(logits, c_mask, cand_mask, match, tau: float = 1.0):
    """logits (B,C,L), c_mask (B,C,L), cand_mask (B,C) 候補の有無, match (B,C) 実受理長。
    L_accept_length: ordinal BCE / L_rank: listwise CE (target ∝ exp(match/tau))"""
    B, C, L = logits.shape
    k = torch.arange(1, L + 1, device=logits.device)
    y = (match[..., None] >= k).float()
    m = c_mask & cand_mask[..., None]
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    l_ord = (bce * m).sum() / m.sum().clamp(min=1)
    e = BlockScorer.expected_len(logits, c_mask)
    e = e.masked_fill(~cand_mask, -1e4)
    tgt = torch.softmax((match.float() / tau).masked_fill(~cand_mask, -1e4), -1)
    l_rank = -(tgt * torch.log_softmax(e, -1)).sum(-1)
    valid = cand_mask.sum(-1) > 1
    l_rank = (l_rank * valid).sum() / valid.sum().clamp(min=1)
    return l_ord, l_rank
