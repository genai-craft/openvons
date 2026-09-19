"""Phase 4 実装順 3〜4: verifier + 有限候補 source (+ DFlash draft) + scorer + threshold controller の実測。

比較する構成 (同一 prompt, greedy, batch=1, HF eager):
  baseline        通常 decode
  finite/prior    有限候補 (ngram+corpus+schema|macro+repo) を prior top-1 で選ぶ
  finite/scorer   同上を scorer で選び、期待受理長 < th なら通常 decode
  dflash          DFlash draft の argmax chain のみ (公式 dflash_generate ではなく同じ verifier 経路で)
  union/scorer    有限候補 + DFlash 候補 (chain + 先頭差し替え) を scorer で選ぶ
  hybrid          有限候補を scorer で採点し、期待受理長 >= th_hi ならそれを使う (draft を呼ばない)。
                  それ未満のときだけ DFlash draft を呼び、union を再採点する (複合 controller)
hidden state は verifier の forward から取る (受理位置の最終確定 token を生んだ位置)。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from openvons.jevpick.paths import DATA  # noqa: E402
from openvons.jevpick.candidates.base import Candidate  # noqa: E402
from openvons.jevpick.candidates.grammar import build_grammar_index  # noqa: E402
from openvons.jevpick.candidates.macro_copy import build_macro_index  # noqa: E402
from openvons.jevpick.candidates.ngram import SuffixIndex, rank  # noqa: E402
from openvons.jevpick.candidates.repository import build_repo_index  # noqa: E402
from openvons.jevpick.candidates.tool_schema import build_schema_index, detect_format, tools_from_prompt_text, user_text_from_prompt_text  # noqa: E402
from openvons.jevpick.scorer.model import SOURCES, BlockScorer  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from experiments.jevpick.phase2_scorer.train import prior_feats  # noqa: E402

MAX_N = 6
SEP = 151643
SITE = Path(sys.prefix) / "lib/python3.12/site-packages"
SRC_ID = {s: i + 1 for i, s in enumerate(SOURCES)}


class Runtime:
    def __init__(self, target, tok, dev, layer_idx: int, draft=None, scorer=None, scorer_L=8, mtp=None):
        self.target, self.tok, self.dev = target, tok, dev
        self.mtp = mtp  # Qwen35MTP (checkpoint 同梱 MTP head) or None
        self.layer_idx = layer_idx  # hidden_states のインデックス
        self.draft = draft
        self.scorer = scorer
        self.scorer_L = scorer_L
        self.emb = target.get_input_embeddings().weight
        self.timing = {}
        self.n_skip_draft = 0

    def _score(self, cands, h_last, x):
        C, Ls = len(cands), self.scorer_L
        tokm = torch.zeros(1, C, Ls, dtype=torch.long, device=self.dev)
        cm = torch.zeros(1, C, Ls, dtype=torch.bool, device=self.dev)
        src = torch.zeros(1, C, dtype=torch.long, device=self.dev)
        pr = torch.zeros(1, C, 3, device=self.dev)
        for i, c in enumerate(cands):
            n = min(len(c.token_ids), Ls)
            tokm[0, i, :n] = torch.tensor(c.token_ids[:n], device=self.dev)
            cm[0, i, :n] = True
            src[0, i] = SRC_ID.get(c.source, 0)
            pr[0, i] = torch.tensor(prior_feats(c.source, c.prior_score), device=self.dev)
        logits = self.scorer(h_last.float(), self.emb[x][None].float(), self.emb[tokm].float(), cm, src, pr)
        return BlockScorer.expected_len(logits, cm)[0]

    def _t(self, key, t0):
        torch.cuda.synchronize()
        self.timing[key] = self.timing.get(key, 0.0) + time.perf_counter() - t0

    @torch.inference_mode()
    def generate(self, prompt_ids, max_new, stop_ids, mode: str, L: int, sources: dict, th: float = 0.0, th_hi: float = 4.0):
        """mode: baseline / prior / scorer / dflash / union"""
        use_finite = mode in ("prior", "scorer", "union", "hybrid", "union_all", "hybrid_all", "union_mtp")
        use_draft = mode in ("dflash", "union", "hybrid", "union_all", "hybrid_all")
        use_mtp = mode in ("mtp", "union_all", "hybrid_all", "union_mtp") and self.mtp is not None
        use_scorer = mode in ("scorer", "union", "hybrid", "union_all", "hybrid_all", "union_mtp")
        is_hybrid = mode in ("hybrid", "hybrid_all")
        need_hidden = use_draft or use_scorer or use_mtp
        cache = DynamicCache(config=self.target.config)
        if hasattr(cache, "activate_past_recording"):
            cache.activate_past_recording()
        P = len(prompt_ids)
        ctx = SuffixIndex(MAX_N)
        ctx.extend(prompt_ids[:-1])
        t0 = time.perf_counter()
        ids = torch.tensor([prompt_ids[:-1]], device=self.dev)
        o = self.target(input_ids=ids, past_key_values=cache, use_cache=True, output_hidden_states=need_hidden)
        if use_draft:
            from dflash.model import extract_context_feature
            th_all = extract_context_feature(o.hidden_states, self.draft.target_layer_ids)
            dcache = DynamicCache(config=self.draft.config)
            if hasattr(dcache, "activate_past_recording"):
                dcache.activate_past_recording()
            d_prev = 0
        h_last = o.hidden_states[self.layer_idx][:, -1] if need_hidden else None  # x を生んだ位置の hidden
        if use_mtp:
            mcache = self.mtp.new_cache()
            self.mtp.prefill(o.hidden_states[-1], prompt_ids, P - 2, mcache)  # index 1..P-2
            hf_last = o.hidden_states[-1][:, -1:]  # MTP 入力用 (最終層 post-norm) の h_{s-1}
        self._t("prefill", t0)
        x = prompt_ids[-1]
        out: list[int] = []
        accepted = []
        full = list(prompt_ids)
        while len(out) < max_new:
            draft = ()
            cands: list[Candidate] = []
            start = len(full) - 1  # x の位置
            if use_finite:
                t0 = time.perf_counter()
                suffix = full[-MAX_N:]
                cands += [c for c in ctx.blocks(ctx.occurrences(suffix), L, "ngram")]
                for name, idx in sources.items():
                    occ = idx.occurrences_excluding(suffix, getattr(idx, "own_path", None)) if name == "repo" else idx.occurrences(suffix)
                    cands += idx.blocks(occ, L, name)
                cands = [Candidate(c.token_ids[: c.token_ids.index(SEP)], c.source, c.prior_score) if SEP in c.token_ids else c for c in cands]
                cands = [c for c in cands if c.token_ids]
                cands = rank(cands, 16)
                self._t("candidates", t0)
            skip_draft = False
            if is_hybrid and cands:
                t0 = time.perf_counter()
                e = self._score(cands, h_last, x)
                best = int(e.argmax())
                self._t("scorer", t0)
                if float(e[best]) >= th_hi:
                    draft = cands[best].token_ids[: self.scorer_L]
                    skip_draft = True
                    self.n_skip_draft += 1
            if use_draft and not skip_draft:
                t0 = time.perf_counter()
                B = self.draft.block_size
                block = torch.full((1, B), self.draft.mask_token_id, dtype=torch.long, device=self.dev)
                block[0, 0] = x
                pos = torch.arange(d_prev, start + B, device=self.dev)[None]
                dh = self.draft(target_hidden=th_all[:, d_prev:start], noise_embedding=F.embedding(block, self.emb),
                                position_ids=pos, past_key_values=dcache, use_cache=True)[:, 1:, :]
                dcache.crop(-B)
                d_prev = start
                logits_d = self.draft.compute_logits(dh, self.target.lm_head).float()
                lp = torch.log_softmax(logits_d, -1)
                top = torch.topk(lp[0, : L], 4 if use_scorer else 1, dim=-1)
                ti, tv = top.indices.tolist(), top.values.tolist()
                base = [r[0] for r in ti]
                if hasattr(self.draft, "propose"):  # DFlash2: candidate selector の経路を chain0 にする
                    path = self.draft.candidate_selector.select(dh, logits_d, block[:, 0], 0.0)[0][0, :L].tolist()
                    base = path
                for k in range(len(ti[0])):
                    chain = list(base)
                    chain[0] = ti[0][k]
                    if k > 0 and chain[0] == base[0]:
                        continue
                    chain = tuple(chain)
                    if any(c.token_ids == chain for c in cands):
                        continue
                    cands.append(Candidate(chain, "dflash", (tv[0][k] + sum(r[0] for r in tv[1:]),)))
                self._t("draft", t0)
            if use_mtp and not skip_draft:
                t0 = time.perf_counter()
                mt, ml = self.mtp.draft(hf_last, x, start, L, mcache)  # cache に index start の step-1 が残る
                mt, ml = mt.tolist(), ml.tolist()
                base = [r[0] for r in mt]
                for k in range(4 if use_scorer else 1):
                    chain = list(base); chain[0] = mt[0][k]
                    if k > 0 and chain[0] == base[0]:
                        continue
                    chain = tuple(chain)
                    if any(c.token_ids == chain for c in cands):
                        continue
                    cands.append(Candidate(chain, "mtp", (ml[0][k] + sum(r[0] for r in ml[1:]),)))
                self._t("mtp", t0)
                if mode == "mtp":
                    cands = [c for c in cands if c.source == "mtp"]
            if cands and not skip_draft:
                if use_scorer:
                    t0 = time.perf_counter()
                    e = self._score(cands, h_last, x)
                    best = int(e.argmax())
                    if float(e[best]) >= th:
                        draft = cands[best].token_ids[: self.scorer_L]
                    self._t("scorer", t0)
                elif mode in ("dflash", "mtp"):
                    draft = cands[0].token_ids
                else:
                    draft = cands[0].token_ids
            # verify
            t0 = time.perf_counter()
            base_len = cache.get_seq_length()
            toks = [x, *draft]
            ids = torch.tensor([toks], device=self.dev)
            pos = torch.arange(base_len, base_len + len(toks), device=self.dev)[None]
            o = self.target(input_ids=ids, past_key_values=cache, position_ids=pos, use_cache=True, output_hidden_states=need_hidden)
            y = o.logits[0].argmax(-1).tolist()
            a = 0
            while a < len(draft) and draft[a] == y[a]:
                a += 1
            new = list(draft[:a]) + [y[a]]
            if a < len(draft):
                cache.crop(-(len(draft) - a))  # 負の crop (linear attention 層は負のみ対応)
            if need_hidden:
                h_last = o.hidden_states[self.layer_idx][:, a]
                if use_mtp:
                    hs_final = o.hidden_states[-1]
                    if not (use_mtp and not skip_draft):
                        # この step で MTP を呼んでいない → index start の step-1 も未計算
                        self.mtp.step(hf_last, torch.tensor([[x]], device=self.dev), torch.tensor([[start]], device=self.dev), mcache)
                    if a > 0:
                        toks_c = torch.tensor([list(draft[:a])], device=self.dev)
                        pos_c = torch.arange(start + 1, start + 1 + a, device=self.dev)[None]
                        self.mtp.step(hs_final[:, :a], toks_c, pos_c, mcache)
                    hf_last = hs_final[:, a : a + 1]
                if use_draft:
                    th_all = torch.cat([th_all, extract_context_feature(o.hidden_states, self.draft.target_layer_ids)[:, : a + 1]], 1)
            self._t("verify", t0)
            accepted.append(a)
            ctx.extend([x])
            committed = []
            for tkn in new:
                out.append(tkn); committed.append(tkn)
                if tkn in stop_ids or len(out) >= max_new:
                    break
            full += committed
            if out[-1] in stop_ids or len(out) >= max_new:
                break
            ctx.extend(committed[:-1])
            x = committed[-1]
            if len(committed) < len(new):  # stop で途中終了
                break
        return out, accepted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--draft", default="z-lab/Qwen3-4B-DFlash-b16")
    ap.add_argument("--scorer", default=None, help="train.py --save の出力 (無ければ scorer/union モード不可)")
    ap.add_argument("--prompts", default=f"{DATA}/prompts_v2.jsonl")
    ap.add_argument("--traces", default=f"{DATA}/traces_v2_Qwen3-4B.jsonl")
    ap.add_argument("--domain", default="toolcall")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--L", type=int, default=8)
    ap.add_argument("--th", type=float, default=0.5)
    ap.add_argument("--th-hi", type=float, default=4.0, help="hybrid: 有限候補の期待受理長がこれ以上なら draft を呼ばない")
    ap.add_argument("--modes", default="baseline,prior,scorer,dflash,union")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--quant", default="", help="fp8 / nf4")
    ap.add_argument("--mtp", action="store_true", help="checkpoint 同梱 MTP head (Qwen3.5 系) を draft source に使う")
    args = ap.parse_args()
    dev = "cuda:0"
    tok = AutoTokenizer.from_pretrained(args.model)
    if args.quant == "fp8":
        from transformers import FineGrainedFP8Config
        qc = FineGrainedFP8Config(modules_to_not_convert=["in_proj_a", "in_proj_b", "lm_head", "conv1d", "norm"])
        target = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa", quantization_config=qc, device_map=dev).eval()
    elif args.quant == "nf4":
        from transformers import BitsAndBytesConfig
        qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
                                llm_int8_skip_modules=["lm_head", "in_proj_a", "in_proj_b"])
        target = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa", quantization_config=qc, device_map=dev).eval()
    else:
        target = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype), attn_implementation="sdpa").to(dev).eval()
    from transformers import AutoConfig
    from dflash.model import DFlash2DraftModel, DFlashDraftModel
    dcfg = AutoConfig.from_pretrained(args.draft)
    dcls = DFlash2DraftModel if "DFlash2DraftModel" in (dcfg.architectures or []) else DFlashDraftModel
    draft = dcls.from_pretrained(args.draft, attn_implementation="sdpa", dtype=torch.bfloat16).to(dev).eval()
    tcfg = getattr(target.config, "text_config", None) or target.config
    nL = tcfg.num_hidden_layers
    LAYER_IDX = tuple(round(nL * f) for f in (0.25, 0.5, 0.75, 1.0))
    scorer, layer_idx, sL = None, LAYER_IDX[-1], args.L
    if args.scorer:
        ck = torch.load(args.scorer, map_location=dev)
        cfg = ck["config"]
        scorer = BlockScorer(tcfg.hidden_size, encoder=cfg["encoder"], max_len=cfg["L"]).to(dev).eval()
        scorer.load_state_dict(ck["state"])
        layer_idx, sL = LAYER_IDX[cfg["layer"]], cfg["L"]
    mtp = None
    if args.mtp:
        from huggingface_hub import snapshot_download
        from openvons.jevpick.candidates.mtp_qwen35 import Qwen35MTP
        mtp = Qwen35MTP(target, snapshot_download(args.model, allow_patterns=["*.json"])).to(dev).eval()
        print("MTP head loaded", flush=True)
    rt = Runtime(target, tok, dev, layer_idx, draft, scorer, sL, mtp)
    stop = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
    prompts = [json.loads(l) for l in open(args.prompts)]
    test = [p for p in prompts if p["domain"] == args.domain and p.get("split") == "test"][: args.n]
    traces = {json.loads(l)["sample_id"]: json.loads(l) for l in open(args.traces)}
    train_ids = {p["sample_id"] for p in prompts if p["domain"] == args.domain and p.get("split") == "train"}
    corpus = SuffixIndex(MAX_N, max_positions=128)
    for sid in train_ids:
        if sid in traces:
            corpus.extend(traces[sid]["output_ids"] + [SEP])
    grammar = build_grammar_index(tok, args.domain, MAX_N)
    modes = args.modes.split(",")
    res = {m: {"tok": 0, "sec": 0.0, "acc": [], "match": 0, "timing": {}} for m in modes}
    for i, p in enumerate(test):
        kw = {"tools": p["tools"]} if p["tools"] else {}
        text = tok.apply_chat_template(p["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False, **kw)
        pid = tok(text, add_special_tokens=False).input_ids
        srcs = {"corpus": corpus, "grammar": grammar}
        if args.domain == "toolcall":
            srcs["schema"] = build_schema_index(tok, tools_from_prompt_text(text), user_text_from_prompt_text(text), MAX_N, detect_format(text))
        else:
            srcs["macro"] = build_macro_index(tok, text, MAX_N)
            meta = p.get("meta") or {}
            if meta.get("repo"):
                srcs["repo"] = build_repo_index(tok, str(SITE / meta["repo"]), 150, 30000, MAX_N)
                srcs["repo"].own_path = meta["path"]
        base_out = None
        for m in modes:
            rt.timing = {}
            torch.cuda.synchronize(); t0 = time.perf_counter()
            rt.n_skip_draft = 0
            out, acc = rt.generate(pid, args.max_new, stop, m, args.L, srcs, args.th, args.th_hi)
            torch.cuda.synchronize(); dt = time.perf_counter() - t0
            if m == "baseline":
                base_out = out
            if i == 0:
                continue  # warm-up
            r = res[m]
            r["tok"] += len(out); r["sec"] += dt; r["acc"] += acc; r["skip_draft"] = r.get("skip_draft", 0) + rt.n_skip_draft
            r["match"] += int(out == base_out)
            for k, v in rt.timing.items():
                r["timing"][k] = r["timing"].get(k, 0.0) + v
        if (i + 1) % 10 == 0:
            print(i + 1, {m: round(res[m]["tok"] / max(res[m]["sec"], 1e-9), 1) for m in modes}, flush=True)
    n = len(test) - 1
    summ = {}
    base = res["baseline"]["tok"] / res["baseline"]["sec"]
    # decode 専用 (prefill を除く, §17.3)
    base_dec = res["baseline"]["tok"] / (res["baseline"]["sec"] - res["baseline"]["timing"].get("prefill", 0.0))
    for m in modes:
        r = res[m]
        acc = np.array(r["acc"]) if r["acc"] else np.zeros(1)
        dec = r["tok"] / (r["sec"] - r["timing"].get("prefill", 0.0))
        summ[m] = {"tok_s": r["tok"] / r["sec"], "speedup": (r["tok"] / r["sec"]) / base, "exact_match": r["match"] / n,
                   "decode_tok_s": dec, "decode_speedup": dec / base_dec,
                   "tok_per_step": r["tok"] / len(acc), "mean_accepted": float(acc.mean()), "zero_accept": float((acc == 0).mean()),
                   "time_frac": {k: v / r["sec"] for k, v in r["timing"].items()}, "draft_skip_frac": r.get("skip_draft", 0) / len(acc)}
    print(json.dumps(summ, indent=1))
    out = Path(args.out or f"experiments/jevpick/phase4_runtime/runtime_v2_{args.domain}_L{args.L}_{args.dtype}{args.tag}.json")
    json.dump({"config": vars(args), "n": n, "summary": summ, "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9}, open(out, "w"), indent=1)
    print("peak VRAM %.1f GB" % (torch.cuda.max_memory_allocated() / 1e9))
    print("->", out)


if __name__ == "__main__":
    main()
