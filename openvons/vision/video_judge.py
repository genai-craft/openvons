"""動画の審判 (judge): 動画を窓に切り、各窓に有限選択肢の質問を VLM (Qwen3-VL) に投げて確率で答えさせる。

- 質問は Noul: 「〜が起きているか」→ A. yes / B. no / C. cannot tell。C が「該当なし・判別できない」の役。
- 1 窓の動画 token 列は 1 回だけ prefill し、KV cache を複製して質問ごとに Decision の 1 token だけ forward する
  (openvons.lm の kv_shared と同じ考え方)。質問を 10 個にしても動画の符号化は 1 回。
- 確率は温度 T で校正 (state/judge/calibration.json)。判断は openvons.core.decide の 3 段 (確定 / 要確認 / 不明)。
- 窓の fps と長さは項目ごと (安全・監視 2 fps × 4 s、スポーツ 10 fps × 1.6 s)。
- live(): 1 フレームずつ流す用途 (ロボットの進路: 左 / 右 / 直進 / 停止 など Choice 質問) の低遅延経路。
"""
from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image

OPTION_LETTERS = "ABCDEFGH"


@dataclass
class VQuestion:
    key: str
    text: str                      # 英語の質問
    options: list[str]             # 英語の選択肢 (最後を "cannot tell" 等の該当なしにする)
    none_index: int | None = None  # 該当なし選択肢の位置 (無ければ None)
    fps: float = 2.0
    window_s: float = 4.0
    risk: str = "low"
    labels_ja: list[str] = field(default_factory=list)


def read_frames(path: str, fps: float, max_s: float | None = None) -> tuple[list[Image.Image], list[float], float]:
    """動画を fps でサンプリング。戻り値 (frames, timestamps, duration)。"""
    import av
    frames, ts = [], []
    with av.open(path) as c:
        st = c.streams.video[0]
        st.thread_type = "AUTO"
        dur = float(st.duration * st.time_base) if st.duration else None
        step = 1.0 / fps
        nxt = 0.0
        for fr in c.decode(st):
            t = float(fr.pts * st.time_base) if fr.pts is not None else len(ts) * step
            if max_s is not None and t > max_s:
                break
            if t + 1e-6 >= nxt:
                im = fr.to_image()
                if im.width > 640:
                    im = im.resize((640, int(im.height * 640 / im.width)))
                frames.append(im); ts.append(t); nxt += step
        if dur is None:
            dur = ts[-1] + step if ts else 0.0
    return frames, ts, dur


class VideoJudge:
    def __init__(self, model: str = "Qwen/Qwen3-VL-4B-Instruct", device: str = "cuda", calibration: str | None = None):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.proc = AutoProcessor.from_pretrained(model)
        self.tok = self.proc.tokenizer
        self.model = AutoModelForImageTextToText.from_pretrained(model, dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
        self.dev = device
        self.letter_ids = [self.tok.encode(f" {L}", add_special_tokens=False)[-1] for L in OPTION_LETTERS]
        self.letter_ids_nospace = [self.tok.encode(L, add_special_tokens=False)[-1] for L in OPTION_LETTERS]
        self.T: dict[str, float] = {}
        self.want_hidden = False      # True なら prefill のたびに last_hidden (最終層 + 3/4 層の最終 token) を残す
        self.last_hidden = None
        if calibration and Path(calibration).exists():
            self.T = json.load(open(calibration)).get("temperature", {})

    # ---- prompt ----
    @staticmethod
    def qtext(q: VQuestion) -> str:
        opts = "\n".join(f"{OPTION_LETTERS[i]}. {o}" for i, o in enumerate(q.options))
        return f"Question: {q.text}\nOptions:\n{opts}\nDecision:"

    def _prefill(self, frames: list[Image.Image], state: str, video: bool = True):
        """動画 (フレーム列) + state を 1 回 forward して KV cache を返す。"""
        content = ([{"type": "video", "video": frames}] if video and len(frames) > 1 else [{"type": "image", "image": frames[0]}])
        content.append({"type": "text", "text": f"State:\n{state}\n"})
        msgs = [{"role": "user", "content": content}]
        kw = {"fps": 2.0} if video and len(frames) > 1 else {}
        inp = self.proc.apply_chat_template(msgs, add_generation_prompt=False, tokenize=True, return_dict=True, return_tensors="pt", **kw).to(self.dev)
        out = self.model(**inp, use_cache=True, output_hidden_states=self.want_hidden)
        if self.want_hidden:
            hs = out.hidden_states; L = len(hs) - 1
            self.last_hidden = torch.cat([hs[L][0, -1], hs[(3 * L) // 4][0, -1]]).float()
        return inp, out.past_key_values

    def _ask(self, inp, cache, q: VQuestion) -> np.ndarray:
        """prefill 済み cache を複製し、質問 + 'Decision:' を流して選択肢文字の logit を読む。"""
        text = self.qtext(q) + "<|im_end|>\n<|im_start|>assistant\n"
        ids = self.tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(self.dev)
        n_past = inp["input_ids"].shape[1]
        c = copy.deepcopy(cache)
        pos = torch.arange(n_past, n_past + ids.shape[1], device=self.dev)[None]
        out = self.model(input_ids=ids, past_key_values=c, use_cache=True, cache_position=pos)
        logits = out.logits[0, -1].float()
        n = len(q.options)
        l1 = logits[self.letter_ids[:n]]; l2 = logits[self.letter_ids_nospace[:n]]
        return torch.logsumexp(torch.stack([l1, l2]), 0).cpu().numpy()

    def _probs(self, key: str, logit: np.ndarray) -> np.ndarray:
        T = self.T.get(key, self.T.get("_default", 1.0))
        z = logit / T
        z = z - z.max()
        p = np.exp(z); return p / p.sum()

    # ---- 動画全体 ----
    @torch.inference_mode()
    def judge(self, path: str, questions: list[VQuestion], state: str = "Security camera footage.", max_s: float = 90.0, progress=None) -> dict:
        """質問群を fps/window でグループ化し、窓ごとに全質問を採点する。戻り値: 質問ごとの時系列と要約。"""
        t_all = time.perf_counter()
        groups: dict[tuple[float, float], list[VQuestion]] = {}
        for q in questions:
            groups.setdefault((q.fps, q.window_s), []).append(q)
        result = {"questions": {}, "timing": {"decode_s": 0.0, "encode_s": 0.0, "ask_s": 0.0, "windows": 0, "asks": 0}}
        duration = 0.0
        for (fps, win), qs in groups.items():
            t0 = time.perf_counter()
            frames, ts, duration = read_frames(path, fps, max_s)
            result["timing"]["decode_s"] += time.perf_counter() - t0
            n_win = max(1, int(round(win * fps)))
            step = max(1, n_win // 2)
            series = {q.key: [] for q in qs}
            for s in range(0, max(1, len(frames) - n_win + 1), step):
                fr = frames[s : s + n_win]
                t0 = time.perf_counter()
                inp, cache = self._prefill(fr, state)
                result["timing"]["encode_s"] += time.perf_counter() - t0
                t0 = time.perf_counter()
                hid = self.last_hidden.cpu().numpy() if self.want_hidden else None
                for q in qs:
                    lg = self._ask(inp, cache, q)
                    p = self._probs(q.key, lg)
                    rec = {"t0": ts[s], "t1": ts[min(s + n_win, len(ts)) - 1], "p": [float(x) for x in p], "logit": [float(x) for x in lg]}
                    if hid is not None:
                        rec["_hidden"] = hid
                    series[q.key].append(rec)
                result["timing"]["ask_s"] += time.perf_counter() - t0
                result["timing"]["windows"] += 1; result["timing"]["asks"] += len(qs)
                if progress:
                    progress(s + n_win, len(frames))
                del cache
            for q in qs:
                result["questions"][q.key] = {"series": series[q.key], "fps": fps, "window_s": win, "options": q.options, "labels_ja": q.labels_ja, "none_index": q.none_index, "risk": q.risk}
        result["duration"] = duration
        result["timing"]["total_s"] = time.perf_counter() - t_all
        return result

    # ---- ライブ (1 フレーム or 直近数フレーム) ----
    @torch.inference_mode()
    def live(self, frames: list[Image.Image], questions: list[VQuestion], state: str) -> dict:
        t0 = time.perf_counter()
        inp, cache = self._prefill(frames, state, video=len(frames) > 1)
        t1 = time.perf_counter()
        out = {}
        for q in questions:
            out[q.key] = [float(x) for x in self._probs(q.key, self._ask(inp, cache, q))]
        return {"probs": out, "timing": {"encode_ms": (t1 - t0) * 1e3, "ask_ms": (time.perf_counter() - t1) * 1e3}}
