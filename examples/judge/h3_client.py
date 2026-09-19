"""ComfyUI (MiniMax H3) クラスタでテキスト→動画を作る小さなクライアント。

mojidance/server の workflows.js (buildH3Workflow, t2v) を Python に写したもの。クラスタは VPN 内 (192.168.100.x:28000) なので
踏み台への SSH の SOCKS プロキシ (ssh -D 1080) 経由で叩く: 環境変数 JUDGE_SOCKS=socks5h://127.0.0.1:1080。
"""
from __future__ import annotations

import copy
import json
import os
import random
import time
from pathlib import Path

import requests

LIB = Path(os.environ.get("H3_LIB", str(Path.home() / "mojidance/server/lib")))
SERVERS = [f"http://192.168.100.{h}:28000" for h in (11, 12, 15, 16, 17, 18, 19, 21, 22, 23, 113, 114, 115)]
TURBO_LORA = "minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_resized_avg_rank_64_bf16.safetensors"
FAST_VAE = "minimax_h3_video_vae_int8_convrot.safetensors"
ASPECT = {"16:9": "16:9 (Widescreen)", "9:16": "9:16 (Portrait Widescreen)", "1:1": "1:1 (Square)"}


def session():
    s = requests.Session()
    p = os.environ.get("JUDGE_SOCKS", "socks5h://127.0.0.1:1080")
    if p:
        s.proxies = {"http": p, "https": p}
    return s


def _by_class(g, cls):
    return [(k, v) for k, v in g.items() if v.get("class_type") == cls]


def build_t2v(prompt: str, *, seconds: float = 15, aspect: str = "16:9", megapixels: float = 0.4, seed: int | None = None,
              job: str = "judge", turbo: bool = True, steps: int = 8, fast_vae: bool = True) -> dict:
    g = copy.deepcopy(json.load(open(LIB / "video_minimax_h3_t2v.json")))
    if fast_vae:
        for _, n in _by_class(g, "VAELoader"):
            if "video" in n["inputs"].get("vae_name", ""):
                n["inputs"]["vae_name"] = FAST_VAE
    rs = _by_class(g, "ResolutionSelector")[0][1]
    rs["inputs"]["aspect_ratio"] = ASPECT[aspect]
    rs["inputs"]["megapixels"] = megapixels
    _by_class(g, "PrimitiveFloat")[0][1]["inputs"]["value"] = seconds
    _by_class(g, "RandomNoise")[0][1]["inputs"]["noise_seed"] = seed if seed is not None else random.getrandbits(52)
    if turbo:
        unet_id = _by_class(g, "UNETLoader")[0][0]
        g["910"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": [unet_id, 0], "lora_name": TURBO_LORA, "strength_model": 1.0}}
        for k, n in g.items():
            if k == "910":
                continue
            for a, v in list(n.get("inputs", {}).items()):
                if isinstance(v, list) and v and v[0] == unet_id and a == "model":
                    n["inputs"][a] = ["910", 0]
        g["911"] = {"class_type": "H3SLAAttention", "inputs": {"model": ["910", 0], "sparsity_ratio": 0.9, "block_size": "64", "min_seq_len": 4096,
                    "dense_last_steps": 1, "protect_audio": True, "enabled": True, "dense_steps": "0", "dense_backend": "comfy_kitchen",
                    "disable_fp16_accum": True, "stabilize_motion": True}}
        for k, n in g.items():
            if k == "911":
                continue
            for a, v in list(n.get("inputs", {}).items()):
                if isinstance(v, list) and v and v[0] == "910" and a == "model":
                    n["inputs"][a] = ["911", 0]
        _by_class(g, "BasicScheduler")[0][1]["inputs"]["steps"] = steps
    _by_class(g, "SaveVideo")[0][1]["inputs"]["filename_prefix"] = f"judge/{job}"
    _by_class(g, "MiniMaxH3ImageToVideo")[0][1]["inputs"]["prompt"] = prompt
    return g


class Cluster:
    def __init__(self):
        self.s = session()

    def queue_len(self, url):
        try:
            q = self.s.get(f"{url}/queue", timeout=8).json()
            return len(q.get("queue_running", [])) + len(q.get("queue_pending", []))
        except Exception:  # noqa: BLE001
            return None

    def alive(self):
        out = []
        for u in SERVERS:
            n = self.queue_len(u)
            if n is not None:
                out.append((n, u))
        return sorted(out)

    def has_vae(self, url, name):
        try:
            j = self.s.get(f"{url}/object_info/VAELoader", timeout=8).json()
            return name in (j["VAELoader"]["input"]["required"]["vae_name"][0] or [])
        except Exception:  # noqa: BLE001
            return False

    def submit(self, url, graph):
        r = self.s.post(f"{url}/prompt", json={"prompt": graph, "client_id": "openvons-judge"}, timeout=30)
        r.raise_for_status()
        j = r.json()
        if "prompt_id" not in j:
            raise RuntimeError(f"submit failed: {str(j)[:300]}")
        return j["prompt_id"]

    def wait(self, url, pid, timeout=3600, poll=5):
        t0 = time.time()
        while time.time() - t0 < timeout:
            h = self.s.get(f"{url}/history/{pid}", timeout=15).json()
            e = h.get(pid)
            if e:
                st = e.get("status", {})
                if st.get("status_str") == "error":
                    raise RuntimeError("comfy error: " + json.dumps(st.get("messages", []))[:500])
                files = []
                for o in e.get("outputs", {}).values():
                    for key in ("images", "videos", "gifs", "video", "files"):
                        for f in o.get(key, []) or []:
                            if isinstance(f, dict) and f.get("filename") and f.get("type", "output") == "output":
                                files.append(f)
                if files:
                    return files
            time.sleep(poll)
        raise TimeoutError("comfy job timed out")

    def download(self, url, f, out: Path):
        r = self.s.get(f"{url}/view", params={"filename": f["filename"], "subfolder": f.get("subfolder", ""), "type": f.get("type", "output")}, timeout=300)
        r.raise_for_status()
        out.write_bytes(r.content)
        return out
