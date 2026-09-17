"""PA-100K (監視カメラの歩行者全身画像 10 万枚 + 属性) を vision decision 形式に変換する。

質問: gender (2) / age (3区分) / orientation (前・横・後) / carrying (手ぶらか荷物か)
FairFace (顔のみ) との対比で「全身の外観だけで年代・性別がどこまで判るか」を見る。
"""
from __future__ import annotations

import io, json, os, sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/data/decision_model/hf_home")
from huggingface_hub import hf_hub_download
import pandas as pd
from PIL import Image

REPO = "tuandunghcmut/PA-100K"
OUT = Path("/data/decision_model/data/vision/pa100k")


def age_label(r) -> int:
    if int(r["AgeLess18"]): return 0
    if int(r["AgeOver60"]): return 2
    return 1


def orient_label(r) -> int:
    if int(r["Front"]): return 0
    if int(r["Side"]): return 1
    return 2


def carry_label(r) -> int:
    return 1 if (int(r["HandBag"]) or int(r["ShoulderBag"]) or int(r["Backpack"]) or int(r["HoldObjectsInFront"])) else 0


def main():
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    for split, fn in [("train", "data/train-00000-of-00001.parquet"),
                      ("valid", "data/val-00000-of-00001.parquet"),
                      ("test", "data/test-00000-of-00001.parquet")]:
        df = pd.read_parquet(hf_hub_download(REPO, fn, repo_type="dataset"))
        if limit and split == "train":
            df = df.iloc[:limit]
        recs = []
        for k, (_, r) in enumerate(df.iterrows()):
            img = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
            name = f"{split}_{k:06d}.jpg"
            img.save(OUT / "images" / name, quality=92)
            recs.append({"id": f"{split}_{k:06d}", "image": f"images/{name}",
                         "labels": {"gender": int(r["Female"]), "age": age_label(r),
                                    "orientation": orient_label(r), "carrying": carry_label(r)},
                         "meta": {"size": list(img.size)}})
            if k % 10000 == 0:
                print(split, k, flush=True)
        (OUT / f"{split}.jsonl").write_text("\n".join(json.dumps(x) for x in recs))
        import collections
        print(f"{split}: {len(recs)} "
              f"gender={dict(collections.Counter(x['labels']['gender'] for x in recs))} "
              f"age={dict(sorted(collections.Counter(x['labels']['age'] for x in recs).items()))} "
              f"orient={dict(sorted(collections.Counter(x['labels']['orientation'] for x in recs).items()))}", flush=True)
    json.dump({"questions": {
        "gender": {"type": "choice", "question": "What is the apparent gender of this pedestrian?",
                   "choices": [{"id": "male", "description": "male"}, {"id": "female", "description": "female"}]},
        "age": {"type": "choice", "question": "What is the apparent age group of this pedestrian?",
                "choices": [{"id": "under18", "description": "a child or teenager, under 18"},
                            {"id": "adult", "description": "an adult between 18 and 60"},
                            {"id": "over60", "description": "an elderly person, over 60"}]},
        "orientation": {"type": "choice", "question": "Which way is this pedestrian facing relative to the camera?",
                        "choices": [{"id": "front", "description": "facing the camera"},
                                    {"id": "side", "description": "seen from the side"},
                                    {"id": "back", "description": "facing away from the camera"}]},
        "carrying": {"type": "choice", "question": "Is this pedestrian carrying a bag or holding an object?",
                     "choices": [{"id": "no", "description": "empty handed"},
                                 {"id": "yes", "description": "carrying a bag or holding something"}]}}},
        open(OUT / "tasks.json", "w"), indent=2, ensure_ascii=False)
    print("done", flush=True)


if __name__ == "__main__":
    main()
