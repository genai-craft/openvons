"""河川ライブカメラのカタログを作る (関東地方整備局のライブカメラ一覧を加工、PDL1.0).

    .venv/bin/python scripts/build_kasen.py --src <crawl.json>   # -> examples/kasen/{cameras,hierarchy}.json

入力: [{"name","river","office","pref","lat","lng","image_url","page_url","interval_min","notes"}]
読み: 地点名から「水位観測所 / 観測所 / 排水機場 / 付近 / カメラ / 上流 / 下流」等を外した地名部分を別名にし、
読みは pyopenjtalk の G2P (固有名詞は誤読しうるので、事前学習の実現読みと人手修正で補う)。
order: 同じ河川内の並び (一覧の掲載順 = おおむね上流→下流) を上流/下流の移動に使う。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from openvons.voice import kana as K  # noqa: E402

STRIP = re.compile(r"(水位観測所|観測所|排水機場|排水樋管|樋管|水門|堰|付近|カメラ|地点|局|上流|下流|左岸|右岸|（.*?）|\(.*?\))")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default=str(ROOT / "examples" / "kasen"))
    args = ap.parse_args()
    rows = json.load(open(args.src, encoding="utf-8"))
    ents = []
    per_river: dict[str, int] = defaultdict(int)
    hier: dict[str, dict[str, dict]] = defaultdict(dict)
    seen_labels: dict[str, int] = defaultdict(int)
    for r in rows:
        name = (r.get("name") or "").strip()
        if not name:
            continue
        river = (r.get("river") or "その他").strip()
        office = (r.get("office") or "関東地方整備局").strip()
        label = re.sub(r"\s+", "", name)
        seen_labels[label] += 1
        if seen_labels[label] > 1:
            label = f"{label}({seen_labels[label]})"
        core = STRIP.sub("", name).strip() or name
        readings = [K.g2p(label)]
        aliases = [core] if core != label else []
        order = per_river[river]; per_river[river] += 1
        ents.append({
            "id": f"kasen_{len(ents):04d}", "label": label, "kind": "camera", "readings": readings, "aliases": aliases,
            "attrs": {"river": river, "office": office, "pref": r.get("pref") or "", "lat": r.get("lat"), "lng": r.get("lng"),
                      "image_url": r.get("image_url"), "page_url": r.get("page_url"), "interval_min": r.get("interval_min"),
                      "order": order, "route_label": river, "core": core},
        })
        h = hier[office].setdefault(river, {"n_cameras": 0, "prefs": set()})
        h["n_cameras"] += 1
        if r.get("pref"):
            h["prefs"].add(r["pref"])
    for office in hier:
        for river, v in hier[office].items():
            v["prefs"] = sorted(v["prefs"]); v["code"] = river
    out = Path(args.out)
    (out / "cameras.json").write_text(json.dumps(ents, ensure_ascii=False, indent=0), encoding="utf-8")
    (out / "hierarchy.json").write_text(json.dumps(hier, ensure_ascii=False, indent=1), encoding="utf-8")
    n_img = sum(1 for e in ents if e["attrs"]["image_url"]); n_geo = sum(1 for e in ents if e["attrs"]["lat"])
    print(f"cameras {len(ents)}  with image {n_img}  with coords {n_geo}  offices {len(hier)}  rivers {sum(len(v) for v in hier.values())}")
    for e in ents[:5]:
        print(" ", e["label"], e["readings"], e["aliases"], e["attrs"]["river"])


if __name__ == "__main__":
    main()
