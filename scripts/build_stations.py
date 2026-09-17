"""駅名デモのカタログを作る (Seo-4d696b75/station_database、CC BY 4.0、駅データ.jp 由来).

    .venv/bin/python scripts/build_stations.py   # /data/openjev/data/stations/{station,line,register}.csv -> examples/stations/{stations,hierarchy}.json

実体 = 駅。読みは name_kana (ひらがな) を ASR 形カタカナに正規化。属性: 都道府県、路線 (複数)、事業者、座標。
担当範囲 (Scope) は路線単位で作るので、路線ごとの駅 index (順序) も持たせる (「次の駅 / 前の駅」用)。
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jev.voice import kana as K  # noqa: E402

SRC = Path("/data/openjev/data/stations")
PREF = ["", "北海道", "青森県", "岩手県", "宮城県", "秋田県", "山形県", "福島県", "茨城県", "栃木県", "群馬県", "埼玉県", "千葉県", "東京都", "神奈川県", "新潟県", "富山県", "石川県", "福井県", "山梨県", "長野県", "岐阜県", "静岡県", "愛知県", "三重県", "滋賀県", "京都府", "大阪府", "兵庫県", "奈良県", "和歌山県", "鳥取県", "島根県", "岡山県", "広島県", "山口県", "徳島県", "香川県", "愛媛県", "高知県", "福岡県", "佐賀県", "長崎県", "熊本県", "大分県", "宮崎県", "鹿児島県", "沖縄県"]
COMPANY = {1: "JR北海道", 2: "JR東日本", 3: "JR東海", 4: "JR西日本", 5: "JR四国", 6: "JR九州"}   # 他は路線名の頭で補う


def main() -> None:
    stations = {r["code"]: r for r in csv.DictReader(open(SRC / "station.csv", encoding="utf-8")) if r["closed"] == "0"}
    lines = {r["code"]: r for r in csv.DictReader(open(SRC / "line.csv", encoding="utf-8")) if r["closed"] == "0"}
    members: dict[str, list[tuple[int, str]]] = defaultdict(list)     # line -> [(index, station_code)]
    for r in csv.DictReader(open(SRC / "register.csv", encoding="utf-8")):
        if r["line_code"] in lines and r["station_code"] in stations:
            members[r["line_code"]].append((int(r["index"]), r["station_code"]))
    st_lines: dict[str, list[dict]] = defaultdict(list)
    for lc, lst in members.items():
        lst.sort()
        for i, (_, sc) in enumerate(lst):
            st_lines[sc].append({"line": lines[lc]["name"], "line_code": lc, "index": i, "n": len(lst)})

    def company_of(line: dict) -> str:
        cc = int(line["company_code"]) if line["company_code"] not in ("", "NULL") else 0
        if cc in COMPANY:
            return COMPANY[cc]
        n = line["name"]
        for key in ("東京メトロ", "都営", "東急", "京急", "京王", "小田急", "西武", "東武", "京成", "相鉄", "近鉄", "阪急", "阪神", "京阪", "南海", "名鉄", "西鉄", "つくばエクスプレス", "りんかい", "ゆりかもめ", "横浜市営", "大阪メトロ", "名古屋市営", "京都市営", "神戸市営", "札幌市営", "仙台市", "福岡市"):
            if n.startswith(key):
                return key
        return "その他"

    entities = []
    for sc, s in stations.items():
        if sc not in st_lines:
            continue
        pref = PREF[int(s["prefecture"])]
        reading = K.normalize(s["name_kana"])
        ls = st_lines[sc]
        companies = sorted({company_of(lines[l["line_code"]]) for l in ls})
        label = s["original_name"] if s.get("original_name") and s["original_name"] != "NULL" else s["name"]
        entities.append({
            "id": f"st_{sc}", "label": label, "kind": "station",
            "readings": [reading], "aliases": [],
            "attrs": {"display": s["name"], "pref": pref, "lines": [l["line"] for l in ls], "line_codes": [l["line_code"] for l in ls], "positions": ls,
                      "companies": companies, "company": companies[0], "lat": float(s["lat"]), "lng": float(s["lng"]),
                      "route_label": " / ".join(l["line"] for l in ls[:2]) + (" ほか" if len(ls) > 2 else "")},
        })
    # 「〜駅」も別名として読みに入れる (「新宿駅まで」)
    for e in entities:
        e["readings"].append(K.normalize(e["readings"][0] + "エキ"))
    hier: dict[str, dict[str, dict]] = defaultdict(dict)
    for lc, l in lines.items():
        if lc not in members:
            continue
        prefs = sorted({PREF[int(stations[sc]["prefecture"])] for _, sc in members[lc]})
        hier[company_of(l)][l["name"]] = {"code": lc, "n_stations": len(members[lc]), "prefs": prefs, "kana": K.normalize(l["name_kana"]), "color": l["color"]}
    out = ROOT / "examples" / "stations"
    (out / "stations.json").write_text(json.dumps(entities, ensure_ascii=False, indent=0), encoding="utf-8")
    (out / "hierarchy.json").write_text(json.dumps(hier, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"stations: {len(entities)}  lines: {sum(len(v) for v in hier.values())}  companies: {len(hier)}")
    # 同名駅 (読みが同じ) の数 = 混同の温床。デモの見せ場なので数えておく
    from collections import Counter
    c = Counter(e["readings"][0] for e in entities)
    dup = {k: v for k, v in c.items() if v > 1}
    print("同読みの駅名:", len(dup), "例:", sorted(dup.items(), key=lambda kv: -kv[1])[:8])


if __name__ == "__main__":
    main()
