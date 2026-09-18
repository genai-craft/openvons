"""駅名で動く路線図ビューア (一般向けデモ): 意図・状態・仮想ビューの操作.

状態:
  MAP      路線図全体。受理: 駅名 (寄る)、路線名で絞る、ヘルプ
  STATION  1 駅を選択中。受理: 次の駅 / 前の駅 / 別の駅名 / 路線を変えて / もっと寄って / 引いて / 一覧 (全体) に戻る / この駅を登録 (お気に入り、要確認)
  CONFIRM  確認待ち。受理: はい / いいえ

道路カメラのデモと同じ状態機械・同じ認識器。差し替えたのは実体 (駅) と意図の一覧だけ。
"""
from __future__ import annotations

import math
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from openvons.voice.engine import Decision
from openvons.voice.common_intents import PAN_INTENT_NAMES, pan_intents
from openvons.voice.grammar import Grammar, Intent, code_hypotheses, example_of, number_hypotheses
from openvons.voice.lexicon import Entity, Lexicon, Scope
from openvons.voice.state import StateDef, StateMachine

CONFIRM_TIMEOUT_SEC = 8.0
ZOOM_STEP = 1.6

INTENTS = [
    Intent("select_station", ["{station}[を](表示|出して|見せて|お願い)", "{station}(まで|に行きたい|へ)", "{station}[に](寄って|ズーム)", "{station}"], description="駅に寄る"),
    Intent("route", ["{origin}から{dest}[まで]"], slots={"origin": "station", "dest": "station"}, primary_only=True,
           description="経路 (乗換込み) を表示"),
    Intent("next_station", ["(次|つぎ)[の駅]", "一つ先", "先に進んで"], description="次の駅へ"),
    Intent("prev_station", ["(前|まえ|一つ前)[の駅]", "一つ戻って", "手前"], description="前の駅へ"),
    Intent("switch_line", ["(路線|線)[を](変えて|切り替え|切り替えて)", "別の路線", "乗り換え"], description="乗換路線に切り替え"),
    Intent("zoom_in", ["[もっと](寄って|寄せて|拡大|ズームイン)", "拡大して"], description="地図を拡大"),
    Intent("zoom_out", ["[もっと](引いて|縮小|ズームアウト|広く)", "縮小して", "全体を見せて"], description="地図を縮小"),
    Intent("back", ["(全体|路線図|元の画面|一覧|地図|ホーム|最初)[に|へ](戻って|戻る|戻して)", "戻る", "閉じて",
                    "ホーム[へ|に]", "地図[へ|に]"], description="路線図全体に戻る (いつでも使えます)"),
    Intent("favorite", ["(この駅|ここ)[を](登録|お気に入り|保存)[して]", "お気に入り登録"], risk="high", description="お気に入りに登録 (要確認)"),
    Intent("yes", ["はい", "そうです", "お願いします", "OK", "実行"], description="確認: はい", allow_embed=False, confirmable=False, positive=True),
    Intent("no", ["いいえ", "違います", "キャンセル", "やめて", "取り消し"], description="確認: いいえ", allow_embed=False, confirmable=False, positive=False),
    Intent("help", ["ヘルプ", "何ができる", "コマンド一覧"], description="使えるコマンド"),
] + pan_intents()

#: どの状態でも受け付ける意図 (確認待ちで行き止まりにならないように)
GLOBAL_INTENTS = ["back", "help"]

STATES = {
    "MAP": StateDef("MAP", ["select_station", "route", "zoom_in", "zoom_out", *PAN_INTENT_NAMES, *GLOBAL_INTENTS],
                    "地図全体。駅名を言うと寄ります。「新宿から東京まで」で経路。拡大縮小と上下左右の移動も"),
    "STATION": StateDef("STATION", ["next_station", "prev_station", "switch_line", "zoom_in", "zoom_out", "favorite",
                                    "select_station", "route", *PAN_INTENT_NAMES, *GLOBAL_INTENTS],
                        "駅を選択中。次/前の駅・路線切替・拡大縮小・経路・戻る"),
    # 確認待ちでも「戻る」と「ヘルプ」は通す。はい / いいえ しか受け付けないと行き止まりに感じる
    "CONFIRM": StateDef("CONFIRM", ["yes", "no", *GLOBAL_INTENTS], "確認待ち。はい / いいえ (「戻る」で取り消し)"),
}

#: 事前学習の校正に使う他状態の発話 (状態名, 受理意図, [(発話, 正解意図 | None=該当なし)])
CALIBRATION_STATES = [
    ("STATION", ["next_station", "prev_station", "switch_line", "zoom_in", "zoom_out", "back", "favorite", "help"], [
        ("次の駅", "next_station"), ("一つ先", "next_station"), ("前の駅", "prev_station"), ("一つ戻って", "prev_station"), ("路線を変えて", "switch_line"),
        ("もっと寄って", "zoom_in"), ("もっと引いて", "zoom_out"), ("全体に戻って", "back"), ("戻る", "back"), ("この駅を登録して", "favorite"),
        ("地図全体に戻して", "back"),
        ("はい", None), ("少々お待ちください", None), ("はい、お世話になっております", None), ("了解しました", None), ("ちょっと待ってね", None), ("何時に着くかな", None),
    ]),
    ("CONFIRM", ["yes", "no"], [
        ("はい", "yes"), ("そうです", "yes"), ("お願いします", "yes"), ("いいえ", "no"), ("違います", "no"), ("キャンセル", "no"),
        ("はい、お世話になっております", None), ("はいはい、大丈夫です", None), ("この駅を登録して", None), ("全体に戻って", None), ("いいえ、こちらこそ", None), ("そうですね、確認します", None),
    ]),
]

SLOT = "station"
SELECT_INTENT = "select_station"
DATA_FILE = "stations.json"
TITLE = "駅名で動く路線図"


def default_scopes(lex: Lexicon) -> list[Scope]:
    return [
        Scope("yamanote", "JR 山手線", filters={"line_codes": ["11302"]}),
        Scope("tokyo_jr", "JR東日本 首都圏 (山手・中央・京浜東北・総武・埼京)", filters={"line_codes": ["11302", "11312", "11332", "11314", "11313"]}),
        Scope("osaka", "大阪環状線 + 御堂筋線", filters={"line_codes": ["11623", "99618"]}),
    ]


ROUTE_MAX_STATIONS = 150     # 2 スロットの経路意図は n² 仮説になるので、範囲がこれより大きいときは外す (150 駅 = 4.5 万仮説)
SPEED_KMH = 45.0             # 駅間の走行速度の目安
STOP_MIN = 0.5               # 1 駅停車の目安 (分)
TRANSFER_MIN = 5.0           # 乗換 1 回の目安 (分)
EXCLUDE_LINE = re.compile(r"新幹線|エクスプレス|ライナー|特急|リゾート|成田|スカイアクセス")   # 経路探索から外す路線 (停車駅が飛ぶ)


@dataclass
class View:
    zoom: float = 1.0
    line_code: str | None = None
    favorites: list[str] = field(default_factory=list)
    route: dict[str, Any] | None = None      # {"stations": [id...], "legs": [{"line", "line_code", "from", "to", "hops"}], "minutes": m}

    def to_dict(self) -> dict[str, Any]:
        return {"zoom": self.zoom, "line_code": self.line_code, "favorites": list(self.favorites), "route": self.route}


class StationApp:
    """1 セッション分。CameraApp と同じ契約 (command_set / apply / snapshot / set_scope / invalidate / expire_confirm)。"""

    def __init__(self, lexicon: Lexicon, scope: Scope):
        self.lexicon = lexicon
        self.scope = scope
        self.grammar = Grammar(INTENTS)
        self.sm = StateMachine(STATES, "MAP", {"station": None, "pending": None})
        self.view = View()
        self.log: list[dict[str, Any]] = []
        self._cs_cache: dict[str, Any] = {}

    # Scope.matches はスカラー属性の一致しか見ないので、リスト属性 (line_codes) は自前で判定する
    def entities(self) -> list[Entity]:
        out = []
        for e in self.lexicon:
            if e.id in self.scope.exclude_ids:
                continue
            if e.id in self.scope.ids:
                out.append(e); continue
            ok = bool(self.scope.filters)
            for k, allowed in self.scope.filters.items():
                v = e.attrs.get(k)
                if isinstance(v, list):
                    if not any(x in allowed for x in v):
                        ok = False; break
                elif v not in allowed:
                    ok = False; break
            if ok:
                out.append(e)
        return out

    def set_scope(self, scope: Scope) -> None:
        self.scope = scope
        self._cs_cache.clear()
        self.sm.goto("MAP", station=None, pending=None)
        self.view = View()

    def invalidate(self) -> None:
        self._cs_cache.clear()

    #: コードを振る上限。範囲が広すぎると番号を探す方が大変になる
    CODE_MAX = 300

    def command_set(self):
        st = self.sm.state
        if st not in self._cs_cache:
            ents = self.entities()
            intents = [i for i in self.sm.allowed_intents() if not (i == "route" and len(ents) > ROUTE_MAX_STATIONS)]
            cs = self.grammar.compile(intents, ents, st)
            # 名前を全部覚えなくて済むように、画面の並び順で S01, S02... を振る
            if SELECT_INTENT in intents and len(ents) <= self.CODE_MAX:
                from openvons.voice.grammar import CommandSet
                coded = self.coded_entities()
                cs = CommandSet(cs.hyps + code_hypotheses(SELECT_INTENT, SLOT, coded)
                                + number_hypotheses(SELECT_INTENT, SLOT, [e for _, e in coded]), st)
            self._cs_cache[st] = cs
        return self._cs_cache[st]

    def coded_entities(self) -> list[tuple[str, Entity]]:
        """画面の並び順に S01, S02, ... を振る。範囲が変われば振り直す。"""
        return [(f"S{i:02d}", e) for i, e in enumerate(self.entities(), 1)]

    # ------------------------------------------------------------ 経路探索 (駅 = 節、同じ路線で隣り合う駅 = 辺、乗換に罰則)
    def _graph(self):
        """辺 = 同じ路線で隣り合う駅。重み = 駅間距離 (haversine) / 速度 + 停車時間。特急系の路線は外す。"""
        if getattr(self, "_g", None) is None:
            by_line: dict[str, list[tuple[int, str]]] = {}
            for e in self.lexicon:
                for p in e.attrs["positions"]:
                    if EXCLUDE_LINE.search(p["line"]):
                        continue
                    by_line.setdefault(p["line_code"], []).append((p["index"], e.id))
            adj: dict[str, list[tuple[str, str, float]]] = {}
            for lc, lst in by_line.items():
                lst.sort()
                for (i1, a), (i2, b) in zip(lst, lst[1:]):
                    if i2 == i1 + 1:
                        w = self._km(a, b) / SPEED_KMH * 60 + STOP_MIN
                        adj.setdefault(a, []).append((b, lc, w)); adj.setdefault(b, []).append((a, lc, w))
            self._g = adj
        return self._g

    def _km(self, a: str, b: str) -> float:
        ea, eb = self.lexicon.get(a).attrs, self.lexicon.get(b).attrs
        la1, lo1, la2, lo2 = map(math.radians, (ea["lat"], ea["lng"], eb["lat"], eb["lng"]))
        h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
        return 2 * 6371 * math.asin(math.sqrt(h))

    def find_route(self, src: str, dst: str) -> dict[str, Any] | None:
        """Dijkstra: 辺の重み = 駅間の所要時間の目安、路線が変わるたび TRANSFER_MIN 分。状態は (駅, 乗ってきた路線)。"""
        import heapq
        adj = self._graph()
        best: dict[tuple[str, str | None], float] = {(src, None): 0.0}
        prev: dict[tuple[str, str | None], tuple[str, str | None] | None] = {(src, None): None}
        pq = [(0.0, src, None)]
        goal = None
        while pq:
            cost, node, line = heapq.heappop(pq)
            if best.get((node, line), 1e18) < cost:
                continue
            if node == dst:
                goal = (node, line); break
            for nxt, lc, w in adj.get(node, []):
                c = cost + w + (TRANSFER_MIN if line is not None and lc != line else 0.0)
                if c < best.get((nxt, lc), 1e18):
                    best[(nxt, lc)] = c; prev[(nxt, lc)] = (node, line); heapq.heappush(pq, (c, nxt, lc))
        if goal is None:
            return None
        path: list[tuple[str, str | None]] = []
        cur: tuple[str, str | None] | None = goal
        while cur is not None:
            path.append(cur); cur = prev[cur]
        path.reverse()
        stations = [n for n, _ in path]
        legs: list[dict[str, Any]] = []
        for (a, _), (b, lc) in zip(path, path[1:]):
            if legs and legs[-1]["line_code"] == lc:
                legs[-1]["to"] = b; legs[-1]["hops"] += 1
            else:
                name = next(p["line"] for p in self.lexicon.get(b).attrs["positions"] if p["line_code"] == lc)
                legs.append({"line": name, "line_code": lc, "from": a, "to": b, "hops": 1})
        return {"stations": stations, "legs": legs, "minutes": round(best[goal]), "transfers": len(legs) - 1}

    def allowed_commands(self) -> list[dict[str, str]]:
        out = []
        for name in self.sm.allowed_intents():
            it = self.grammar.intents[name]
            ex = example_of(it.patterns[0], {SLOT: "<駅名>"})
            out.append({"intent": name, "example": ex.replace("{station}", "<駅名>").replace("{origin}", "<駅名>").replace("{dest}", "<駅名>"), "description": it.description, "risk": it.risk})
        return out

    def expire_confirm(self) -> bool:
        if self.sm.state == "CONFIRM" and time.time() - self.sm.context.get("confirm_at", 0) > CONFIRM_TIMEOUT_SEC:
            self.sm.back(); return True
        return False

    # ------------------------------------------------------------ 適用
    def apply(self, d: Decision) -> dict[str, Any]:
        ev: dict[str, Any] = {"type": "result", "decision": d.to_dict(), "applied": None, "speech": None}
        if d.action == "reject":
            ev["speech"] = "聞き取れませんでした"
        elif d.action == "confirm":
            h = d.top.hypothesis
            self.sm.goto("CONFIRM", pending={"intent": h.intent, "slots": h.slots, "params": h.params, "text": h.text}, confirm_at=time.time())
            ev["speech"] = f"{h.text} ですか？"
        elif d.action == "execute":
            h = d.top.hypothesis
            ev["applied"] = self._execute(h.intent, h.slots, h.params, h.text)
            ev["speech"] = ev["applied"].get("speech")
        ev["state"] = self.snapshot()
        self.log.append({"free": d.free_kana, "action": d.action, "top": d.top.to_dict() if d.top else None, "state": self.sm.state})
        return ev

    def _neighbor(self, sid: str, step: int) -> Entity | None:
        e = self.lexicon.get(sid)
        pos = [p for p in e.attrs["positions"] if p["line_code"] == self.view.line_code] or e.attrs["positions"]
        p = pos[0]
        self.view.line_code = p["line_code"]
        target_index = p["index"] + step
        for cand in self.entities():
            for q in cand.attrs["positions"]:
                if q["line_code"] == p["line_code"] and q["index"] == target_index:
                    return cand
        # 範囲外の駅も許す (路線の続き)
        for cand in self.lexicon:
            for q in cand.attrs["positions"]:
                if q["line_code"] == p["line_code"] and q["index"] == target_index:
                    return cand
        return None

    def _execute(self, intent: str, slots: dict[str, str], params: dict[str, Any], text: str) -> dict[str, Any]:
        pending = self.sm.context.get("pending")
        if self.sm.state == "CONFIRM":
            if intent == "yes" and pending:
                self.sm.back()
                return self._execute(pending["intent"], pending["slots"], pending["params"], pending["text"])
            self.sm.back()
            if intent in ("back", "help"):       # 確認待ちからの逃げ道
                self.sm.context["pending"] = None
                return self._execute(intent, slots, params, text)
            return {"intent": intent, "speech": "取り消しました"}
        # 地図の移動はどの状態でも効く (駅を選んでいなくても)
        if intent.startswith("pan_"):
            return {"intent": intent, "map": {"pan": params.get("dir"), "amount": params.get("amount", 0.6)},
                    "speech": text}
        sid = self.sm.context.get("station")
        if intent == "select_station":
            self.view.route = None
            e = self.lexicon.get(slots[SLOT])
            lines = [p["line_code"] for p in e.attrs["positions"]]
            if self.view.line_code not in lines:
                self.view.line_code = lines[0]
            self.view.zoom = max(self.view.zoom, 3.0)
            self.sm.goto("STATION", station=e.id, pending=None)
            return {"intent": intent, "station": e.id, "speech": f"{e.label} です。{e.attrs['route_label']}"}
        if intent == "route":
            a = self.lexicon.get(slots["origin"]); b = self.lexicon.get(slots["dest"])
            r = self.find_route(a.id, b.id)
            if r is None:
                return {"intent": intent, "speech": f"{a.label} から {b.label} への経路が見つかりません"}
            self.view.route = r
            self.view.line_code = r["legs"][0]["line_code"]
            self.sm.goto("STATION", station=b.id, pending=None)
            legs = "、".join(f"{lg['line']} {lg['hops']} 駅" for lg in r["legs"])
            return {"intent": intent, "station": b.id, "route": r,
                    "speech": f"{a.label} から {b.label} まで、{legs}。乗換 {r['transfers']} 回、約 {r['minutes']} 分"}
        if intent == "back":
            self.sm.goto("MAP", station=None, pending=None); self.view.zoom = 1.0; self.view.route = None
            return {"intent": intent, "speech": "地図全体に戻ります"}
        if intent == "help":
            return {"intent": intent, "speech": "、".join(c["example"] for c in self.allowed_commands()[:5])}
        if sid is None:
            return {"intent": intent, "speech": "駅を選んでください"}
        if intent in ("next_station", "prev_station"):
            nb = self._neighbor(sid, 1 if intent == "next_station" else -1)
            if nb is None:
                return {"intent": intent, "speech": "終点です"}
            self.sm.context["station"] = nb.id
            return {"intent": intent, "station": nb.id, "speech": f"{nb.label}"}
        if intent == "switch_line":
            e = self.lexicon.get(sid)
            lines = [p["line_code"] for p in e.attrs["positions"]]
            if len(lines) < 2:
                return {"intent": intent, "speech": "乗換路線はありません"}
            i = lines.index(self.view.line_code) if self.view.line_code in lines else -1
            self.view.line_code = lines[(i + 1) % len(lines)]
            name = next(p["line"] for p in e.attrs["positions"] if p["line_code"] == self.view.line_code)
            return {"intent": intent, "speech": f"{name} に切り替えました"}
        if intent == "zoom_in": self.view.zoom = min(self.view.zoom * ZOOM_STEP, 20.0)
        elif intent == "zoom_out": self.view.zoom = max(self.view.zoom / ZOOM_STEP, 1.0)
        elif intent == "favorite":
            if sid not in self.view.favorites:
                self.view.favorites.append(sid)
            return {"intent": intent, "speech": f"{self.lexicon.get(sid).label} を登録しました"}
        return {"intent": intent, "station": sid, "speech": text}

    def snapshot(self) -> dict[str, Any]:
        s = self.sm.snapshot()
        sid = s["context"].get("station")
        e = self.lexicon.get(sid) if sid else None
        s["camera"] = {"id": sid, "label": e.label, "attrs": e.attrs} if e else None    # UI 互換のキー名
        s["station"] = s["camera"]
        s["view"] = self.view.to_dict()
        s["allowed_commands"] = self.allowed_commands()
        s["n_hypotheses"] = len(self.command_set())
        s["n_cameras"] = len(self.entities())
        s["scope"] = {"id": self.scope.id, "name": self.scope.name}
        return s


AppClass = StationApp
