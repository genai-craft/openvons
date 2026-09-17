"""選択肢 (実体) と担当範囲の管理.

Entity  = 音声で指せる「もの」1 個 (カメラ 1 台など)。表示名・複数の読み・属性 (整備局/事務所/路線/方向...) を持つ。
Lexicon = Entity の集合。属性で絞り込める。JSON で永続化。
Scope   = 担当範囲。属性フィルタと個別 ID の和集合で「この人が今指せる実体」を決める。
          選択肢が最初から絞られることが、この方式の信頼性の源泉。

読みは複数持てる (readings)。優先順: 人手の読み > 事前学習 (TTS 往復) で採れた読み > 郵便番号データ > G2P。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from . import kana as K


@dataclass
class Entity:
    id: str
    label: str                                  # 表示名 (漢字かな交じり) 例: 船橋南1上り
    kind: str = "camera"
    readings: list[str] = field(default_factory=list)   # ASR 形カナ。空なら G2P で補う
    aliases: list[str] = field(default_factory=list)    # 別名 (漢字) 例: 「船橋南」だけでも指せる
    attrs: dict[str, Any] = field(default_factory=dict)  # bureau/office/pref/route/direction/lat/lon ...

    def primary_readings(self) -> list[str]:
        """表示名そのものの読み (人手・事前学習・郵便番号データ、無ければ G2P)。"""
        out: list[str] = []
        seen: set[str] = set()
        for r in self.readings:
            r = K.normalize(r)
            if r and r not in seen:
                seen.add(r); out.append(r)
        if not out:
            out.append(K.g2p(self.label))
        return out

    def alias_readings(self) -> list[str]:
        """別名の読み。範囲内で他の実体と衝突するものは grammar.compile が落とす
        (「船橋南」は上り/下りの両方を指すので単独では選べない)。"""
        prim = set(self.primary_readings())
        out: list[str] = []
        for a in self.aliases:
            g = K.g2p(a)
            if g and g not in prim and g not in out:
                out.append(g)
        return out

    def all_readings(self) -> list[str]:
        return self.primary_readings() + self.alias_readings()


@dataclass
class Scope:
    """担当範囲。filters は {属性名: 許可値リスト} の AND、ids は個別追加。exclude_ids で個別除外。"""
    id: str
    name: str
    filters: dict[str, list[Any]] = field(default_factory=dict)
    ids: list[str] = field(default_factory=list)
    exclude_ids: list[str] = field(default_factory=list)
    profile: dict[str, Any] = field(default_factory=dict)   # 事前学習の結果 (校正値・実測精度・混同対) を置く

    def matches(self, e: Entity) -> bool:
        if e.id in self.exclude_ids:
            return False
        if e.id in self.ids:
            return True
        if not self.filters:
            return False
        for k, allowed in self.filters.items():
            if e.attrs.get(k) not in allowed:
                return False
        return True


class Lexicon:
    def __init__(self, entities: Iterable[Entity] = ()):
        self._by_id: dict[str, Entity] = {}
        for e in entities:
            self.add(e)

    # --- 実体
    def add(self, e: Entity) -> None:
        self._by_id[e.id] = e

    def remove(self, eid: str) -> None:
        self._by_id.pop(eid, None)

    def get(self, eid: str) -> Entity | None:
        return self._by_id.get(eid)

    def __len__(self) -> int:
        return len(self._by_id)

    def __iter__(self):
        return iter(self._by_id.values())

    def filter(self, **attrs: Any) -> list[Entity]:
        out = []
        for e in self._by_id.values():
            if all(e.attrs.get(k) == v for k, v in attrs.items()):
                out.append(e)
        return out

    def values_of(self, attr: str, **where: Any) -> list[Any]:
        vals = []
        seen: set[Any] = set()
        for e in self.filter(**where):
            v = e.attrs.get(attr)
            if v is not None and v not in seen:
                seen.add(v); vals.append(v)
        return vals

    def in_scope(self, scope: Scope) -> list[Entity]:
        return [e for e in self._by_id.values() if scope.matches(e)]

    def add_reading(self, eid: str, reading: str) -> bool:
        """事前学習で採れた読みを追加。既にあれば False。"""
        e = self._by_id[eid]
        r = K.normalize(reading)
        if not r or r in e.all_readings():
            return False
        e.readings.append(r)
        return True

    # --- 永続化
    def to_json(self) -> list[dict[str, Any]]:
        return [asdict(e) for e in self._by_id.values()]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), ensure_ascii=False, indent=0), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Lexicon":
        rows = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(Entity(**r) for r in rows)


class ScopeStore:
    """担当範囲の CRUD + JSON 永続化。"""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.scopes: dict[str, Scope] = {}
        if self.path and self.path.exists():
            for r in json.loads(self.path.read_text(encoding="utf-8")):
                self.scopes[r["id"]] = Scope(**r)

    def upsert(self, s: Scope) -> Scope:
        self.scopes[s.id] = s
        self.save()
        return s

    def delete(self, sid: str) -> None:
        self.scopes.pop(sid, None)
        self.save()

    def get(self, sid: str) -> Scope | None:
        return self.scopes.get(sid)

    def save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps([asdict(s) for s in self.scopes.values()], ensure_ascii=False, indent=1), encoding="utf-8")
