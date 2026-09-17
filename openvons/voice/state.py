"""汎用の状態機械: 状態ごとに「今使える意図」を決める.

ウェイクワードの代わりに、システムの状態が受理する発話を絞る。
例 (カメラ監視): 一覧 (IDLE) では「<カメラ名>」「<カメラ名>を表示」だけ、
拡大中 (FOCUS) では PTZ 操作と「戻る」、確認中 (CONFIRM) では「はい/いいえ」だけ。
状態が小さいほど選択肢が少なく、認識は確実になる。

アプリ側は StateMachine を継承せず、States 定義 + on_intent コールバックで振る舞いを与える。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class StateDef:
    name: str
    intents: list[str]                     # この状態で受理する意図名
    description: str = ""
    # 状態で意図ごとに上書きしたい追加テンプレート (省略可)
    extra_patterns: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class Transition:
    state: str
    context: dict[str, Any]
    message: str = ""


class StateMachine:
    def __init__(self, states: dict[str, StateDef], initial: str, context: dict[str, Any] | None = None):
        self.states = states
        self.state = initial
        self.context: dict[str, Any] = dict(context or {})
        self.history: list[tuple[str, dict[str, Any]]] = []
        self.listeners: list[Callable[[str, dict[str, Any]], None]] = []

    @property
    def current(self) -> StateDef:
        return self.states[self.state]

    def allowed_intents(self) -> list[str]:
        return list(self.current.intents)

    def goto(self, state: str, message: str = "", **ctx: Any) -> Transition:
        if state not in self.states:
            raise KeyError(state)
        self.history.append((self.state, dict(self.context)))
        if len(self.history) > 50:
            self.history.pop(0)
        self.state = state
        self.context.update(ctx)
        for f in self.listeners:
            f(self.state, self.context)
        return Transition(self.state, dict(self.context), message)

    def back(self) -> Transition | None:
        if not self.history:
            return None
        st, ctx = self.history.pop()
        self.state, self.context = st, ctx
        for f in self.listeners:
            f(self.state, self.context)
        return Transition(self.state, dict(self.context))

    def snapshot(self) -> dict[str, Any]:
        return {"state": self.state, "context": dict(self.context), "allowed_intents": self.allowed_intents(), "description": self.current.description}
