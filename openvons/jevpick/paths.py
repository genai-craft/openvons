"""JevPick のデータ置き場。環境変数 JEVPICK_DATA で変更できる (既定: リポジトリ直下の state/jevpick)。

prompt・greedy trace・hidden state・候補・学習済み JevPick・ログをここに置く。数十 GB になる (hidden state が大半)。
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("JEVPICK_DATA", str(REPO / "state" / "jevpick")))
DATA.mkdir(parents=True, exist_ok=True)
