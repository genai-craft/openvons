"""openvons.voice (旧 sashizu / 指図) — 有限選択肢に確率で答える日本語音声コマンド.

構成 (音声 -> 実行までの 1 本の流れ):

    音声 --kana-whisper(自由認識)--> カナ列
         --状態に応じたコマンド集合と照合 (mora 編集距離で絞り込み)--> 上位 K 候補
         --kana-whisper で候補文を教師強制採点 log p(候補|音声)--> 校正済み確率 (+「該当なし」)
         --3 段閾値 (実行 / 確認 / 棄却)--> アプリの状態遷移

主要モジュール:
    kana        カナ正規化・G2P・モーラ編集距離
    lexicon     選択肢 (カメラ等の実体) と担当範囲 (Scope) の管理
    grammar     意図・テンプレート・状態ごとのコマンド集合のコンパイル
    asr         kana-whisper の自由認識と候補一括採点
    engine      認識パイプラインと判断 (実行/確認/棄却)
    (校正と判断ポリシーは openvons.core.none_calibration / openvons.core.decision)
    state       汎用の状態機械 (状態 -> 使える意図)
    synth       TTS を使った事前学習 (実際の読みの採取・混同分析・校正)
"""
__version__ = "0.1.0"
