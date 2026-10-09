# openvons.lm の学習記録 (openvons/lm/training/train.py の出力先)

指標だけの JSON (学習データは含まない)。

| ファイル | 内容 | 結果 |
|---|---|---|
| fb_rerank_hard.json / fb_rerank_soft.json | ragnarok (genai-craft/ragnarok) の FinanceBench 候補ページ選択 (25 択) を、凍結した Qwen3-4B + MLP head に蒸留できるかの試験 (2026-10-02)。hard = 27B の判定の argmax、soft = 27B の選択肢確率への KL | **失敗**。test 正解率 0.10 (25 択の偶然 0.04 よりは上だが実用外)。長い選択肢 (ページの抜粋) を比べる課題は「最終トークンの hidden + head」では表せない。ragnarok は 4B の確率判定 (light) をそのまま使う形にした |
