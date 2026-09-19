"""JevPick (ジェヴピック) — 文章生成の先読み (speculative decoding) に openvons の「候補から選ぶ」を転用したもの。

1. 候補メニュー (candidates/): 次に出そうな数 token の列を、ツール定義の展開 (tool_schema)・過去の出力や prompt の
   n-gram (ngram)・同 repo のコード (repository)・定型 (grammar, macro_copy)・同梱 MTP head (mtp_qwen35) から作る。
2. JevPick 本体 (scorer/): target モデル自身の hidden state を見て、候補メニューから 1 つ選び、期待受理長を出す。
   期待受理長が低ければ「先読みしない」と判断する (controller)。
3. verifier (runtime/): 選んだ候補を target に一括で検算させ、一致した prefix だけ採用する。greedy なら出力は不変。

データの作り方は data/、実験は experiments/jevpick/、結果と考察は docs/jevpick/。
"""
