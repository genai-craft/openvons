"""Source C (簡易版): 固定 template の corpus を suffix 一致で引く (§6.3, §22 の「固定grammar」)。

parser state を持つ本格実装ではなく、ドメイン別の定型断片を token 化して
SuffixIndex に入れ、文脈末尾と一致する断片の続きを候補にする。"""
from __future__ import annotations

from .ngram import SuffixIndex

TEMPLATES = {
    "python": [
        "```python\n", "\n```", "def ", "(self, ", "):\n    ", "):\n        ", "    return ",
        "        return ", "if __name__ == \"__main__\":\n    ", " is None:\n        return ",
        " is not None", "for i in range(len(", "for i in range(", "for x in ", " in enumerate(",
        "    \"\"\"", "\"\"\"\n    ", "import re\n", "import math\n", "from collections import ",
        "    if ", "        if ", "    else:\n        ", "    elif ", "    while ", "    try:\n        ",
        "    except ", "return True\n", "return False\n", "return None\n", "return result\n",
        "result = []\n", "result.append(", "        result.append(", " = {}\n", " = []\n", " = 0\n",
        ".append(", ".items():\n", ".keys()", ".values()", " == ", " != ", " += 1\n", " -= 1\n",
        "len(", "range(", "sorted(", "max(", "min(", "sum(", "str(", "int(", "list(", "dict(",
        "lambda x: ", "key=lambda ", "\n\n", "\n    ", "\n        ", "\n            ",
        "assert ", ") == ", "print(", "def test_", "    pass\n",
    ],
    "toolcall": [
        "<tool_call>\n{\"name\": \"", "\", \"arguments\": {\"", "\"}}\n</tool_call>", "}}\n</tool_call>",
        "\": \"", "\", \"", "\": ", ", \"", "\"}", "</tool_call>", "<tool_call>\n", "{\"name\": \"",
        "\"arguments\": {", "\": [", "\"], \"", "\": true", "\": false", "\": null",
    ],
    "chat": [
        "\n\n", "\n\n**", "**\n", "**:", ":\n\n", "\n- ", "\n1. ", "\n2. ", "\n3. ", "\n4. ", "\n5. ",
        "\n\n### ", "\n\n## ", "。\n\n", "です。", "ます。", "ました。", "ください。", "\n\n---\n\n",
        "```\n", "\n```\n\n", "**Note", "In summary", "Here are", "Here's", "I hope this helps",
        "Let me know if", "Sure!", "Certainly!", "Of course!", "はい、", "以下に", "まず、", "次に、",
        "最後に、", "例えば、", "また、", "しかし、", "そのため、", "〜", "・",
    ],
}


def build_grammar_index(tok, domain: str, max_n: int = 6) -> SuffixIndex:
    idx = SuffixIndex(max_n=max_n, max_positions=256)
    sep = tok.convert_tokens_to_ids("<|endoftext|>")
    for s in TEMPLATES.get(domain, []):
        idx.extend(tok.encode(s, add_special_tokens=False))
        idx.extend([sep])  # template 境界を跨がないよう区切る
    return idx
