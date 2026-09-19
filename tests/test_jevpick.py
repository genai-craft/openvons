"""JevPick の GPU 不要なテスト: 候補メニューの生成と一致長。"""
from openvons.jevpick.candidates.base import match_len
from openvons.jevpick.candidates.ngram import SuffixIndex, rank
from openvons.jevpick.candidates.tool_schema import detect_format, expand_tool_templates


def test_match_len():
    assert match_len((1, 2, 3), [1, 2, 3, 4]) == 3
    assert match_len((1, 9, 3), [1, 2, 3]) == 1
    assert match_len((), [1]) == 0


def test_suffix_index_finds_continuation():
    idx = SuffixIndex(max_n=4)
    seq = [5, 6, 7, 8, 9, 5, 6, 7, 8, 9]
    idx.extend(seq)
    occ = idx.occurrences([5, 6, 7], exclude_from=len(seq))
    cands = idx.blocks(occ, 2, "ngram")
    assert cands, "suffix 5 6 7 must be found"
    assert cands[0].token_ids == (8, 9)
    # 最長一致 n が prior の先頭に来る
    assert rank(cands, 1)[0].prior_score[0] == 3


def test_tool_schema_expansion_both_formats():
    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {
        "type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}},
        "required": ["city"]}}}]
    js = expand_tool_templates(tools, "Weather in Tokyo?", "json")
    xml = expand_tool_templates(tools, "Weather in Tokyo?", "xml")
    assert any('{"name": "get_weather", "arguments": {' in s for s in js)
    assert any('"city": "Tokyo"' in s for s in js)  # user 発話からの copy slot
    assert any("<function=get_weather>" in s for s in xml)
    assert any("<parameter=unit>\nc\n</parameter>" in s for s in xml)  # enum の展開
    assert detect_format("<tools>...</tools> ... <function=") == "xml"
    assert detect_format('{"name": <function-name>, "arguments":') == "json"
