from ragkb.chunking import StructureChunker
from ragkb.parser_worker import from_content_list, parse_text
from ragkb.schemas import Element


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text)


def chunker():
    return StructureChunker(CharacterTokenizer(), target_tokens=160, overlap_tokens=16)


def test_long_chinese_preserves_all_content_with_budget():
    text = "".join(f"第{i}条规则需核验来源。" for i in range(60))
    chunks = chunker().split([Element(text, page=3)], "doc", "课程.pdf")
    assert len(chunks) > 1
    assert all(c.tokens <= 160 and c.pages == [3] for c in chunks)
    combined = "".join(c.text for c in chunks)
    assert all(f"第{i}条规则需核验来源。" in combined for i in range(60))


def test_headings_and_page_provenance():
    chunks = chunker().split(
        [
            Element("第一章", heading_level=1),
            Element("第一部分", page=1),
            Element("第二部分", page=2),
            Element("第二章", heading_level=1),
            Element("第三部分", page=3),
        ],
        "doc",
        "教材.pdf",
    )
    assert len(chunks) == 2
    assert chunks[0].pages == [1, 2]
    assert chunks[1].heading == "第二章"
    assert "第一部分" not in chunks[1].text


def test_large_table_repeats_header_and_keeps_rows():
    html = (
        "<table><tr><th>编号</th><th>内容</th></tr>"
        + "".join(f"<tr><td>{i}</td><td>课程模块说明{i}</td></tr>" for i in range(40))
        + "</table>"
    )
    chunks = chunker().split([Element(html, "table", 2)], "doc", "清单.pdf")
    assert len(chunks) > 1
    assert all(c.kind == "table" and c.text.startswith("编号 | 内容") for c in chunks)
    assert all(c.tokens <= 160 for c in chunks)
    combined = "\n".join(c.text for c in chunks)
    assert all(f"课程模块说明{i}" in combined for i in range(40))


def test_markdown_heading_and_table():
    elements = parse_text("# 课程\n说明\n\n| 模块 | 功能 |\n| --- | --- |\n| RAG | 检索 |")
    assert elements[0].heading_level == 1
    assert any(e.kind == "table" for e in elements)


def test_pdf_pages_are_one_based_but_docx_pages_are_not_invented():
    items = [{"type": "text", "text": "依据", "page_idx": 0}]
    assert from_content_list(items, True)[0].page == 1
    assert from_content_list(items, False)[0].page is None


def test_code_fence_does_not_create_false_headings():
    elements = parse_text("# 标题\n\n```python\n# this is a comment\n\nprint(1)\n```\n\n正文")
    assert len(elements) == 3
    assert elements[1].kind == "code"
    assert "# this is a comment" in elements[1].text
    assert elements[2].text == "正文"
