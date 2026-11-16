"""Internal worker. Called as a subprocess, never as a remote/cloud parser."""

import json
import re
import sys
from pathlib import Path

from .config import Settings
from .parsing import MinerUParser
from .schemas import Element


def parse_text(text: str) -> list[Element]:
    import html

    result, pending = [], []
    fence = None

    def flush(kind="text"):
        if not pending:
            return
        raw = "\n".join(pending).strip()
        if not raw:
            pending.clear()
            return
        if kind == "text" and len(pending) > 1 and re.match(r"\s*\|?\s*:?-{3}", pending[1]):
            rows = [line for line in pending if not re.fullmatch(r"[\s|:\-]+", line)]
            raw = (
                "<table>"
                + "".join(
                    "<tr>"
                    + "".join(
                        "<td>" + html.escape(cell.strip()) + "</td>"
                        for cell in row.strip().strip("|").split("|")
                    )
                    + "</tr>"
                    for row in rows
                )
                + "</table>"
            )
            kind = "table"
        result.append(Element(raw, kind=kind))
        pending.clear()

    for line in text.splitlines():
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        if fence:
            pending.append(line)
            if marker and marker.group(1)[0] == fence[0] and len(marker.group(1)) >= len(fence):
                flush("code")
                fence = None
        elif marker:
            flush()
            fence = marker.group(1)
            pending.append(line)
        elif heading := re.match(r"^(#{1,6})\s+(.+)$", line):
            flush()
            result.append(Element(heading.group(2), heading_level=len(heading.group(1))))
        elif not line.strip():
            flush()
        else:
            pending.append(line)
    flush("code" if fence else "text")
    return result


def from_content_list(items, has_pages: bool) -> list[Element]:
    result = []
    for item in items:
        kind = item.get("type", "text")
        page = item.get("page_idx")
        page = int(page) + 1 if has_pages and page is not None else None
        level = int(item.get("text_level") or 0)
        if kind == "table":
            caption = "\n".join(item.get("table_caption") or [])
            body = item.get("table_body", "")
            footnote = "\n".join(item.get("table_footnote") or [])
            if caption:
                result.append(Element(caption, page=page))
            if body:
                result.append(Element(body, "table", page))
            if footnote:
                result.append(Element(footnote, page=page))
        elif kind == "image":
            caption = "\n".join(item.get("image_caption") or [])
            if caption:
                result.append(Element(caption, "image_caption", page))
        else:
            text = item.get("text") or item.get("content") or ""
            if isinstance(text, str) and text.strip():
                result.append(Element(text, kind, page, level))
    return result


def main():
    settings = Settings()
    settings.prepare()
    path, output = Path(sys.argv[1]), Path(sys.argv[2])
    if path.suffix.lower() in {".md", ".txt"}:
        text = path.read_text(encoding="utf-8-sig")
        elements = parse_text(text)
        (output / "markdown.md").write_text(text, encoding="utf-8")
    else:
        from mineru.parser import parse
        from mineru.render import RenderFormat, render

        tier = "flash" if path.suffix.lower() == ".docx" else "basic"
        result = parse(str(path), tier=tier)
        (output / "middle_json.json").write_text(result.to_json(), encoding="utf-8")
        (output / "markdown.md").write_text(result.markdown(), encoding="utf-8")
        items = render(result.middle_json, RenderFormat.CONTENT_LIST)
        (output / "content_list.json").write_text(
            json.dumps(items, ensure_ascii=False), encoding="utf-8"
        )
        elements = from_content_list(items, path.suffix.lower() == ".pdf")
    MinerUParser.save_elements(elements, output)


if __name__ == "__main__":
    main()
