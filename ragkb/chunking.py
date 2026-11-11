"""Structure-aware splitting with token budgets, headings, tables and provenance."""

import re

from .schemas import Chunk, Element


class StructureChunker:
    def __init__(self, tokenizer, target_tokens=512, overlap_tokens=64):
        self.tokenizer = tokenizer
        self.target = target_tokens
        self.overlap = overlap_tokens

    def count(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def trim(self, text, budget):
        if self.count(text) <= budget:
            return text
        lo, hi = 0, len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.count(text[:mid]) <= budget:
                lo = mid
            else:
                hi = mid - 1
        return text[:lo]

    def split_long(self, text, budget, overlap=True):
        remaining = text.strip()
        while remaining:
            if self.count(remaining) <= budget:
                yield remaining
                break
            prefix = self.trim(remaining, budget)
            boundaries = list(re.finditer(r"[。！？!?；;\n]|\.\s", prefix))
            if boundaries and boundaries[-1].end() > len(prefix) // 2:
                prefix = prefix[: boundaries[-1].end()]
            cut = len(prefix)
            yield prefix.strip()
            # Keep a short suffix only inside an oversized semantic element.
            tail = ""
            if overlap:
                candidate = prefix[-max(1, len(prefix) // 3) :]
                while candidate and self.count(candidate) > min(self.overlap, budget // 4):
                    candidate = candidate[1:]
                tail = candidate
            remaining = (tail + remaining[cut:]).strip()

    def table_parts(self, html, budget):
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        rows = [
            " | ".join(cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"]))
            for row in soup.find_all("tr")
        ]
        if not rows:
            yield from self.split_long(html, budget, overlap=False)
            return
        header = self.trim(rows[0], max(1, budget // 3))
        body_budget = budget - self.count(header + "\n") - 4
        current = header
        for row in rows[1:]:
            for part in self.split_long(row, max(1, body_budget), overlap=False):
                combined = current + "\n" + part
                if self.count(combined) > budget and current != header:
                    yield current
                    current = header
                current += "\n" + part
        if current:
            yield current

    def split(self, elements: list[Element], doc_id: str, filename: str) -> list[Chunk]:
        chunks, headings, buffer, pages = [], [], [], set()

        def prefix():
            title = self.trim(" > ".join(headings), 64)
            return f"文档：{self.trim(filename, 32)}\n章节：{title}\n\n"

        def emit(text, kind="text", source_pages=None):
            if not text.strip():
                return
            embedded = prefix() + text.strip()
            chunks.append(
                Chunk(
                    id=f"{doc_id}:{len(chunks):06d}",
                    document_id=doc_id,
                    filename=filename,
                    text=text.strip(),
                    embedding_text=embedded,
                    heading=" > ".join(headings),
                    index=len(chunks),
                    pages=sorted(source_pages if source_pages is not None else pages),
                    kind=kind,
                    tokens=self.count(embedded),
                )
            )

        def flush():
            if buffer:
                emit("\n\n".join(buffer))
            buffer.clear()
            pages.clear()

        for element in elements:
            if not element.text.strip():
                continue
            if element.heading_level:
                flush()
                level = max(1, min(element.heading_level, 6))
                headings[:] = headings[: level - 1] + [element.text.strip()]
                continue
            budget = self.target - self.count(prefix()) - 8
            source = {element.page} if element.page is not None else set()
            if element.kind == "table":
                flush()
                for part in self.table_parts(element.text, budget):
                    emit(part, "table", source)
                continue
            if self.count(element.text) > budget:
                flush()
                for part in self.split_long(element.text, budget):
                    emit(part, element.kind, source)
                continue
            candidate = "\n\n".join(buffer + [element.text])
            if self.count(candidate) > budget:
                flush()
            buffer.append(element.text)
            pages.update(source)
        flush()
        return chunks
