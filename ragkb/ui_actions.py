import time
from pathlib import Path

import gradio as gr

from .schemas import ChatRequest, HistoryMessage, SearchRequest


class UIActions:
    def __init__(self, services):
        self.services = services

    def documents(self):
        docs = self.services.repository.list()
        labels = {"queued": "排队中", "processing": "处理中", "ready": "已就绪", "failed": "失败"}
        rows = [
            [d["filename"], labels[d["status"]], d["stage"], d["chunk_count"], d["error"]]
            for d in docs
        ]
        choices = [(d["filename"] + " · " + labels[d["status"]], d["id"]) for d in docs]
        return rows, gr.update(choices=choices)

    def health(self):
        info = self.services.health()
        key = "已配置" if info["deepseek_configured"] else "待填写 .env"
        return (
            f"**文档 {info['documents']}** · **文本块 {info['chunks']}** · "
            f"DeepSeek 密钥：{key} · 推理设备：{info['device']}"
        )

    def upload(self, paths):
        if not paths:
            yield "请先选择文件。", *self.documents(), self.health()
            return
        documents = []
        notes = []
        for path in paths:
            try:
                with Path(path).open("rb") as stream:
                    document = self.services.ingestion.submit(stream, Path(path).name)
                documents.append(document["id"])
                if document.get("duplicate"):
                    notes.append(f"{document['filename']}：相同内容已存在，复用索引。")
            except Exception as error:
                notes.append(f"{Path(path).name}：{error}")
        while True:
            current = [self.services.repository.get(d) for d in documents]
            current = [d for d in current if d]
            report = notes + [
                f"{d['filename']}：{d['stage']}" + (f" — {d['error']}" if d["error"] else "")
                for d in current
            ]
            yield "\n\n".join(report), *self.documents(), self.health()
            if all(d["status"] in {"ready", "failed"} for d in current):
                break
            time.sleep(1)

    def preview(self, doc_id):
        if not doc_id:
            return [], "请选择文档。"
        doc = self.services.repository.get(doc_id)
        if not doc:
            return [], "文档不存在。"
        chunks = self.services.repository.chunks(doc_id, limit=100)
        rows = [
            [
                c["index"],
                c["kind"],
                c["heading"],
                "、".join(map(str, c["pages"])),
                c["tokens"],
                c["text"],
            ]
            for c in chunks
        ]
        return rows, f"{doc['filename']} · 共 {doc['chunk_count']} 块，展示前 100 块。"

    def delete(self, doc_id):
        if not doc_id:
            return "请选择文档。", *self.documents(), [], self.health()
        try:
            self.services.ingestion.delete(doc_id)
            message = "文档、解析结果和对应向量已删除。"
        except Exception as error:
            message = str(error)
        return message, *self.documents(), [], self.health()

    @staticmethod
    def source_rows(trace):
        return [
            [
                s["citation"],
                s["filename"],
                "、".join(map(str, s["pages"])) or "—",
                s["heading"],
                round(s["similarity"], 4),
                round(s["rerank_score"], 4) if s["rerank_score"] is not None else "—",
                s["text"],
            ]
            for s in trace.get("sources", [])
        ]

    @staticmethod
    def source_links(trace):
        # Display filenames only in the dataframe; link labels contain no document content.
        return " · ".join(
            f"[来源 {s['citation']} 原文件](/api/documents/{s['document_id']}/file)"
            for s in trace.get("sources", [])
        )

    def chat(self, query, history, retrieval_k, rerank_k, use_rerank):
        history = history or []
        if not query.strip():
            raise gr.Error("请输入问题。")
        display = history + [
            {"role": "user", "content": query},
            {"role": "assistant", "content": ""},
        ]
        answer, rows, trace, links = "", [], {}, ""
        yield display, history, "", rows, trace, links
        try:
            request = ChatRequest(
                query=query,
                history=[HistoryMessage(**m) for m in history[-20:]],
                retrieval_k=int(retrieval_k),
                rerank_k=int(rerank_k),
                use_rerank=use_rerank,
            )
            for event in self.services.qa.stream(request):
                if event["event"] == "sources":
                    trace = event["data"]
                    rows = self.source_rows(trace)
                    links = self.source_links(trace)
                elif event["event"] == "token":
                    answer += event["data"]
                elif event["event"] == "done":
                    trace["generation_total"] = event["data"]
                display[-1]["content"] = answer
                yield display, history, "", rows, trace, links
            history = (
                history
                + [{"role": "user", "content": query}, {"role": "assistant", "content": answer}]
            )[-20:]
            yield display, history, "", rows, trace, links
        except Exception as error:
            import logging

            logging.getLogger(__name__).exception("UI chat failed")
            message = (
                str(error)
                if isinstance(error, ValueError)
                else ("问答调用失败，请检查密钥、网络和服务日志。")
            )
            display[-1]["content"] = (answer + "\n\n⚠️ " + message).strip()
            yield display, history, query, rows, trace, links

    def search(self, query, retrieval_k, rerank_k, use_rerank):
        try:
            trace = self.services.retriever.search(
                SearchRequest(
                    query=query,
                    retrieval_k=int(retrieval_k),
                    rerank_k=int(rerank_k),
                    use_rerank=use_rerank,
                )
            )
            ranks = {s["id"]: s["citation"] for s in trace["sources"]}
            rows = [
                [
                    c["retrieval_rank"],
                    ranks.get(c["id"], "未入选"),
                    c["filename"],
                    "、".join(map(str, c["pages"])) or "—",
                    round(c["similarity"], 4),
                    round(c["rerank_score"], 4) if c["rerank_score"] is not None else "—",
                    c["text"],
                ]
                for c in trace["candidates"]
            ]
            return rows, trace
        except ValueError as error:
            raise gr.Error(str(error)) from error
