import logging
import time

from .schemas import SearchRequest

logger = logging.getLogger(__name__)


class QAService:
    def __init__(self, retriever, llm):
        self.retriever, self.llm = retriever, llm

    def stream(self, request):
        started = time.perf_counter()
        self.llm.ensure_configured()
        query = request.query
        rewrite_note = ""
        if request.history:
            try:
                query = self.llm.rewrite(query, request.history)
            except Exception:
                # Explicit fallback keeps the previous user topic available for pronouns.
                previous = next(
                    (m.content for m in reversed(request.history) if m.role == "user"), ""
                )
                query = (previous[:1000] + "\n" + request.query).strip()
                rewrite_note = "问题改写暂不可用，已使用上一轮问题辅助检索。"
        trace = self.retriever.search(
            SearchRequest(
                query=query,
                retrieval_k=request.retrieval_k,
                rerank_k=request.rerank_k,
                use_rerank=request.use_rerank,
            )
        )
        trace["original_query"] = request.query
        trace["rewrite_note"] = rewrite_note
        yield {"event": "sources", "data": trace}
        if not trace["sources"]:
            yield {"event": "token", "data": "当前知识库没有可用资料，请先上传并完成文档处理。"}
        else:
            for token in self.llm.stream(request.query, request.history, trace["sources"]):
                yield {"event": "token", "data": token}
        yield {
            "event": "done",
            "data": {"total_ms": round((time.perf_counter() - started) * 1000, 1)},
        }
