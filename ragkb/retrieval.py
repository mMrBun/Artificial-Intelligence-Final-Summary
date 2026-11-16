import time

from .schemas import SearchRequest


class Retriever:
    def __init__(self, embeddings, store, reranker, models):
        self.embeddings, self.store = embeddings, store
        self.reranker, self.models = reranker, models

    def search(self, request: SearchRequest):
        if not request.query.strip():
            raise ValueError("请输入有效问题。")
        if request.rerank_k > request.retrieval_k:
            raise ValueError("最终保留数量不能超过初始召回数量。")
        start = time.perf_counter()
        with self.models.lock:
            if not self.store.count():
                return {
                    "query": request.query,
                    "candidates": [],
                    "sources": [],
                    "timings": {"retrieval_ms": 0, "rerank_ms": 0, "total_ms": 0},
                }
            vector = self.embeddings.query(request.query)
            candidates = self.store.search(vector, request.retrieval_k)
            retrieved = time.perf_counter()
            if request.use_rerank:
                scores = self.reranker.score(
                    request.query, [item["embedding_text"] for item in candidates]
                )
                for candidate, score in zip(candidates, scores):
                    candidate["rerank_score"] = score
                ordered = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
            else:
                ordered = list(candidates)
            sources = [
                {**item, "citation": i + 1} for i, item in enumerate(ordered[: request.rerank_k])
            ]
        end = time.perf_counter()
        return {
            "query": request.query,
            "candidates": candidates,
            "sources": sources,
            "timings": {
                "retrieval_ms": round((retrieved - start) * 1000, 1),
                "rerank_ms": round((end - retrieved) * 1000, 1),
                "total_ms": round((end - start) * 1000, 1),
            },
        }
