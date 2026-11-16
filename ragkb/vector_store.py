"""Persistent Chroma collection with an explicit embedding signature."""

import hashlib
import json
from pathlib import Path

from .schemas import Chunk


class ChromaStore:
    def __init__(self, path: Path, embedding_path: Path):
        import chromadb
        from chromadb.config import Settings

        self.client = chromadb.PersistentClient(
            path=str(path), settings=Settings(anonymized_telemetry=False)
        )
        config_file = embedding_path / "config.json"
        digest = (
            hashlib.sha256(config_file.read_bytes()).hexdigest()
            if config_file.exists()
            else "not-downloaded"
        )
        self.signature = f"Qwen3-Embedding-0.6B:1024:normalized:query-prompt:{digest}"
        self.collection = self.client.get_or_create_collection(
            "knowledge_base_v1",
            embedding_function=None,
            metadata={"hnsw:space": "cosine", "embedding_signature": self.signature},
        )
        current = (self.collection.metadata or {}).get("embedding_signature")
        if current != self.signature:
            if self.collection.count():
                raise RuntimeError("向量库模型配置已变化，请使用新数据目录重新建立索引。")
            self.collection.modify(
                metadata={"hnsw:space": "cosine", "embedding_signature": self.signature}
            )

    def add(self, chunks: list[Chunk], embeddings):
        for start in range(0, len(chunks), 64):
            group = chunks[start : start + 64]
            self.collection.upsert(
                ids=[c.id for c in group],
                embeddings=embeddings[start : start + 64],
                documents=[c.text for c in group],
                metadatas=[
                    {
                        "document_id": c.document_id,
                        "filename": c.filename,
                        "heading": c.heading,
                        "chunk_index": c.index,
                        "pages": json.dumps(c.pages),
                        "kind": c.kind,
                        "tokens": c.tokens,
                        "embedding_text": c.embedding_text,
                    }
                    for c in group
                ],
            )

    def search(self, vector, k):
        size = self.collection.count()
        if not size:
            return []
        data = self.collection.query(
            query_embeddings=[vector],
            n_results=min(k, size),
            include=["documents", "metadatas", "distances"],
        )
        result = []
        for chunk_id, text, meta, distance in zip(
            data["ids"][0], data["documents"][0], data["metadatas"][0], data["distances"][0]
        ):
            result.append(
                {
                    "id": chunk_id,
                    "text": text,
                    **meta,
                    "pages": json.loads(meta["pages"]),
                    "similarity": 1.0 - float(distance),
                    "retrieval_rank": len(result) + 1,
                    "rerank_score": None,
                }
            )
        return result

    def delete(self, doc_id):
        self.collection.delete(where={"document_id": doc_id})

    def count(self):
        return self.collection.count()
