import hashlib
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

from .chunking import StructureChunker


class IngestionService:
    SUPPORTED = {".pdf", ".docx", ".md", ".txt", ".png", ".jpg", ".jpeg"}

    def __init__(self, settings, repository, parser, embeddings, store, models):
        self.settings, self.repository, self.parser = settings, repository, parser
        self.embeddings, self.store, self.models = embeddings, store, models
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ingestion")
        self.submission_lock = Lock()
        # A restart never leaves documents falsely marked as processing/complete.
        for doc in repository.list():
            if doc["status"] not in {"ready", "failed"}:
                store.delete(doc["id"])
                repository.update(doc["id"], "failed", "处理被中断", "服务重启，请重新上传。")

    def submit(self, stream, filename):
        filename = Path(filename.replace("\\", "/")).name
        suffix = Path(filename).suffix.lower()
        if suffix not in self.SUPPORTED:
            raise ValueError("支持 PDF、DOCX、Markdown、UTF-8 TXT、PNG/JPG。")
        if not filename or len(filename) > 180:
            raise ValueError("文件名为空或过长。")
        temporary = self.settings.data_root / "uploads" / (".upload-" + uuid.uuid4().hex)
        digest, size = hashlib.sha256(), 0
        try:
            with temporary.open("wb") as out:
                while block := stream.read(1024 * 1024):
                    size += len(block)
                    if size > self.settings.rag_max_upload_mb * 1024 * 1024:
                        raise ValueError(f"单个文件不能超过 {self.settings.rag_max_upload_mb} MB。")
                    digest.update(block)
                    out.write(block)
            if not size:
                raise ValueError("不能上传空文件。")
            doc_id = digest.hexdigest()
            with self.submission_lock:
                previous = self.repository.get(doc_id)
                if previous and previous["status"] != "failed":
                    return {**previous, "duplicate": True}
                if previous:
                    self.delete(doc_id)
                path = self.settings.data_root / "uploads" / (doc_id + suffix)
                temporary.replace(path)
                self.repository.add(doc_id, filename, path)
                self.executor.submit(self._process, doc_id)
            return {**self.repository.get(doc_id), "duplicate": False}
        finally:
            temporary.unlink(missing_ok=True)

    def _process(self, doc_id):
        with self.models.lock:
            doc = self.repository.get(doc_id)
            output = self.settings.data_root / "parsed" / doc_id
            try:
                self.repository.update(doc_id, "processing", "MinerU 本地解析")
                self.models.unload()
                elements = self.parser.parse(Path(doc["path"]), output)
                self.repository.update(doc_id, "processing", "按文档结构切分")
                chunker = StructureChunker(
                    self.models.tokenizer(),
                    self.settings.rag_chunk_tokens,
                    self.settings.rag_chunk_overlap,
                )
                chunks = chunker.split(elements, doc_id, doc["filename"])
                if not chunks:
                    raise ValueError("未生成有效文本块。")
                vectors = []
                for start in range(0, len(chunks), 16):
                    self.repository.update(
                        doc_id, "processing", f"向量化 {min(start + 16, len(chunks))}/{len(chunks)}"
                    )
                    vectors.extend(
                        self.embeddings.documents(
                            [c.embedding_text for c in chunks[start : start + 16]]
                        )
                    )
                self.repository.update(doc_id, "processing", "写入 Chroma")
                self.store.add(chunks, vectors)
                self.repository.save_chunks(doc_id, chunks)
                self.repository.update(doc_id, "ready", "已就绪", count=len(chunks))
            except Exception as error:
                self.store.delete(doc_id)
                self.repository.update(doc_id, "failed", "处理失败", str(error)[-2000:])

    def delete(self, doc_id):
        with self.models.lock:
            doc = self.repository.get(doc_id)
            if not doc:
                raise KeyError("文档不存在。")
            if doc["status"] in {"queued", "processing"}:
                raise ValueError("文档正在处理，请完成后再删除。")
            self.store.delete(doc_id)
            self.repository.delete(doc_id)
            Path(doc["path"]).unlink(missing_ok=True)
            shutil.rmtree(self.settings.data_root / "parsed" / doc_id, ignore_errors=True)

    def close(self):
        self.executor.shutdown(wait=True)
