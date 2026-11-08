import io
from pathlib import Path
from threading import RLock
from types import SimpleNamespace

from ragkb.ingestion import IngestionService
from ragkb.repository import DocumentRepository
from ragkb.schemas import Element
from tests_support import CharacterTokenizer


class FakeStore:
    def __init__(self, fail=False):
        self.items, self.deleted, self.fail = {}, [], fail

    def add(self, chunks, vectors):
        for chunk in chunks:
            self.items[chunk.id] = chunk
        if self.fail:
            raise RuntimeError("simulated partial vector write")

    def delete(self, doc_id):
        self.deleted.append(doc_id)
        self.items = {k: v for k, v in self.items.items() if v.document_id != doc_id}


def service(tmp_path, fail=False):
    (tmp_path / "uploads").mkdir()
    (tmp_path / "parsed").mkdir()
    settings = SimpleNamespace(
        data_root=tmp_path, rag_max_upload_mb=1, rag_chunk_tokens=160, rag_chunk_overlap=16
    )
    models = SimpleNamespace(
        lock=RLock(), unload=lambda: None, tokenizer=lambda: CharacterTokenizer()
    )
    parser = SimpleNamespace(parse=lambda path, output: [Element("课程演示需要来源证据。")])
    embedding = SimpleNamespace(documents=lambda texts: [[1.0, 0.0] for text in texts])
    repo = DocumentRepository(tmp_path / "catalog.db")
    store = FakeStore(fail)
    return IngestionService(settings, repo, parser, embedding, store, models)


def test_duplicate_upload_and_complete_delete(tmp_path):
    s = service(tmp_path)
    first = s.submit(io.BytesIO("同一文档".encode()), "课程.txt")
    s.executor.shutdown(wait=True)
    second = s.submit(io.BytesIO("同一文档".encode()), "重命名.txt")
    assert second["duplicate"] and second["id"] == first["id"]
    assert len(s.repository.list()) == 1 and s.store.items
    path = Path(s.repository.get(first["id"])["path"])
    s.delete(first["id"])
    assert not path.exists() and not s.store.items and not s.repository.list()


def test_partial_vector_write_rolls_back_and_is_retryable(tmp_path):
    s = service(tmp_path, fail=True)
    doc = s.submit(io.BytesIO(b"test"), "failure.txt")
    s.close()
    assert s.repository.get(doc["id"])["status"] == "failed"
    assert not s.store.items
    assert not s.repository.chunks(doc["id"])


def test_upload_validation_cleans_temporary_files(tmp_path):
    import pytest

    s = service(tmp_path)
    with pytest.raises(ValueError, match="空文件"):
        s.submit(io.BytesIO(b""), "empty.txt")
    with pytest.raises(ValueError, match="不能超过"):
        s.submit(io.BytesIO(b"x" * (1024 * 1024 + 1)), "large.txt")
    assert list((tmp_path / "uploads").iterdir()) == []
    s.close()
