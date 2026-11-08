import pytest

from ragkb.repository import DocumentRepository
from ragkb.schemas import Chunk
from ragkb.vector_store import ChromaStore


def test_persistent_vectors_delete_only_target_document(tmp_path):
    store = ChromaStore(tmp_path / "chroma", tmp_path / "model")
    chunks = [
        Chunk("a:0", "a", "a.txt", "苹果", "苹果", "", 0, [1]),
        Chunk("b:0", "b", "b.txt", "香蕉", "香蕉", "", 0, [2]),
    ]
    store.add(chunks, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    reopened = ChromaStore(tmp_path / "chroma", tmp_path / "model")
    assert reopened.search([1.0, 0.0, 0.0], 20)[0]["document_id"] == "a"
    reopened.delete("a")
    assert [x["document_id"] for x in reopened.search([1.0, 0.0, 0.0], 20)] == ["b"]
    reopened.delete("b")
    assert reopened.search([1.0, 0.0, 0.0], 20) == []


def test_model_signature_change_requires_reindex(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"version":1}')
    store = ChromaStore(tmp_path / "chroma", model)
    store.add([Chunk("a:0", "a", "a", "text", "text", "", 0)], [[1.0, 0.0]])
    (model / "config.json").write_text('{"version":2}')
    with pytest.raises(RuntimeError, match="重新建立索引"):
        ChromaStore(tmp_path / "chroma", model)


def test_catalog_deletion_cascades_and_persists(tmp_path):
    repo = DocumentRepository(tmp_path / "catalog.db")
    repo.add("a", "a.txt", tmp_path / "a.txt")
    repo.save_chunks("a", [Chunk("a:0", "a", "a.txt", "内容", "内容", "", 0)])
    assert len(DocumentRepository(tmp_path / "catalog.db").chunks("a")) == 1
    repo.delete("a")
    assert repo.get("a") is None
    assert repo.chunks("a") == []
