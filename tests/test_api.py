from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ragkb.api import build_router
from ragkb.repository import DocumentRepository


class MissingKey:
    def ensure_configured(self):
        raise ValueError("尚未配置密钥")


class BrokenQA:
    def stream(self, request):
        yield {"event": "sources", "data": {"sources": []}}
        raise RuntimeError("upstream failed")


def make_client(tmp_path, configured=False):
    llm = SimpleNamespace(ensure_configured=lambda: None) if configured else MissingKey()
    services = SimpleNamespace(
        repository=DocumentRepository(tmp_path / "catalog.db"),
        llm=llm,
        qa=BrokenQA(),
        health=lambda: {"status": "ok"},
    )
    app = FastAPI()
    app.include_router(build_router(services))
    return TestClient(app), services


def test_missing_key_does_not_start_stream(tmp_path):
    client, _ = make_client(tmp_path)
    response = client.post("/api/chat/stream", json={"query": "问题"})
    assert response.status_code == 503
    assert "尚未配置" in response.json()["detail"]


def test_invalid_limits_rejected_before_stream(tmp_path):
    client, _ = make_client(tmp_path, True)
    assert (
        client.post(
            "/api/chat/stream", json={"query": "问题", "retrieval_k": 2, "rerank_k": 5}
        ).status_code
        == 400
    )
    assert client.post("/api/chat/stream", json={"query": "   "}).status_code == 400


def test_sse_error_is_explicit_and_does_not_emit_done(tmp_path):
    client, _ = make_client(tmp_path, True)
    response = client.post("/api/chat/stream", json={"query": "问题"})
    assert "event: sources" in response.text
    assert "event: error" in response.text
    assert "event: done" not in response.text


def test_catalog_hides_server_paths_and_missing_doc_returns_404(tmp_path):
    client, services = make_client(tmp_path)
    services.repository.add("a", "课程.txt", tmp_path / "secret-location.txt")
    assert "path" not in client.get("/api/documents").json()[0]
    assert client.get("/api/documents/missing").status_code == 404
