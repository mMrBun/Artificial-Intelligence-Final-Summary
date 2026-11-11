import json
import logging
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from .schemas import ChatRequest, SearchRequest

logger = logging.getLogger(__name__)


def public_document(doc):
    return {key: value for key, value in doc.items() if key != "path"}


def build_router(services):
    router = APIRouter(prefix="/api")

    @router.get("/health")
    def health():
        return services.health()

    @router.get("/documents")
    def documents():
        return [public_document(d) for d in services.repository.list()]

    @router.post("/documents", status_code=202)
    def upload(file: UploadFile = File(...)):
        try:
            return public_document(services.ingestion.submit(file.file, file.filename or ""))
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        finally:
            file.file.close()

    @router.get("/documents/{doc_id}")
    def document(doc_id: str):
        doc = services.repository.get(doc_id)
        if not doc:
            raise HTTPException(404, "文档不存在。")
        return public_document(doc)

    @router.get("/documents/{doc_id}/chunks")
    def chunks(doc_id: str, offset: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=500)):
        document(doc_id)
        return services.repository.chunks(doc_id, offset, limit)

    @router.get("/documents/{doc_id}/file")
    def download(doc_id: str):
        doc = services.repository.get(doc_id)
        if not doc:
            raise HTTPException(404, "文档不存在。")
        return FileResponse(Path(doc["path"]), filename=doc["filename"])

    @router.delete("/documents/{doc_id}")
    def delete(doc_id: str):
        try:
            services.ingestion.delete(doc_id)
        except KeyError as error:
            raise HTTPException(404, "文档不存在。") from error
        except ValueError as error:
            raise HTTPException(409, str(error)) from error
        return {"deleted": doc_id}

    @router.post("/search")
    def search(request: SearchRequest):
        try:
            return services.retriever.search(request)
        except ValueError as error:
            raise HTTPException(400, str(error)) from error

    @router.post("/chat/stream")
    def chat(request: ChatRequest):
        if not request.query.strip() or request.rerank_k > request.retrieval_k:
            raise HTTPException(400, "问题不能为空，且保留数量不能超过召回数量。")
        try:
            services.llm.ensure_configured()
        except ValueError as error:
            raise HTTPException(503, str(error)) from error

        def events():
            try:
                for event in services.qa.stream(request):
                    yield (
                        "event: "
                        + event["event"]
                        + "\ndata: "
                        + json.dumps(event["data"], ensure_ascii=False)
                        + "\n\n"
                    )
            except Exception as error:
                logger.exception("Chat failed")
                message = (
                    str(error)
                    if isinstance(error, ValueError)
                    else ("问答调用失败，请检查密钥、模型配置、网络和服务日志。")
                )
                yield (
                    "event: error\ndata: "
                    + json.dumps({"message": message}, ensure_ascii=False)
                    + "\n\n"
                )

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return router
