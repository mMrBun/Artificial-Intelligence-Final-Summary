import logging
from contextlib import asynccontextmanager

import gradio as gr
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from .api import build_router
from .config import Settings
from .container import AppContainer
from .ui import DemoUI

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def create_app(services=None):
    services = services or AppContainer()

    @asynccontextmanager
    async def lifespan(app):
        yield
        services.close()

    app = FastAPI(title="课程 RAG 知识库", version="0.1.0", lifespan=lifespan)
    app.state.services = services
    app.include_router(build_router(services))

    @app.get("/", include_in_schema=False)
    def home():
        return RedirectResponse("/ui/")

    app = gr.mount_gradio_app(
        app,
        DemoUI(services).build(),
        path="/ui",
        max_file_size=f"{services.settings.rag_max_upload_mb}mb",
    )
    return app


def main():
    import uvicorn

    settings = Settings()
    uvicorn.run(
        create_app(),
        host=settings.rag_host,
        port=settings.rag_port,
        workers=1,
        timeout_graceful_shutdown=15,
        log_level="info",
    )


if __name__ == "__main__":
    main()
