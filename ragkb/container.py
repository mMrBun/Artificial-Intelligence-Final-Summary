from .config import Settings
from .embedding import EmbeddingService
from .ingestion import IngestionService
from .llm import DeepSeekClient
from .model_manager import ModelManager
from .parsing import MinerUParser
from .qa import QAService
from .repository import DocumentRepository
from .reranker import Reranker
from .retrieval import Retriever
from .vector_store import ChromaStore


class AppContainer:
    """One shared service graph for the API and UI; never duplicate GPU models."""

    def __init__(self, settings=None):
        self.settings = settings or Settings()
        self.settings.prepare()
        self.models = ModelManager(self.settings)
        self.repository = DocumentRepository(self.settings.data_root / "catalog.sqlite3")
        self.parser = MinerUParser(self.settings)
        self.embedding = EmbeddingService(self.models)
        self.store = ChromaStore(self.settings.data_root / "chroma", self.settings.embedding_path)
        self.reranker = Reranker(self.models)
        self.retriever = Retriever(self.embedding, self.store, self.reranker, self.models)
        self.ingestion = IngestionService(
            self.settings, self.repository, self.parser, self.embedding, self.store, self.models
        )
        self.llm = DeepSeekClient()
        self.qa = QAService(self.retriever, self.llm)

    def health(self):
        import importlib.metadata

        return {
            "status": "ok",
            "documents": len(self.repository.list()),
            "chunks": self.store.count(),
            "deepseek_configured": bool(Settings().deepseek_api_key.strip()),
            "deepseek_model": Settings().deepseek_model,
            "embedding_ready": self.settings.weights_ready(self.settings.embedding_path),
            "reranker_ready": self.settings.weights_ready(self.settings.reranker_path),
            "device": self.settings.rag_device,
            "versions": {
                p: importlib.metadata.version(p)
                for p in ("mineru", "gradio", "chromadb", "sentence-transformers")
            },
        }

    def close(self):
        self.ingestion.close()
        self.models.unload()
