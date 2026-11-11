from .model_manager import ModelManager


class EmbeddingService:
    def __init__(self, models: ModelManager):
        self.models = models

    def documents(self, texts: list[str]) -> list[list[float]]:
        with self.models.lock:
            return (
                self.models.embedding()
                .encode(texts, batch_size=4, normalize_embeddings=True, show_progress_bar=False)
                .tolist()
            )

    def query(self, text: str) -> list[float]:
        with self.models.lock:
            return (
                self.models.embedding()
                .encode(
                    [text], prompt_name="query", normalize_embeddings=True, show_progress_bar=False
                )[0]
                .tolist()
            )
