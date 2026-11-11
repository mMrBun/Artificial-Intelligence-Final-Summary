"""Central paths and settings; model files never go to the user's global cache."""

import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-flash"
    rag_device: str = "cuda"
    rag_chunk_tokens: int = Field(default=512, ge=128, le=2048)
    rag_chunk_overlap: int = Field(default=64, ge=0, le=128)
    rag_retrieval_k: int = Field(default=20, ge=1, le=100)
    rag_rerank_k: int = Field(default=5, ge=1, le=20)
    rag_max_upload_mb: int = Field(default=30, ge=1, le=200)
    rag_host: str = "0.0.0.0"
    rag_port: int = 8000

    @property
    def model_root(self) -> Path:
        return ROOT / "models"

    @property
    def embedding_path(self) -> Path:
        return self.model_root / "Qwen3-Embedding-0.6B"

    @property
    def reranker_path(self) -> Path:
        return self.model_root / "Qwen3-Reranker-0.6B"

    @property
    def data_root(self) -> Path:
        return ROOT / "data"

    @staticmethod
    def weights_ready(path: Path) -> bool:
        if not (path / "config.json").is_file() or not (path / "tokenizer.json").is_file():
            return False
        if (path / "model.safetensors").is_file():
            return True
        index = path / "model.safetensors.index.json"
        if index.is_file():
            import json

            shards = set(json.loads(index.read_text())["weight_map"].values())
            return bool(shards) and all((path / name).is_file() for name in shards)
        return False

    def prepare(self) -> None:
        for path in (
            self.model_root,
            self.data_root / "uploads",
            self.data_root / "parsed",
            self.data_root / "chroma",
            ROOT / "logs",
        ):
            path.mkdir(parents=True, exist_ok=True)
        defaults = {
            "MINERU_HOME": str(ROOT / "config" / "mineru"),
            "MINERU_CONFIG": str(ROOT / "config" / "mineru.yaml"),
            "MINERU_MODEL_SOURCE": "local",
            "MINERU_MODEL_SMALL_BACKEND": "torch",
            "MINERU_TELEMETRY_DISABLED": "1",
            "HF_HOME": str(self.model_root / "_cache" / "huggingface"),
            "MODELSCOPE_CACHE": str(self.model_root / "_cache" / "modelscope"),
            "TORCH_HOME": str(self.model_root / "_cache" / "torch"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "ANONYMIZED_TELEMETRY": "false",
            "GRADIO_ANALYTICS_ENABLED": "false",
            "HF_HUB_DISABLE_TELEMETRY": "1",
        }
        for key, value in defaults.items():
            os.environ[key] = value
