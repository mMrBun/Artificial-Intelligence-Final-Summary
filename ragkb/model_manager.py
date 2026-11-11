"""Serialize GPU work and explicitly release models before MinerU subprocesses."""

import gc
from threading import RLock

from .config import Settings


class ModelManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.lock = RLock()
        self._embedding = None
        self._reranker = None
        self._tokenizer = None

    def tokenizer(self):
        with self.lock:
            if self._tokenizer is None:
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(
                    str(self.settings.embedding_path), local_files_only=True
                )
            return self._tokenizer

    def embedding(self):
        with self.lock:
            if self._embedding is None:
                import torch
                from sentence_transformers import SentenceTransformer

                self._embedding = SentenceTransformer(
                    str(self.settings.embedding_path),
                    device=self.settings.rag_device,
                    local_files_only=True,
                    model_kwargs={
                        "torch_dtype": torch.float16
                        if self.settings.rag_device == "cuda"
                        else torch.float32
                    },
                )
                self._embedding.max_seq_length = 2048
            return self._embedding

    def reranker(self):
        with self.lock:
            if self._reranker is None:
                # Qwen's official yes/no likelihood scoring, not generic sequence classification.
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(
                    str(self.settings.reranker_path), padding_side="left", local_files_only=True
                )
                model = (
                    AutoModelForCausalLM.from_pretrained(
                        str(self.settings.reranker_path),
                        local_files_only=True,
                        torch_dtype=torch.float16
                        if self.settings.rag_device == "cuda"
                        else torch.float32,
                        attn_implementation="sdpa",
                    )
                    .to(self.settings.rag_device)
                    .eval()
                )
                self._reranker = (tokenizer, model)
            return self._reranker

    def unload(self) -> None:
        with self.lock:
            self._embedding = None
            self._reranker = None
            gc.collect()
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
