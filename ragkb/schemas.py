from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field


@dataclass
class Element:
    text: str
    kind: str = "text"
    page: int | None = None
    heading_level: int = 0


@dataclass
class Chunk:
    id: str
    document_id: str
    filename: str
    text: str
    embedding_text: str
    heading: str
    index: int
    pages: list[int] = field(default_factory=list)
    kind: str = "text"
    tokens: int = 0


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=16000)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    retrieval_k: int = Field(default=20, ge=1, le=100)
    rerank_k: int = Field(default=5, ge=1, le=20)
    use_rerank: bool = True


class ChatRequest(SearchRequest):
    history: list[HistoryMessage] = Field(default_factory=list, max_length=20)
