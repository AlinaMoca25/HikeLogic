from typing import Any, Literal

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    mode: Literal["full", "retrieve"] = "full"


class SourceItem(BaseModel):
    index: int
    text: str
    score: float
    metadata: dict[str, Any]


class AskResponse(BaseModel):
    query: str
    answer: str
    sources: list[SourceItem]
    abstained: bool
    mode: str


class HealthResponse(BaseModel):
    status: str
    service: str
