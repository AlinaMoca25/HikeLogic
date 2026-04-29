from dataclasses import dataclass

from .config import TOP_K_RERANK, TOP_K_RETRIEVE
from .reranker import Reranker
from .retriever import hybrid_search


@dataclass
class Hit:
    text: str
    score: float
    metadata: dict


def search(query: str) -> list[Hit]:
    candidates = hybrid_search(query, limit=TOP_K_RETRIEVE)
    reranker = Reranker.get_instance()
    reranked = reranker.rerank(query, candidates, top_k=TOP_K_RERANK)

    hits: list[Hit] = []
    for h in reranked:
        payload = dict(h.payload or {})
        text = payload.pop("text", "")
        hits.append(Hit(text=text, score=h.score, metadata=payload))
    return hits
