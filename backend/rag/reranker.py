from threading import Lock

from sentence_transformers import CrossEncoder

from .config import RERANKER_MODEL, TOP_K_RERANK


class Reranker:
    _instance = None
    _lock = Lock()

    @classmethod
    def get_instance(cls) -> "Reranker":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.model = CrossEncoder(RERANKER_MODEL)

    def rerank(self, query: str, hits: list, top_k: int = TOP_K_RERANK) -> list:
        if not hits:
            return []

        pairs = [(query, (h.payload or {}).get("text", "")) for h in hits]
        scores = self.model.predict(pairs)

        scored = sorted(
            zip(hits, scores), key=lambda pair: pair[1], reverse=True
        )[:top_k]

        return [hit.model_copy(update={"score": float(score)}) for hit, score in scored]
