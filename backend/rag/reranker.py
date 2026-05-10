from threading import Lock
import warnings

from sentence_transformers import CrossEncoder

from .config import RERANKER_DEVICE, RERANKER_MAX_CHARS, RERANKER_MODEL, TOP_K_RERANK


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
        self.model = CrossEncoder(RERANKER_MODEL, device=RERANKER_DEVICE)

    @staticmethod
    def _payload_text(hit) -> str:
        return ((hit.payload or {}).get("text") or "").strip()[:RERANKER_MAX_CHARS]

    @staticmethod
    def _fallback(hits: list, top_k: int) -> list:
        return hits[:top_k]

    def rerank(self, query: str, hits: list, top_k: int = TOP_K_RERANK) -> list:
        if not hits:
            return []

        pairs = [(query, self._payload_text(hit)) for hit in hits]
        try:
            scores = self.model.predict(pairs)
        except Exception as exc:
            warnings.warn(f"Reranking failed; using fused retrieval order: {exc}")
            return self._fallback(hits, top_k)

        scored = sorted(
            zip(hits, scores), key=lambda pair: pair[1], reverse=True
        )[:top_k]

        return [hit.model_copy(update={"score": float(score)}) for hit, score in scored]
