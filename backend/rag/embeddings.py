from threading import Lock

from FlagEmbedding import BGEM3FlagModel

from .config import DENSE_MODEL


class BGEM3Embedder:
    _instance = None
    _lock = Lock()

    @classmethod
    def get_instance(cls) -> "BGEM3Embedder":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.model = BGEM3FlagModel(DENSE_MODEL, use_fp16=True)

    @staticmethod
    def _format_sparse(sparse_dict: dict) -> dict:
        return {
            "indices": [int(k) for k in sparse_dict.keys()],
            "values": [float(v) for v in sparse_dict.values()],
        }

    def embed_texts(self, texts: list[str]) -> list[dict]:
        output = self.model.encode(
            texts,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        embeddings = []
        for dense_vec, sparse_dict in zip(output["dense_vecs"], output["lexical_weights"]):
            dense = dense_vec.tolist() if hasattr(dense_vec, "tolist") else list(dense_vec)
            embeddings.append(
                {
                    "dense": dense,
                    "sparse": self._format_sparse(sparse_dict),
                }
            )
        return embeddings

    def embed_query(self, text: str) -> dict:
        return self.embed_texts([text])[0]
