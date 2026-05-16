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

    def embed_query(self, text: str) -> dict:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[dict]:
        if not texts:
            return []
        output = self.model.encode(
            texts,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
            batch_size=batch_size,
        )
        results = []
        for i in range(len(texts)):
            sparse_dict = output["lexical_weights"][i]
            results.append({
                "dense": output["dense_vecs"][i].tolist(),
                "sparse": {
                    "indices": [int(k) for k in sparse_dict.keys()],
                    "values": [float(v) for v in sparse_dict.values()],
                },
            })
        return results
