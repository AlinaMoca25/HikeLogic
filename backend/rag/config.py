import os
from pathlib import Path
from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(ENV_PATH)


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Missing required env var '{name}'. Set it in {ENV_PATH}."
        )
    return value


QDRANT_URL = _require("QDRANT_URL")
QDRANT_API_KEY = _require("QDRANT_API_KEY")

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "hike_logic_romania")

DENSE_VECTOR_NAME = os.getenv("DENSE_VECTOR_NAME", "dense")
SPARSE_VECTOR_NAME = os.getenv("SPARSE_VECTOR_NAME", "sparse")

DENSE_MODEL = "BAAI/bge-m3"
DENSE_DIM = 1024

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

TOP_K_RETRIEVE = 20
TOP_K_RERANK = 5
