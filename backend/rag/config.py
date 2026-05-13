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
RERANKER_DEVICE = os.getenv("RERANKER_DEVICE", "cpu")
RERANKER_MAX_CHARS = int(os.getenv("RERANKER_MAX_CHARS", "1800"))
MIN_RERANK_SCORE_FOR_GENERATION = float(
    os.getenv("MIN_RERANK_SCORE_FOR_GENERATION", "0.001")
)

TOP_K_RETRIEVE = 20
TOP_K_RERANK = 5

HF_TOKEN = os.getenv("HF_TOKEN")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "alinamoca25/hikelogic-qwen2.5-1.5b")
GENERATION_PROVIDER = os.getenv("GENERATION_PROVIDER", "auto")
GENERATION_MAX_TOKENS = int(os.getenv("GENERATION_MAX_TOKENS", "512"))
GENERATION_TEMPERATURE = float(os.getenv("GENERATION_TEMPERATURE", "0.1"))
