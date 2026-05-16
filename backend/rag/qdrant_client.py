from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PayloadSchemaType,
    SparseVectorParams,
    VectorParams,
)

from .config import (
    COLLECTION_NAME,
    DENSE_DIM,
    DENSE_VECTOR_NAME,
    QDRANT_API_KEY,
    QDRANT_URL,
    SPARSE_VECTOR_NAME,
)

_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return _client


def create_collection(collection_name: str | None = None) -> None:
    name = collection_name or COLLECTION_NAME
    client = get_client()

    if client.collection_exists(name):
        client.delete_collection(name)

    client.create_collection(
        collection_name=name,
        vectors_config={
            DENSE_VECTOR_NAME: VectorParams(size=DENSE_DIM, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: SparseVectorParams(),
        },
    )
    # Payload index needed so retriever.py can filter by mountain_range.
    client.create_payload_index(
        collection_name=name,
        field_name="mountain_range",
        field_schema=PayloadSchemaType.KEYWORD,
    )
