from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
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


def create_collection(collection_name: str | None = None, recreate: bool = False) -> None:
    name = collection_name or COLLECTION_NAME
    client = get_client()

    if client.collection_exists(name):
        if not recreate:
            print(
                f"Collection {name!r} already exists. "
                "Set RECREATE_QDRANT_COLLECTION=1 to delete and rebuild it."
            )
            return
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
    create_payload_indexes(name)


def create_payload_indexes(collection_name: str | None = None) -> None:
    name = collection_name or COLLECTION_NAME
    client = get_client()

    for field_name in ("entity_type", "poi_type", "region", "difficulty"):
        try:
            client.create_payload_index(
                collection_name=name,
                field_name=field_name,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except UnexpectedResponse as exc:
            if "already exists" not in str(exc).casefold():
                raise
