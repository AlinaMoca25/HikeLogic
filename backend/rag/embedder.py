import uuid

from qdrant_client.models import PointStruct, SparseVector

from .config import COLLECTION_NAME, DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME
from .embeddings import BGEM3Embedder
from .qdrant_client import get_client


def _point_from_embedding(trail_metadata: dict, body_content: str, embedded: dict) -> PointStruct:
    point_id = trail_metadata.get("id")
    if not point_id:
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, trail_metadata.get("name", "")))

    payload = {
        key: value
        for key, value in trail_metadata.items()
        if key != "id" and value is not None
    }
    payload["text"] = body_content

    return PointStruct(
        id=point_id,
        vector={
            DENSE_VECTOR_NAME: embedded["dense"],
            SPARSE_VECTOR_NAME: SparseVector(
                indices=embedded["sparse"]["indices"],
                values=embedded["sparse"]["values"],
            ),
        },
        payload=payload,
    )


def upsert_trail_data(trail_metadata: dict, body_content: str) -> None:
    upsert_trail_batch([(trail_metadata, body_content)])


def upsert_trail_batch(items: list[tuple[dict, str]]) -> None:
    if not items:
        return

    client = get_client()
    embedder = BGEM3Embedder.get_instance()
    texts = [body_content for _, body_content in items]
    embeddings = embedder.embed_texts(texts)
    points = [
        _point_from_embedding(trail_metadata, body_content, embedded)
        for (trail_metadata, body_content), embedded in zip(items, embeddings)
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)
