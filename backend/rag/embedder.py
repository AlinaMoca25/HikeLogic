import uuid

from qdrant_client.models import PointStruct, SparseVector

from .config import COLLECTION_NAME, DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME
from .embeddings import BGEM3Embedder
from .qdrant_client import get_client


def upsert_trail_data(trail_metadata: dict, body_content: str) -> None:
    client = get_client()
    embedder = BGEM3Embedder.get_instance()
    embedded = embedder.embed_query(body_content)

    point_id = trail_metadata.get("id")
    if not point_id:
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, trail_metadata.get("name", "")))

    point = PointStruct(
        id=point_id,
        vector={
            DENSE_VECTOR_NAME: embedded["dense"],
            SPARSE_VECTOR_NAME: SparseVector(
                indices=embedded["sparse"]["indices"],
                values=embedded["sparse"]["values"],
            ),
        },
        payload={
            "text": body_content,
            "name": trail_metadata.get("name"),
            "difficulty": trail_metadata.get("difficulty"),
            "region": trail_metadata.get("region"),
            "marking": trail_metadata.get("marking"),
        },
    )

    client.upsert(collection_name=COLLECTION_NAME, points=[point])
