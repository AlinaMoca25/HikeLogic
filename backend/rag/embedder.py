import uuid

from qdrant_client.models import PointStruct, SparseVector
from tqdm import tqdm

from .config import COLLECTION_NAME, DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME
from .embeddings import BGEM3Embedder
from .qdrant_client import get_client


def _make_point(point_id: str, metadata: dict, body_content: str, embedded: dict) -> PointStruct:
    return PointStruct(
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
            "name": metadata.get("name"),
            "type": metadata.get("type"),
            "difficulty": metadata.get("difficulty"),
            "region": metadata.get("region"),
            "mountain_range": metadata.get("mountain_range"),
            "marking": metadata.get("marking"),
            "lat": metadata.get("lat"),
            "lon": metadata.get("lon"),
            "osm_url": metadata.get("osm_url"),
        },
    )


def upsert_batch(
    items: list[tuple[str, dict, str]],
    chunk_size: int = 200,
    embed_batch_size: int = 32,
) -> None:
    if not items:
        return
    client = get_client()
    embedder = BGEM3Embedder.get_instance()
    for start in tqdm(range(0, len(items), chunk_size), desc="Embed + upsert"):
        chunk = items[start:start + chunk_size]
        texts = [c[2] for c in chunk]
        embeddings = embedder.embed_batch(texts, batch_size=embed_batch_size)
        points = [
            _make_point(pid, meta, body, emb)
            for (pid, meta, body), emb in zip(chunk, embeddings)
        ]
        client.upsert(collection_name=COLLECTION_NAME, points=points)


def upsert_trail_data(trail_metadata: dict, body_content: str) -> None:
    point_id = trail_metadata.get("id")
    if not point_id:
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, trail_metadata.get("name", "")))
    upsert_batch([(point_id, trail_metadata, body_content)])
