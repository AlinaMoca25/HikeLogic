import unicodedata

from qdrant_client.models import (
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    Prefetch,
    SparseVector,
)

from .config import (
    COLLECTION_NAME,
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    TOP_K_RETRIEVE,
)
from .embeddings import BGEM3Embedder
from .qdrant_client import get_client


_MOUNTAIN_RANGES = [
    "Bucegi", "Făgăraș", "Piatra Craiului", "Iezer-Păpușa", "Ciucaș",
    "Postăvarul", "Piatra Mare", "Retezat", "Țarcu-Godeanu", "Parâng",
    "Șureanu", "Cindrel", "Lotrului", "Vâlcan", "Mehedinți", "Apuseni",
    "Trascău", "Rodna", "Maramureș", "Suhard", "Călimani", "Bistriței",
    "Ceahlău", "Hășmaș", "Harghita",
]


def _strip_diacritics(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


_RANGE_LOOKUP = {_strip_diacritics(r).lower(): r for r in _MOUNTAIN_RANGES}


def detect_range(query: str) -> str | None:
    folded = _strip_diacritics(query).lower()
    for key, canonical in _RANGE_LOOKUP.items():
        if key in folded:
            return canonical
    return None


def _query_points(client, embedded, limit, query_filter):
    return client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            Prefetch(
                query=embedded["dense"],
                using=DENSE_VECTOR_NAME,
                limit=limit,
                filter=query_filter,
            ),
            Prefetch(
                query=SparseVector(
                    indices=embedded["sparse"]["indices"],
                    values=embedded["sparse"]["values"],
                ),
                using=SPARSE_VECTOR_NAME,
                limit=limit,
                filter=query_filter,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=limit,
        with_payload=True,
    ).points


def hybrid_search(query: str, limit: int = TOP_K_RETRIEVE):
    """Hybrid retrieve. When the query mentions a recognized mountain range,
    filter chunks by `mountain_range` to disambiguate (e.g. Vf. Omu Bucegi vs
    Suhard). Falls back to unfiltered search if the filter returns nothing or
    the payload index doesn't exist yet."""
    client = get_client()
    embedder = BGEM3Embedder.get_instance()
    embedded = embedder.embed_query(query)

    detected = detect_range(query)
    if detected:
        query_filter = Filter(must=[
            FieldCondition(key="mountain_range", match=MatchValue(value=detected))
        ])
        try:
            points = _query_points(client, embedded, limit, query_filter)
            if points:
                return points
        except Exception:
            # Likely the payload index for mountain_range hasn't been created
            # (older collection). Fall through to unfiltered search.
            pass
    return _query_points(client, embedded, limit, None)
