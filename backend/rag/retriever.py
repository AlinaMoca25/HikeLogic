# Step 0 verification (assumptions baked into this module):
#   - Collection name: 'hike_logic_romania' (set via setup_qdrant.py -> create_collection).
#   - Named vectors: 'dense' (BGE-M3, 1024-d, cosine) and 'sparse' (BGE-M3 lexical_weights).
#     The pre-existing setup used a single unnamed 384-d MiniLM dense vector with no sparse;
#     this module assumes setup_qdrant.py + ingest_all.py have been re-run against the
#     updated rag/qdrant_client.py + rag/embedder.py (BGE-M3 named dense+sparse).
#   - Sparse vectors come from BGE-M3's built-in lexical_weights, NOT a separate BM25 encoder.
#   - Payload keys written by the ingest: text, name, difficulty, region, marking.
#   - Point id: integer OSM id (extracted from frontmatter 'osm_<id>' in ingest_all.py).

from qdrant_client.models import Fusion, FusionQuery, Prefetch, SparseVector

from .config import (
    COLLECTION_NAME,
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    TOP_K_RETRIEVE,
)
from .embeddings import BGEM3Embedder
from .qdrant_client import get_client


def hybrid_search(query: str, limit: int = TOP_K_RETRIEVE):
    client = get_client()
    embedder = BGEM3Embedder.get_instance()
    embedded = embedder.embed_query(query)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            Prefetch(
                query=embedded["dense"],
                using=DENSE_VECTOR_NAME,
                limit=limit,
            ),
            Prefetch(
                query=SparseVector(
                    indices=embedded["sparse"]["indices"],
                    values=embedded["sparse"]["values"],
                ),
                using=SPARSE_VECTOR_NAME,
                limit=limit,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=limit,
        with_payload=True,
    )
    return results.points
