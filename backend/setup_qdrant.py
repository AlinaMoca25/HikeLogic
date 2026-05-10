import os

from rag.qdrant_client import create_collection, create_payload_indexes

create_collection(recreate=os.getenv("RECREATE_QDRANT_COLLECTION") == "1")
create_payload_indexes()
