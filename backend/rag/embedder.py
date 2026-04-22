import uuid
from rag.qdrant_client import get_client
from qdrant_client.http import models

from sentence_transformers import SentenceTransformer 

model = SentenceTransformer('all-MiniLM-L6-v2') 

def upsert_trail_data(trail_metadata: dict, body_content: str):
    client = get_client()
    vector = model.encode(body_content).tolist()

    # FIX: Use the OSM ID from your metadata or a hash of the name
    # This ensures that if the 'osm_id' is the same, Qdrant overwrites it instead of duplicating.
    point_id = trail_metadata.get("id") 
    
    # If the ID is a string like "osm_123", we should turn it into a UUID format
    # or just use a stable hash of the filename.
    if not point_id:
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, trail_metadata.get("name")))

    point = models.PointStruct(
        id=point_id, # Now it's stable!
        vector=vector,
        payload={
            "text": body_content,
            "name": trail_metadata.get("name"),
            "difficulty": trail_metadata.get("difficulty"),
            "region": trail_metadata.get("region"),
            "marking": trail_metadata.get("marking")
        }
    )

    client.upsert(collection_name="hike_logic_romania", points=[point])
    print(f"✅ Successfully uploaded: {trail_metadata.get('name')}")

# Example of how you'd call this:
# metadata = {"name": "Vf. Omu", "difficulty": "Hard", "region": "Bucegi"}
# content = "This trail starts from Busteni and is very steep..."
# upsert_trail_data(metadata, content)
