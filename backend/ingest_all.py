import os
import uuid
import frontmatter
from rag.embedder import upsert_batch

DOCS_PATH = "../chunking_setup/hiking_docs"


def run_ingestion():
    if not os.path.exists(DOCS_PATH):
        print(f"❌ Path not found: {DOCS_PATH}")
        return

    files = [f for f in os.listdir(DOCS_PATH) if f.endswith(".md")]
    print(f"🚀 Found {len(files)} hiking documents. Loading into memory...")

    items: list[tuple[str, dict, str]] = []
    parse_errors = 0
    for filename in files:
        file_path = os.path.join(DOCS_PATH, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                post = frontmatter.load(f)
        except Exception as e:
            parse_errors += 1
            print(f"⚠️  Skipping {filename}: {e}")
            continue

        osm_id = post.get("id") or filename
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(osm_id)))

        metadata = {
            "id": point_id,
            "name": post.get("name", filename.replace(".md", "")),
            "type": post.get("type", "trail"),
            "difficulty": post.get("difficulty", "unknown"),
            "region": post.get("region", "unknown"),
            "mountain_range": post.get("mountain_range"),
            "marking": post.get("marking", "none"),
            "lat": post.get("lat"),
            "lon": post.get("lon"),
            "osm_url": post.get("osm_url"),
        }
        items.append((point_id, metadata, post.content))

    if parse_errors:
        print(f"⚠️  {parse_errors} files skipped due to YAML parse errors")

    print(f"📦 Loaded {len(items)} docs. Batched embed + upsert starting...")
    upsert_batch(items)
    print(f"✨ Done. Ingested {len(items)} docs into Qdrant.")


if __name__ == "__main__":
    run_ingestion()
