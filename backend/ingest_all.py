import os
import frontmatter
from rag.embedder import upsert_trail_data

DOCS_PATH = "../chunking_setup/hiking_docs"

def run_ingestion():
    if not os.path.exists(DOCS_PATH):
        print(f"❌ Path not found: {DOCS_PATH}")
        return

    files = [f for f in os.listdir(DOCS_PATH) if f.endswith(".md")]
    print(f"🚀 Found {len(files)} hiking documents. Starting ingestion...")

    for filename in files:
        file_path = os.path.join(DOCS_PATH, filename)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            # Parse the YAML header and the text body separately
            post = frontmatter.load(f)
            
            # The metadata (from your YAML)
            metadata = {
                "id": int(post.get("id").replace("osm_", "")),
                "name": post.get("name", filename.replace(".md", "")),
                "difficulty": post.get("difficulty", "unknown"),
                "region": post.get("region", "unknown"),
                "marking": post.get("marking", "none")
            }
            
            # The actual text content
            content = post.content

            # Send it to your embedder
            try:
                upsert_trail_data(metadata, content)
            except Exception as e:
                print(f"⚠️ Failed to upload {filename}: {e}")

    print("✨ All trails have been indexed in Qdrant!")

if __name__ == "__main__":
    run_ingestion()

