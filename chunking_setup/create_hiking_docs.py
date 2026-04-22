import json
import os

# Load the data you exported from Overpass Turbo
with open('romania_hiking.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Create a directory for your text documents
os.makedirs('hiking_docs', exist_ok=True)

success_count = 0

for element in data['elements']:
    tags = element.get('tags', {})
    name = tags.get('name', f"Unnamed_{element['id']}")
    
    # We only care about things with names for our RAG system
    if 'name' in tags:
        # Building a rich text description for the SLM
        doc = [
            f"# {name}",
            f"Type: {tags.get('route', 'Point of Interest')}",
            f"Region: Romania",
            f"Difficulty (SAC Scale): {tags.get('sac_scale', 'Information not available')}",
            f"Trail Marking (Symbol): {tags.get('osmc:symbol', 'Check local signage')}",
            f"Elevation: {tags.get('ele', 'Unknown')} meters",
            f"Network: {tags.get('network', 'Local network')}",
            f"Source: OpenStreetMap Data 2026"
        ]
        
        # Adding 'vibe' for the Fine-tuning part
        if 'description' in tags:
            doc.append(f"Description: {tags['description']}")
            
        # FIX: Sanitize the name, truncate it, and add the unique element ID
        safe_name = "".join([c if c.isalnum() else "_" for c in name])
        short_name = safe_name[:50] # Keep only the first 50 characters
        filename = f"{short_name}_{element['id']}" # Append ID to guarantee uniqueness
        
        # Save as Markdown (.md)
        with open(f"hiking_docs/{filename}.md", "w", encoding='utf-8') as out_f:
            out_f.write("\n".join(doc))
            
        success_count += 1

print(f"Successfully created {success_count} hiking documents!")