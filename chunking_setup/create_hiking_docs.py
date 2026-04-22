import json
import os
import math

# Funcția Haversine pentru calculul distanțelor
def haversine(lat1, lon1, lat2, lon2):
    if None in (lat1, lon1, lat2, lon2): return float('inf')
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))

def get_coords(element):
    if element['type'] == 'node':
        return element.get('lat'), element.get('lon')
    elif 'center' in element:
        return element['center'].get('lat'), element['center'].get('lon')
    return None, None

with open('romania_hiking.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

os.makedirs('hiking_docs', exist_ok=True)

trails = []
pois = []

# Separăm datele
for el in data['elements']:
    tags = el.get('tags', {})
    if 'route' in tags and tags['route'] in ['hiking', 'foot', 'walking']:
        trails.append(el)
    else:
        pois.append(el)

success_count = 0

for trail in trails:
    tags = trail.get('tags', {})
    name = tags.get('name')
    if not name:
        continue

    trail_lat, trail_lon = get_coords(trail)

    # ==========================================
    # 1. CONSTRUIREA ANTETULUI YAML (FRONTMATTER)
    # ==========================================
    t_id = trail['id']
    t_diff = tags.get('sac_scale', 'unknown')
    t_mark = tags.get('osmc:symbol', 'unknown')
    t_asc = tags.get('ascent', 'unknown') # OSM folosește tag-ul 'ascent' pentru elevation gain
    
    # Formatul strict cerut de Vector DB
    yaml_header = f"""---
id: osm_{t_id}
type: trail
region: Romania
difficulty: {t_diff}
marking: {t_mark}
elevation_gain: {t_asc}
---
"""

    # ==========================================
    # 2. CONSTRUIREA CONȚINUTULUI MARKDOWN
    # ==========================================
    doc = [yaml_header]
    doc.append(f"# {name}")
    doc.append(f"Tip: Traseu de drumeție")
    doc.append(f"Dificultate (SAC Scale): {t_diff}")
    doc.append(f"Marcaj: {t_mark}")
    doc.append(f"Timp estimat: {tags.get('duration', 'Nespecificat')}")

    if 'description' in tags:
        doc.append(f"\n## Descriere:\n{tags['description']}")

    # ==========================================
    # 3. CALCULUL PUNCTELOR DE INTERES (POI)
    # ==========================================
    nearby_pois = []
    if trail_lat and trail_lon:
        for poi in pois:
            poi_tags = poi.get('tags', {})
            poi_lat, poi_lon = get_coords(poi)
            dist = haversine(trail_lat, trail_lon, poi_lat, poi_lon)
            
            if dist <= 5.0:
                poi_name = poi_tags.get('name', 'Punct nenumit')
                poi_type = "Punct de interes"
                
                if poi_tags.get('natural') == 'spring': poi_type = "Izvor natural (Apă)"
                elif poi_tags.get('amenity') == 'drinking_water': poi_type = "Sursă de apă potabilă amenajată"
                elif poi_tags.get('natural') == 'peak': poi_type = f"Vârf montan ({poi_tags.get('ele', '?')}m)"
                elif poi_tags.get('emergency') == 'mountain_rescue': poi_type = "Bază / Refugiu Salvamont"
                elif poi_tags.get('tourism') == 'alpine_hut': poi_type = "Cabană montană"
                elif poi_tags.get('amenity') == 'parking': poi_type = "Parcare auto"
                elif poi_tags.get('highway') == 'via_ferrata': poi_type = "Traseu expus / Via Ferrata"
                
                if poi_name != 'Punct nenumit' or 'Apă' in poi_type or 'Salvamont' in poi_type:
                    nearby_pois.append(f"- **{poi_type}**: {poi_name} (la aprox. {dist:.1f} km distanță)")

    if nearby_pois:
        doc.append("\n## Puncte de Interes și Logistică în apropiere:")
        doc.extend(list(set(nearby_pois))[:20])

    # ==========================================
    # 4. SALVAREA FIȘIERULUI (WINDOWS SAFE)
    # ==========================================
    safe_name = "".join([c if c.isalnum() else "_" for c in name])
    short_name = safe_name[:50]
    filename = f"{short_name}_{t_id}"
    
    with open(f"hiking_docs/{filename}.md", "w", encoding='utf-8') as out_f:
        out_f.write("\n".join(doc))
        
    success_count += 1

print(f"Gata! Au fost generate {success_count} documente .md cu metadata YAML complet funcțională.")