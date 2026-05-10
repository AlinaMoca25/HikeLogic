import json
import os
import math


def yaml_value(value):
    if value in (None, ""):
        return "unknown"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value), ensure_ascii=False)


def parse_marking(marking):
    if not marking or marking == "unknown":
        return "unknown", "unknown"

    parts = marking.split(":")
    color = parts[0] if parts else "unknown"
    shape = parts[-1] if len(parts) >= 3 and parts[-1] else "unknown"
    return color, shape


def safe_filename(value):
    safe_name = "".join([c if c.isalnum() else "_" for c in value])
    return safe_name[:80]


def poi_kind(tags):
    if tags.get('natural') == 'spring':
        return "spring", "Izvor natural (Apă)"
    if tags.get('amenity') == 'drinking_water':
        return "drinking_water", "Sursă de apă potabilă amenajată"
    if tags.get('natural') == 'peak':
        return "peak", f"Vârf montan ({tags.get('ele', '?')}m)"
    if tags.get('emergency') == 'mountain_rescue':
        return "mountain_rescue", "Bază / Refugiu Salvamont"
    if tags.get('tourism') == 'alpine_hut':
        return "alpine_hut", "Cabană montană"
    if tags.get('amenity') == 'parking':
        return "parking", "Parcare auto"
    if tags.get('highway') == 'via_ferrata':
        return "via_ferrata", "Traseu expus / Via Ferrata"
    return None, None


def should_index_poi(tags):
    kind, _ = poi_kind(tags)
    if not kind:
        return False
    if kind in {"spring", "drinking_water", "mountain_rescue", "alpine_hut", "via_ferrata"}:
        return True
    return bool(tags.get('name'))


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

for filename in os.listdir('hiking_docs'):
    if filename.endswith('.md'):
        os.remove(os.path.join('hiking_docs', filename))

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
poi_success_count = 0

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
    t_mark_color, t_mark_shape = parse_marking(t_mark)
    t_asc = tags.get('ascent', 'unknown') # OSM folosește tag-ul 'ascent' pentru elevation gain
    t_duration = tags.get('duration', 'unknown')
    t_from = tags.get('from', 'unknown')
    t_to = tags.get('to', 'unknown')
    
    # Formatul strict cerut de Vector DB
    yaml_header = f"""---
id: osm_{t_id}
osm_id: {t_id}
type: trail
name: {yaml_value(name)}
region: Romania
difficulty: {yaml_value(t_diff)}
marking: {yaml_value(t_mark)}
marking_color: {yaml_value(t_mark_color)}
marking_shape: {yaml_value(t_mark_shape)}
duration: {yaml_value(t_duration)}
elevation_gain: {yaml_value(t_asc)}
from: {yaml_value(t_from)}
to: {yaml_value(t_to)}
lat: {yaml_value(trail_lat)}
lon: {yaml_value(trail_lon)}
source: openstreetmap
osm_url: {yaml_value(f"https://www.openstreetmap.org/{trail['type']}/{t_id}")}
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
    doc.append(f"Timp estimat: {t_duration if t_duration != 'unknown' else 'Nespecificat'}")

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
                _, poi_type = poi_kind(poi_tags)
                poi_type = poi_type or "Punct de interes"
                
                if poi_name != 'Punct nenumit' or 'Apă' in poi_type or 'Salvamont' in poi_type:
                    nearby_pois.append(f"- **{poi_type}**: {poi_name} (la aprox. {dist:.1f} km distanță)")

    if nearby_pois:
        doc.append("\n## Puncte de Interes și Logistică în apropiere:")
        doc.extend(list(set(nearby_pois))[:20])

    # ==========================================
    # 4. SALVAREA FIȘIERULUI (WINDOWS SAFE)
    # ==========================================
    filename = f"{safe_filename(name)[:50]}_{t_id}"
    
    with open(f"hiking_docs/{filename}.md", "w", encoding='utf-8') as out_f:
        out_f.write("\n".join(doc))
        
    success_count += 1

indexed_pois = set()

for poi in pois:
    tags = poi.get('tags', {})
    if not should_index_poi(tags):
        continue

    poi_id = poi['id']
    poi_osm_type = poi['type']
    unique_id = f"{poi_osm_type}_{poi_id}"
    if unique_id in indexed_pois:
        continue

    poi_lat, poi_lon = get_coords(poi)
    poi_type, poi_label = poi_kind(tags)
    poi_name = tags.get('name') or f"{poi_label} {poi_id}"
    poi_ele = tags.get('ele', 'unknown')

    yaml_header = f"""---
id: osm_{unique_id}
osm_id: {poi_id}
osm_type: {yaml_value(poi_osm_type)}
type: poi
poi_type: {yaml_value(poi_type)}
name: {yaml_value(poi_name)}
region: Romania
lat: {yaml_value(poi_lat)}
lon: {yaml_value(poi_lon)}
ele: {yaml_value(poi_ele)}
source: openstreetmap
osm_url: {yaml_value(f"https://www.openstreetmap.org/{poi_osm_type}/{poi_id}")}
---
"""

    doc = [yaml_header]
    doc.append(f"# {poi_name}")
    doc.append(f"Tip: {poi_label}")

    if poi_ele != 'unknown':
        doc.append(f"Altitudine: {poi_ele} m")

    if 'description' in tags:
        doc.append(f"\n## Descriere:\n{tags['description']}")

    filename = f"POI_{safe_filename(poi_type)}_{safe_filename(poi_name)}_{poi_id}"

    with open(f"hiking_docs/{filename}.md", "w", encoding='utf-8') as out_f:
        out_f.write("\n".join(doc))

    indexed_pois.add(unique_id)
    poi_success_count += 1

print(
    f"Gata! Au fost generate {success_count} trasee și "
    f"{poi_success_count} puncte de interes."
)
