import json
import os
import math
import yaml


def make_yaml_header(d: dict) -> str:
    # safe_dump handles names with embedded quotes that would break a hand-rolled f-string
    body = yaml.safe_dump(d, allow_unicode=True, sort_keys=False, default_flow_style=False)
    return f"---\n{body}---\n"


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


# Romanian mountain ranges as approximate (lat_min, lat_max, lon_min, lon_max).
# Used to tag every chunk with a mountain_range so retrieval can disambiguate
# (e.g. "Vf. Omu din Bucegi" vs "Vf. Omu din Suhard").
MOUNTAIN_RANGES = [
    ("Bucegi",          45.30, 45.50, 25.30, 25.65),
    ("Făgăraș",         45.55, 45.72, 24.30, 25.10),
    ("Piatra Craiului", 45.50, 45.65, 25.18, 25.32),
    ("Iezer-Păpușa",    45.40, 45.55, 24.95, 25.25),
    ("Ciucaș",          45.45, 45.55, 25.85, 26.05),
    ("Postăvarul",      45.55, 45.65, 25.55, 25.70),
    ("Piatra Mare",     45.55, 45.65, 25.60, 25.75),
    ("Retezat",         45.30, 45.45, 22.70, 23.05),
    ("Țarcu-Godeanu",   45.20, 45.50, 22.35, 22.85),
    ("Parâng",          45.30, 45.45, 23.40, 23.75),
    ("Șureanu",         45.50, 45.70, 23.30, 23.70),
    ("Cindrel",         45.55, 45.70, 23.70, 24.00),
    ("Lotrului",        45.45, 45.60, 23.65, 23.95),
    ("Vâlcan",          45.00, 45.25, 22.85, 23.30),
    ("Mehedinți",       44.85, 45.15, 22.40, 22.70),
    ("Apuseni",         46.20, 46.85, 22.50, 23.60),
    ("Trascău",         46.20, 46.50, 23.40, 23.75),
    ("Rodna",           47.40, 47.70, 24.55, 25.15),
    ("Maramureș",       47.60, 47.95, 24.40, 25.10),
    ("Suhard",          47.40, 47.60, 25.30, 25.55),
    ("Călimani",        47.00, 47.30, 24.85, 25.40),
    ("Bistriței",       46.85, 47.15, 25.55, 25.95),
    ("Ceahlău",         46.92, 47.02, 25.85, 26.05),
    ("Hășmaș",          46.65, 46.85, 25.70, 26.00),
    ("Harghita",        46.30, 46.70, 25.50, 25.95),
]


def mountain_range_for(lat, lon):
    if lat is None or lon is None:
        return None
    for name, la1, la2, lo1, lo2 in MOUNTAIN_RANGES:
        if la1 <= lat <= la2 and lo1 <= lon <= lo2:
            return name
    return None

# Only listed (tag, value) pairs become standalone POI chunks; unlisted POIs
# still appear as bulleted lines inside nearby trail docs.
POI_TYPES = [
    (('tourism', 'alpine_hut'), ('cabin', 'cabană montană')),
    (('tourism', 'guest_house'), ('cabin', 'cabană / pensiune')),
    (('tourism', 'chalet'), ('cabin', 'cabană / chalet')),
    (('tourism', 'hotel'), ('cabin', 'cabană / hotel montan')),
    (('tourism', 'motel'), ('cabin', 'cabană / motel')),
    (('tourism', 'hostel'), ('cabin', 'cabană / hostel')),
    (('tourism', 'wilderness_hut'), ('shelter', 'refugiu de munte')),
    (('amenity', 'shelter'), ('shelter', 'refugiu / adăpost montan')),
    (('natural', 'spring'), ('spring', 'izvor natural (sursă de apă)')),
    (('amenity', 'drinking_water'), ('water', 'sursă de apă potabilă amenajată')),
    (('natural', 'peak'), ('peak', 'vârf montan')),
    (('tourism', 'viewpoint'), ('viewpoint', 'punct de belvedere')),
    (('natural', 'cave_entrance'), ('cave', 'intrare în peșteră')),
    (('natural', 'saddle'), ('saddle', 'curmătură / șa montană')),
    (('emergency', 'mountain_rescue'), ('rescue', 'bază Salvamont / refugiu de salvare')),
    (('highway', 'via_ferrata'), ('via_ferrata', 'traseu de via ferrata')),
]

def classify_poi(tags):
    for (k, v), result in POI_TYPES:
        if tags.get(k) == v:
            return result
    return (None, None)

with open('romania_hiking.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

os.makedirs('hiking_docs', exist_ok=True)

trails = []
pois = []

for el in data['elements']:
    tags = el.get('tags', {})
    if 'route' in tags and tags['route'] in ['hiking', 'foot', 'walking']:
        trails.append(el)
    else:
        pois.append(el)

# One chunk per hiking route
trail_count = 0

for trail in trails:
    tags = trail.get('tags', {})
    name = tags.get('name')
    if not name:
        continue

    trail_lat, trail_lon = get_coords(trail)

    t_id = trail['id']
    t_type = trail.get('type', 'relation')
    t_diff = tags.get('sac_scale', 'unknown')
    t_mark = tags.get('osmc:symbol', 'unknown')
    t_asc = tags.get('ascent', 'unknown')
    t_url = f"https://www.openstreetmap.org/{t_type}/{t_id}"

    trail_range = mountain_range_for(trail_lat, trail_lon)
    trail_meta = {
        "id": f"osm_{t_type}_{t_id}",
        "type": "trail",
        "name": name,
        "region": "Romania",
        "difficulty": t_diff,
        "marking": t_mark,
        "elevation_gain": t_asc,
    }
    if trail_range:
        trail_meta["mountain_range"] = trail_range
    if trail_lat is not None:
        trail_meta["lat"] = trail_lat
    if trail_lon is not None:
        trail_meta["lon"] = trail_lon
    trail_meta["osm_url"] = t_url
    yaml_header = make_yaml_header(trail_meta)

    doc = [yaml_header]
    doc.append(f"# {name}")
    doc.append(f"Tip: Traseu de drumeție")
    if trail_range:
        doc.append(f"Masiv: {trail_range}")
    doc.append(f"Dificultate (SAC Scale): {t_diff}")
    doc.append(f"Marcaj: {t_mark}")
    doc.append(f"Timp estimat: {tags.get('duration', 'Nespecificat')}")

    if 'description' in tags:
        doc.append(f"\n## Descriere:\n{tags['description']}")

    nearby_pois = []
    if trail_lat and trail_lon:
        for poi in pois:
            poi_tags = poi.get('tags', {})
            poi_lat, poi_lon = get_coords(poi)
            dist = haversine(trail_lat, trail_lon, poi_lat, poi_lon)

            if dist <= 5.0:
                poi_name = poi_tags.get('name')
                if not poi_name:
                    continue  # skip unnamed POIs — they pollute answers ("Punct nenumit")
                poi_type = "Punct de interes"
                if poi_tags.get('natural') == 'spring': poi_type = "Izvor natural (Apă)"
                elif poi_tags.get('amenity') == 'drinking_water': poi_type = "Sursă de apă potabilă amenajată"
                elif poi_tags.get('natural') == 'peak': poi_type = f"Vârf montan ({poi_tags.get('ele', '?')}m)"
                elif poi_tags.get('emergency') == 'mountain_rescue': poi_type = "Bază / Refugiu Salvamont"
                elif poi_tags.get('tourism') == 'alpine_hut': poi_type = "Cabană montană"
                elif poi_tags.get('amenity') == 'parking': poi_type = "Parcare auto"
                elif poi_tags.get('highway') == 'via_ferrata': poi_type = "Traseu expus / Via Ferrata"
                nearby_pois.append(f"- **{poi_type}**: {poi_name} (la aprox. {dist:.1f} km distanță)")

    if nearby_pois:
        doc.append("\n## Puncte de Interes și Logistică în apropiere:")
        doc.extend(list(set(nearby_pois))[:20])

    safe_name = "".join([c if c.isalnum() else "_" for c in name])
    short_name = safe_name[:50]
    filename = f"{short_name}_{t_id}"

    with open(f"hiking_docs/{filename}.md", "w", encoding='utf-8') as out_f:
        out_f.write("\n".join(doc))

    trail_count += 1

# One chunk per named POI whose type is in POI_TYPES
poi_count = 0

for poi in pois:
    tags = poi.get('tags', {})
    name = tags.get('name')
    if not name:
        continue

    canonical_type, desc = classify_poi(tags)
    if canonical_type is None:
        continue

    lat, lon = get_coords(poi)
    if lat is None or lon is None:
        continue

    el_type = poi.get('type', 'node')
    p_id = poi['id']
    p_url = f"https://www.openstreetmap.org/{el_type}/{p_id}"
    poi_range = mountain_range_for(lat, lon)

    poi_meta = {
        "id": f"osm_{el_type}_{p_id}",
        "type": canonical_type,
        "name": name,
        "region": "Romania",
        "lat": lat,
        "lon": lon,
        "osm_url": p_url,
    }
    if poi_range:
        poi_meta["mountain_range"] = poi_range
    yaml_header = make_yaml_header(poi_meta)

    # Name repeated through the body so sparse-retrieval ranks this short POI
    # doc against much longer trail docs that mention the same name in passing.
    range_line = f"{name} se află în masivul {poi_range}." if poi_range else ""
    body = [
        yaml_header,
        f"# {name}",
        "",
        f"Acest document descrie locul numit **{name}**.",
        f"{name} este {desc}, în Romania.",
        range_line,
        "",
        f"Nume: {name}",
        f"Tip: {desc}",
        f"Coordonatele lui {name}: {lat}, {lon}",
        f"Regiune: Romania",
    ]
    if poi_range:
        body.append(f"Masiv: {poi_range}")

    if canonical_type == 'peak':
        ele = tags.get('ele')
        if ele:
            body.append(f"{name} are altitudinea de {ele} m.")
    if 'description' in tags:
        body.append(f"\n## Despre {name}\n{tags['description']}")
    if 'operator' in tags:
        body.append(f"Operatorul {name}: {tags['operator']}")
    if 'phone' in tags:
        body.append(f"Telefon ({name}): {tags['phone']}")
    if tags.get('drinking_water') == 'yes':
        body.append(f"La {name} există apă potabilă.")
    elif tags.get('drinking_water') == 'no':
        body.append(f"La {name} nu există apă potabilă amenajată.")

    # Section header says "NU descriu" so the model doesn't conflate the POI
    # with adjacent trails when both share name tokens (e.g. Bâlea Lac vs. trails).
    nearby_trails = []
    for trail in trails:
        t_name = trail.get('tags', {}).get('name')
        if not t_name:
            continue
        t_lat, t_lon = get_coords(trail)
        if t_lat is None or t_lon is None:
            continue
        d = haversine(lat, lon, t_lat, t_lon)
        if d <= 3.0:
            nearby_trails.append((d, t_name))

    if nearby_trails:
        nearby_trails.sort()
        body.append(f"\n## Trasee separate care trec pe lângă {name}")
        body.append(
            f"(Notă: traseele de mai jos NU descriu {name}. Sunt obiecte "
            f"geografice distincte, listate aici doar pentru orientare în zonă.)"
        )
        body.append("")
        for d, n in nearby_trails[:10]:
            body.append(f"- {n} (la aprox. {d:.1f} km de {name})")

    safe_name = "".join([c if c.isalnum() else "_" for c in name])
    short_name = safe_name[:50]
    filename = f"{short_name}_{el_type}_{p_id}"

    with open(f"hiking_docs/{filename}.md", "w", encoding='utf-8') as out_f:
        out_f.write("\n".join(body))

    poi_count += 1

print(f"Gata! Generate {trail_count} documente de trasee și {poi_count} documente de POI-uri.")
