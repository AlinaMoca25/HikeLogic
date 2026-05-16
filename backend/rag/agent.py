from __future__ import annotations

import re

from .generator import Generator
from .pipeline import (
    Answer,
    _OUT_OF_DOMAIN_ABSTENTION,
    _SCORE_FLOOR,
    check_entity_grounding,
)
from .prompt import SYSTEM_PROMPT, build_user_message
from .search import Hit, search
from .tools import ToolResult, compute_distance, detect_intent, fetch_weather

_ENTITY_RE = re.compile(
    r"[A-ZÀ-ÝĂÂÎȘȚ][a-zà-ÿăâîșț]+(?:\s+[A-ZÀ-ÝĂÂÎȘȚ][a-zà-ÿăâîșț]+)+"
)
_TOKEN_RE = re.compile(r"[\wăâîșțĂÂÎȘȚ]+", re.UNICODE)

# Entity prefix → expected chunk type. Used to prefer the right kind of chunk
# when a query names a specific entity (e.g. "Cabana X" should pick a cabin
# chunk, not a trail that mentions X).
_ENTITY_TYPE_HINTS = {
    "cabana": "cabin", "cabane": "cabin",
    "refugiul": "shelter", "refugiu": "shelter",
    "vârful": "peak", "vârf": "peak", "vf": "peak",
    "izvorul": "spring", "izvor": "spring",
    "fântâna": "water",
    "șaua": "saddle", "saua": "saddle", "curmătura": "saddle",
    "peștera": "cave", "pestera": "cave",
}


def _entity_type_hint(entity: str) -> str | None:
    first = entity.lower().split()[0] if entity else ""
    return _ENTITY_TYPE_HINTS.get(first)


def _append_tool(hits: list[Hit], tool: ToolResult) -> list[Hit]:
    extra = Hit(
        text=tool.text,
        score=1.0,
        metadata={"name": f"Tool: {tool.name}", "type": "tool"},
    )
    return list(hits) + [extra]


def _hit_coords(h: Hit) -> tuple[float, float] | None:
    lat = h.metadata.get("lat") or h.metadata.get("latitude")
    lon = h.metadata.get("lon") or h.metadata.get("longitude")
    if lat is None or lon is None:
        return None
    return float(lat), float(lon)


def _try_weather(query: str, hits: list[Hit]) -> ToolResult | None:
    # If the query names an entity, fetch weather for its coords specifically
    # so "weather at Vf. Negoiu" doesn't pull coords from the first random hit.
    entities = _ENTITY_RE.findall(query)
    target = None
    for e in entities:
        h = _find_hit_for_entity(e, hits)
        if h is not None:
            target = h
            break
    if target is None:
        for h in hits:
            if _hit_coords(h) is not None:
                target = h
                break
    if target is None:
        return None
    c = _hit_coords(target)
    try:
        return fetch_weather(c[0], c[1])
    except Exception:
        return None


def _score_hit_against_entity(h: Hit, entity: str) -> int:
    tokens = [t for t in _TOKEN_RE.findall(entity.lower()) if len(t) >= 4]
    if not tokens:
        return 0
    name = (h.metadata.get("name") or "").lower()
    text = (h.text or "").lower()
    return sum((2 if t in name else 0) + (1 if t in text else 0) for t in tokens)


def _find_hit_for_entity(entity: str, hits: list[Hit]) -> Hit | None:
    expected = _entity_type_hint(entity)
    best, best_score = None, 0
    for h in hits:
        if _hit_coords(h) is None:
            continue
        score = _score_hit_against_entity(h, entity)
        if expected and h.metadata.get("type") == expected:
            score *= 2  # boost matching type so the right kind of chunk wins
        if score > best_score:
            best, best_score = h, score
    return best if best_score > 0 else None


def _try_distance(query: str, hits: list[Hit]) -> ToolResult | None:
    entities = _ENTITY_RE.findall(query)
    if len(entities) >= 2:
        a = _find_hit_for_entity(entities[0], hits)
        b = _find_hit_for_entity(entities[1], hits)
        if a is not None and b is not None and a is not b:
            ca, cb = _hit_coords(a), _hit_coords(b)
            return compute_distance(
                ca[0], ca[1], cb[0], cb[1],
                name1=entities[0], name2=entities[1],
            )

    picked: list[tuple[float, float, str]] = []
    for h in hits:
        c = _hit_coords(h)
        if c is None:
            continue
        picked.append((c[0], c[1], h.metadata.get("name") or "?"))
        if len(picked) == 2:
            break
    if len(picked) < 2:
        return None
    a, b = picked[0], picked[1]
    return compute_distance(a[0], a[1], b[0], b[1], name1=a[2], name2=b[2])


def answer_with_agent(query: str) -> Answer:
    hits = search(query)
    if hits and max(h.score for h in hits) < _SCORE_FLOOR:
        return Answer(query=query, text=_OUT_OF_DOMAIN_ABSTENTION, sources=hits)
    intent = detect_intent(query)
    tool: ToolResult | None = None
    if intent == "weather":
        tool = _try_weather(query, hits)
    elif intent == "distance":
        tool = _try_distance(query, hits)
    if tool is not None:
        hits = _append_tool(hits, tool)
    abstention = check_entity_grounding(query, hits)
    if abstention is not None:
        return Answer(query=query, text=abstention, sources=hits)
    user_message = build_user_message(query, hits)
    text = Generator.get_instance().generate(SYSTEM_PROMPT, user_message)
    return Answer(query=query, text=text, sources=hits)


def ask_agent(query: str) -> str:
    return answer_with_agent(query).text
