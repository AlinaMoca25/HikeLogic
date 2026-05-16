import re
from dataclasses import dataclass

from .generator import Generator
from .prompt import SYSTEM_PROMPT, build_user_message
from .search import Hit, search


_ENTITY_RE = re.compile(
    r"[A-ZÀ-ÝĂÂÎȘȚ][a-zà-ÿăâîșț]+(?:\s+[A-ZÀ-ÝĂÂÎȘȚ][a-zà-ÿăâîșț]+)+"
)
_TOKEN_RE = re.compile(r"[\wăâîșțĂÂÎȘȚ]+", re.UNICODE)

# Generic words that match many chunk names; discounted so short proper nouns
# like "Omu" still dominate the score under the 3-char token threshold.
_COMMON_MODIFIERS = {
    "cabana", "cabane", "vârful", "vârf", "vf", "refugiul", "refugiu",
    "lacul", "lac", "muntele", "valea", "vale", "cascada",
    "șaua", "saua", "sa", "curmătura",
    "izvorul", "izvor", "fântâna", "peștera", "stația",
}


def _grounding_score(entity: str, hit: Hit) -> float:
    tokens = [t for t in _TOKEN_RE.findall(entity.lower()) if len(t) >= 3]
    if not tokens:
        return 0.0
    name = (hit.metadata.get("name") or "").lower()
    text = (hit.text or "").lower()
    score = 0.0
    for t in tokens:
        weight_name = 2.0 if t not in _COMMON_MODIFIERS else 0.3
        weight_text = 1.0 if t not in _COMMON_MODIFIERS else 0.1
        if t in name:
            score += weight_name
        if t in text:
            score += weight_text
    return score


def check_entity_grounding(query: str, hits: list[Hit], *, top_n: int = 5, threshold: float = 1.5) -> str | None:
    """Abstain if any multi-word capitalized phrase in `query` is missing from top-N hits."""
    entities = _ENTITY_RE.findall(query)
    if not entities:
        return None
    top_hits = hits[:top_n]
    missing = [
        e for e in entities
        if max((_grounding_score(e, h) for h in top_hits), default=0) < threshold
    ]
    if not missing:
        return None
    names = " și ".join(missing)
    return (
        f"Nu am găsit informații specifice despre {names} în baza de trasee. "
        f"Pentru a evita afirmații nesigure, nu pot răspunde la această întrebare."
    )


@dataclass
class Answer:
    query: str
    text: str
    sources: list[Hit]


_SCORE_FLOOR = 0.05
_OUT_OF_DOMAIN_ABSTENTION = (
    "Nu am găsit surse relevante în baza de trasee pentru această întrebare, "
    "deci nu pot răspunde în siguranță."
)


def answer(query: str) -> Answer:
    hits = search(query)
    # Score-floor abstention: catches "matches by coincidence" cases where every
    # hit scores essentially zero but contains a query token (e.g. a hotel doc
    # surfacing on a "restaurant in Sibiu" query).
    if hits and max(h.score for h in hits) < _SCORE_FLOOR:
        return Answer(query=query, text=_OUT_OF_DOMAIN_ABSTENTION, sources=hits)
    abstention = check_entity_grounding(query, hits)
    if abstention is not None:
        return Answer(query=query, text=abstention, sources=hits)
    user_message = build_user_message(query, hits)
    generated = Generator.get_instance().generate(SYSTEM_PROMPT, user_message)
    return Answer(query=query, text=generated, sources=hits)


def ask(query: str) -> str:
    return answer(query).text
