from dataclasses import dataclass

from .config import MIN_RERANK_SCORE_FOR_GENERATION
from .generator import Generator
from .prompt import SYSTEM_PROMPT, build_user_message
from .search import Hit, search


@dataclass
class Answer:
    query: str
    text: str
    sources: list[Hit]


def _abstention(query: str, sources: list[Hit] | None = None) -> Answer:
    return Answer(
        query=query,
        text=(
            "Nu am găsit surse relevante în baza de trasee pentru această "
            "întrebare, deci nu pot răspunde în siguranță."
        ),
        sources=sources or [],
    )


def _has_relevant_context(hits: list[Hit]) -> bool:
    if not hits:
        return False

    return max(hit.score for hit in hits) >= MIN_RERANK_SCORE_FOR_GENERATION


def answer(query: str) -> Answer:
    hits = search(query)
    if not _has_relevant_context(hits):
        return _abstention(query, hits)

    user_message = build_user_message(query, hits)
    generated = Generator.get_instance().generate(SYSTEM_PROMPT, user_message)
    return Answer(query=query, text=generated, sources=hits)
