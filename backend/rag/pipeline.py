from dataclasses import dataclass

from .generator import Generator
from .prompt import SYSTEM_PROMPT, build_user_message
from .search import Hit, search


@dataclass
class Answer:
    query: str
    text: str
    sources: list[Hit]


def answer(query: str) -> Answer:
    hits = search(query)
    user_message = build_user_message(query, hits)
    generated = Generator.get_instance().generate(SYSTEM_PROMPT, user_message)
    return Answer(query=query, text=generated, sources=hits)
