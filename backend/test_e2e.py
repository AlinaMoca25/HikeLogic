"""Quick end-to-end smoke test of the full pipeline.

Run: python -m backend.test_e2e
"""
from .rag.agent import answer_with_agent
from .rag.pipeline import answer

QUERIES = [
    ("rag",   "Ce marcaj are traseul către Cabana Bâlea Lac?"),
    ("rag",   "Există surse de apă lângă Vârful Omu?"),
    ("rag",   "Care este cel mai bun restaurant din Sibiu?"),
    ("agent", "Care este prognoza meteo pentru Bucegi?"),
]


def main() -> None:
    for mode, q in QUERIES:
        result = answer_with_agent(q) if mode == "agent" else answer(q)
        print(f"[{mode}] Q: {q}")
        print(f"        A: {result.text}\n")


if __name__ == "__main__":
    main()
