"""Sanity check for the full RAG pipeline. Run from project root: python -m backend.test_pipeline"""

from backend.rag.pipeline import answer

QUERIES = [
    "Cum ajung la Cabana Bâlea?",
    "Care este cel mai dificil traseu din Făgăraș?",
    "Recomandă-mi un traseu ușor lângă un lac.",
    "What's a family-friendly trail with a waterfall?",
    "Vârful Omu — de unde pornesc traseele?",
]


def main() -> None:
    for q in QUERIES:
        print("=" * 80)
        print(f"QUERY: {q}")
        print("-" * 80)
        result = answer(q)
        print(result.text)
        print()
        print("Sources:")
        for i, h in enumerate(result.sources, 1):
            name = (h.metadata or {}).get("name") or "?"
            print(f"  [{i}] {name}  (score: {h.score:.4f})")
        print()


if __name__ == "__main__":
    main()
