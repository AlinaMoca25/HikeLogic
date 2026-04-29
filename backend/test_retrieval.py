"""Sanity check for the retriever. Run from project root: python -m backend.test_retrieval"""

from backend.rag.search import search

QUERIES = [
    # Vague semantic — dense should win
    "easy walk near a lake",
    "family-friendly trail with a waterfall",
    "short hike for beginners",
    # Specific named entities — sparse should win
    "Cabana Bâlea",
    "Retezat trails",
    "Vârful Omu",
    # Mixed — fusion matters
    "difficult trail in Făgăraș",
    "marked routes near Sinaia",
]


def main() -> None:
    for q in QUERIES:
        print("=" * 80)
        print(f"QUERY: {q}")
        print("-" * 80)
        results = search(q)
        if not results:
            print("  (no results)")
            print()
            continue
        for i, hit in enumerate(results, 1):
            name = hit.metadata.get("name") or "?"
            region = hit.metadata.get("region") or "?"
            difficulty = hit.metadata.get("difficulty") or "?"
            preview = (hit.text or "").strip().replace("\n", " ")[:200]
            print(f"  {i}. [{hit.score:.4f}] {name}  (region: {region}, difficulty: {difficulty})")
            print(f"     {preview}")
        print()


if __name__ == "__main__":
    main()
