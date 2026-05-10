"""Evaluate retrieval against a small gold set.

Run after Qdrant ingestion:
    python -m backend.evaluate_retrieval
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.rag.retriever import hybrid_search
from backend.rag.search import Hit, search

DEFAULT_GOLD_PATH = Path(__file__).resolve().parent / "eval" / "retrieval_gold.json"


@dataclass
class CaseResult:
    case_id: str
    query: str
    passed: bool
    rank: int | None
    top_name: str | None
    top_entity_type: str | None

    @property
    def reciprocal_rank(self) -> float:
        return 0.0 if self.rank is None else 1.0 / self.rank


def _norm(value: Any) -> str:
    return str(value or "").casefold()


def _hit_matches(hit: Hit, expected: dict) -> bool:
    metadata = hit.metadata or {}
    name = _norm(metadata.get("name"))

    name_contains = expected.get("name_contains")
    if name_contains and _norm(name_contains) not in name:
        return False

    for key in ("entity_type", "poi_type", "region", "difficulty"):
        expected_value = expected.get(key)
        if expected_value is not None and _norm(metadata.get(key)) != _norm(expected_value):
            return False

    return True


def _point_to_hit(point) -> Hit:
    payload = dict(point.payload or {})
    text = payload.pop("text", "")
    return Hit(text=text, score=point.score, metadata=payload)


def _first_match_rank(hits: list[Hit], expected_items: list[dict]) -> int | None:
    for rank, hit in enumerate(hits, 1):
        if any(_hit_matches(hit, expected) for expected in expected_items):
            return rank
    return None


def retrieve(query: str, mode: str, top_k: int) -> list[Hit]:
    if mode == "rerank":
        return search(query)[:top_k]

    if mode == "hybrid":
        return [_point_to_hit(point) for point in hybrid_search(query, limit=top_k)]

    raise ValueError(f"Unknown retrieval mode: {mode}")


def evaluate_case(case: dict, mode: str, top_k: int) -> CaseResult:
    hits = retrieve(case["query"], mode=mode, top_k=top_k)
    rank = _first_match_rank(hits, case["expected"])
    top = hits[0] if hits else None
    top_metadata = top.metadata if top else {}

    return CaseResult(
        case_id=case["id"],
        query=case["query"],
        passed=rank is not None,
        rank=rank,
        top_name=top_metadata.get("name") if top else None,
        top_entity_type=top_metadata.get("entity_type") if top else None,
    )


def load_cases(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        cases = json.load(f)

    if not isinstance(cases, list) or not cases:
        raise ValueError(f"Gold file must contain a non-empty list: {path}")

    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD_PATH)
    parser.add_argument("--mode", choices=["hybrid", "rerank"], default="hybrid")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    cases = load_cases(args.gold)
    results = [evaluate_case(case, mode=args.mode, top_k=args.top_k) for case in cases]

    passed = sum(result.passed for result in results)
    mrr = sum(result.reciprocal_rank for result in results) / len(results)

    print(f"Cases: {len(results)}")
    print(f"Mode: {args.mode}")
    print(f"Recall@{args.top_k}: {passed / len(results):.3f}")
    print(f"MRR@{args.top_k}: {mrr:.3f}")
    print()

    for result in results:
        status = "PASS" if result.passed else "FAIL"
        rank = result.rank if result.rank is not None else "-"
        print(
            f"{status} {result.case_id} rank={rank} "
            f"query={result.query!r} top={result.top_name!r} "
            f"top_type={result.top_entity_type!r}"
        )

    if passed != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
