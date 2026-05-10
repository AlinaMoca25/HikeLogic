"""Evaluate grounded generation behavior.

Run after retrieval ingestion and generation credentials are configured:
    python -m backend.evaluate_generation
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.rag.pipeline import Answer, answer
from backend.rag.config import MIN_RERANK_SCORE_FOR_GENERATION
from backend.rag.search import search

DEFAULT_GOLD_PATH = Path(__file__).resolve().parent / "eval" / "generation_gold.json"
ABSTENTION_TERMS = (
    "nu am găsit",
    "nu pot răspunde",
    "nu conține",
    "does not contain",
    "cannot answer",
    "can't answer",
)


@dataclass
class CaseResult:
    case_id: str
    query: str
    passed: bool
    failures: list[str]
    answer_text: str


def _norm(value: Any) -> str:
    return str(value or "").casefold()


def _citation_ids(text: str) -> set[int]:
    ids: set[int] = set()
    for match in re.finditer(r"\[(\d+)\]", text):
        ids.add(int(match.group(1)))
    return ids


def _has_abstention(text: str) -> bool:
    normalized = _norm(text)
    return any(term in normalized for term in ABSTENTION_TERMS)


def dry_run_answer(query: str) -> Answer:
    return Answer(
        query=query,
        text="(dry run: generation was not called)",
        sources=search(query),
    )


def _dry_run_has_relevant_context(result: Answer) -> bool:
    if not result.sources:
        return False

    return max(source.score for source in result.sources) >= MIN_RERANK_SCORE_FOR_GENERATION


def evaluate_case(case: dict, dry_run: bool = False) -> CaseResult:
    result = dry_run_answer(case["query"]) if dry_run else answer(case["query"])
    text = result.text
    normalized = _norm(text)
    failures: list[str] = []

    if dry_run:
        if case.get("expect_citations") and not result.sources:
            failures.append("retrieval returned no sources for citation-required case")

        if case.get("expect_abstention") and _dry_run_has_relevant_context(result):
            failures.append("retrieval would pass relevance gate for abstention case")

        for term in case.get("required_terms", []):
            source_blob = "\n".join(
                f"{source.metadata.get('name', '')}\n{source.text}" for source in result.sources
            )
            if _norm(term) not in _norm(source_blob):
                failures.append(f"required term absent from retrieved context: {term}")

        return CaseResult(
            case_id=case["id"],
            query=case["query"],
            passed=not failures,
            failures=failures,
            answer_text=text,
        )

    if case.get("expect_citations"):
        citation_ids = _citation_ids(text)
        if not citation_ids:
            failures.append("missing source citations")

        valid_ids = set(range(1, len(result.sources) + 1))
        invalid_ids = citation_ids - valid_ids
        if invalid_ids:
            failures.append(f"invalid citation ids: {sorted(invalid_ids)}")

    if case.get("expect_abstention") and not _has_abstention(text):
        failures.append("missing abstention")

    for term in case.get("required_terms", []):
        if _norm(term) not in normalized:
            failures.append(f"missing required term: {term}")

    for term in case.get("forbidden_terms", []):
        if _norm(term) in normalized and not _has_abstention(text):
            failures.append(f"unsupported/unsafe term present: {term}")

    return CaseResult(
        case_id=case["id"],
        query=case["query"],
        passed=not failures,
        failures=failures,
        answer_text=text,
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
    parser.add_argument("--show-answers", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate retrieval/context prerequisites without calling the LLM.",
    )
    args = parser.parse_args()

    cases = load_cases(args.gold)
    results = [evaluate_case(case, dry_run=args.dry_run) for case in cases]
    passed = sum(result.passed for result in results)

    print(f"Cases: {len(results)}")
    if args.dry_run:
        print("Mode: dry-run retrieval/context validation")
    print(f"Pass rate: {passed / len(results):.3f}")
    print()

    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"{status} {result.case_id} query={result.query!r}")
        if result.failures:
            print(f"  failures: {', '.join(result.failures)}")
        if args.show_answers:
            print(f"  answer: {result.answer_text}")

    if passed != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
