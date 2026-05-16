"""Run RAGAS on backend/eval/ragas_queries.json. Needs OPENAI_API_KEY for the judge."""
from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, faithfulness

from .rag.pipeline import answer

EVAL_PATH = Path(__file__).resolve().parent / "eval" / "ragas_queries.json"


def main() -> None:
    queries = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    rows = []
    for q in queries:
        result = answer(q["query"])
        rows.append({
            "question": q["query"],
            "answer": result.text,
            "contexts": [h.text for h in result.sources] or [""],
            "ground_truth": q.get("expected", ""),
        })

    ds = Dataset.from_list(rows)
    scores = evaluate(ds, metrics=[faithfulness, answer_relevancy, context_precision])
    print(scores)


if __name__ == "__main__":
    main()
