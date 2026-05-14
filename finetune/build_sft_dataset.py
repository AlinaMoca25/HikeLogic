"""Build a grounded SFT dataset from local hiking documents.

The fine-tuning target is behavior, not memorized trail facts:
given a retrieved context block, answer concisely, cite sources, and abstain
when the context is missing or not enough for a safety claim.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import frontmatter

from backend.rag.prompt import SYSTEM_PROMPT, build_user_message

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOCS_DIR = ROOT / "chunking_setup" / "hiking_docs"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "data"


@dataclass
class LocalHit:
    text: str
    score: float
    metadata: dict[str, Any]


def _known(value: Any) -> bool:
    return value is not None and str(value).strip().lower() not in {"", "unknown", "none", "nan"}


def _text(value: Any) -> str:
    return str(value).strip()


def _load_docs(docs_dir: Path) -> list[LocalHit]:
    hits: list[LocalHit] = []
    for path in sorted(docs_dir.glob("*.md")):
        post = frontmatter.load(path)
        metadata = dict(post.metadata)
        name = metadata.get("name")
        if not _known(name):
            name = path.stem.replace("_", " ").replace("-", " ").title()
            metadata["name"] = name
        body = post.content.strip()
        if not body:
            print(f"Skipping {path.name}: File body is empty.")
            continue
        hits.append(LocalHit(text=body, score=1.0, metadata=metadata))
    return hits


def _message(query: str, hits: list[LocalHit], answer: str) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(query, hits)},
            {"role": "assistant", "content": answer},
        ]
    }


def _trail_answer(hit: LocalHit) -> str:
    meta = hit.metadata
    name = _text(meta["name"])
    parts = [f"Traseul este {name} [1]."]

    if _known(meta.get("marking")):
        parts.append(f"Marcajul este {meta['marking']} [1].")
    if _known(meta.get("difficulty")):
        parts.append(f"Dificultatea SAC este {meta['difficulty']} [1].")
    if _known(meta.get("duration")):
        parts.append(f"Timpul estimat este {meta['duration']} [1].")

    if len(parts) == 1:
        parts.append("Contextul nu oferă detalii sigure despre dificultate sau durată [1].")
    return " ".join(parts)


def _trail_examples(hit: LocalHit) -> Iterable[dict[str, Any]]:
    name = _text(hit.metadata["name"])
    yield _message(f"Ce știi despre traseul {name}?", [hit], _trail_answer(hit))

    if _known(hit.metadata.get("marking")):
        yield _message(
            f"Ce marcaj are traseul {name}?",
            [hit],
            f"Marcajul pentru traseul {name} este {hit.metadata['marking']} [1].",
        )

    if _known(hit.metadata.get("duration")):
        yield _message(
            f"Cât durează traseul {name}?",
            [hit],
            f"Contextul indică un timp estimat de {hit.metadata['duration']} pentru {name} [1].",
        )

    yield _message(
        f"Este sigur să parcurg mâine traseul {name} fără echipament?",
        [hit],
        (
            "Nu pot confirma că traseul este sigur mâine sau că poate fi parcurs fără "
            "echipament, deoarece contextul nu conține prognoză, stare actuală a traseului "
            "sau o recomandare explicită de siguranță [1]."
        ),
    )


def _poi_label(poi_type: str) -> str:
    return {
        "alpine_hut": "cabană montană",
        "spring": "izvor",
        "drinking_water": "sursă de apă potabilă",
        "mountain_rescue": "punct Salvamont",
        "via_ferrata": "via ferrata",
        "peak": "vârf montan",
        "parking": "parcare",
    }.get(poi_type, "punct de interes")


def _poi_answer(hit: LocalHit) -> str:
    meta = hit.metadata
    name = _text(meta["name"])
    poi_type = _text(meta.get("poi_type") or "poi")
    label = _poi_label(poi_type)
    parts = [f"{name} este un punct de interes de tip {label} în contextul HikeLogic [1]."]

    if _known(meta.get("ele")):
        parts.append(f"Altitudinea indicată este {meta['ele']} m [1].")
    return " ".join(parts)


def _poi_examples(hit: LocalHit) -> Iterable[dict[str, Any]]:
    meta = hit.metadata
    name = _text(meta["name"])
    poi_type = _text(meta.get("poi_type") or "poi")
    label = _poi_label(poi_type)

    yield _message(f"Ce este {name}?", [hit], _poi_answer(hit))

    if poi_type in {"spring", "drinking_water"}:
        yield _message(
            f"Unde găsesc apă la {name}?",
            [hit],
            f"Contextul listează {name} ca {label} [1]. Verifică local dacă apa este disponibilă și potabilă înainte de consum.",
        )
    elif poi_type == "mountain_rescue":
        yield _message(
            f"Există Salvamont la {name}?",
            [hit],
            f"Da, contextul listează {name} ca punct Salvamont [1]. Pentru urgențe folosește canalele oficiale de alertare.",
        )
    elif poi_type == "alpine_hut":
        yield _message(
            f"Există cabană la {name}?",
            [hit],
            f"Da, contextul listează {name} ca {label} [1]. Nu inventez detalii despre program, locuri sau rezervări dacă nu apar în context.",
        )


def _negative_examples() -> list[dict[str, Any]]:
    no_hits: list[LocalHit] = []
    abstain = (
        "Nu am găsit surse relevante în baza de trasee pentru această întrebare, "
        "deci nu pot răspunde în siguranță."
    )
    return [
        _message("Care este cel mai bun restaurant sushi din București?", no_hits, abstain),
        _message("Ce hotel de lux recomanzi pentru plajă în Grecia?", no_hits, abstain),
    ]


def _build_examples(docs: list[LocalHit], max_trails: int, max_pois: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    trails = [hit for hit in docs if hit.metadata.get("type") == "trail"]
    pois = [hit for hit in docs if hit.metadata.get("type") == "poi"]
    rng.shuffle(trails)
    rng.shuffle(pois)

    examples: list[dict[str, Any]] = []
    for hit in trails[:max_trails]:
        examples.extend(_trail_examples(hit))
    for hit in pois[:max_pois]:
        examples.extend(_poi_examples(hit))
    examples.extend(_negative_examples())
    rng.shuffle(examples)
    return examples


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-trails", type=int, default=100)
    parser.add_argument("--max-pois", type=int, default=200)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    docs = _load_docs(args.docs_dir)
    print(f"Loaded {len(docs)} documents from {args.docs_dir}")
    examples = _build_examples(docs, args.max_trails, args.max_pois, args.seed)

    split_at = max(1, int(len(examples) * (1 - args.eval_ratio)))
    train_rows = examples[:split_at]
    eval_rows = examples[split_at:]

    _write_jsonl(args.output_dir / "train.jsonl", train_rows)
    _write_jsonl(args.output_dir / "eval.jsonl", eval_rows)
    print(f"Loaded docs: {len(docs)}")
    print(f"Training examples: {len(train_rows)}")
    print(f"Eval examples: {len(eval_rows)}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
