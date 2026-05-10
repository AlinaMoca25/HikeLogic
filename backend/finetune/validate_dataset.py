"""Validate HikeLogic fine-tuning JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"


def _validate_row(row: dict[str, Any], path: Path, line_no: int) -> None:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        raise ValueError(f"{path}:{line_no} expected exactly 3 chat messages")

    roles = [message.get("role") for message in messages]
    if roles != ["system", "user", "assistant"]:
        raise ValueError(f"{path}:{line_no} invalid roles: {roles}")

    for message in messages:
        if not isinstance(message.get("content"), str) or not message["content"].strip():
            raise ValueError(f"{path}:{line_no} empty message content")

    assistant = messages[-1]["content"]
    if "[1]" not in assistant and "Nu am găsit surse relevante" not in assistant:
        raise ValueError(f"{path}:{line_no} assistant answer is missing citation or abstention")


def validate_file(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            _validate_row(json.loads(line), path, line_no)
            count += 1

    if count == 0:
        raise ValueError(f"{path} is empty")
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, default=DEFAULT_DATA_DIR / "train.jsonl")
    parser.add_argument("--eval-file", type=Path, default=DEFAULT_DATA_DIR / "eval.jsonl")
    args = parser.parse_args()

    train_count = validate_file(args.train_file)
    eval_count = validate_file(args.eval_file)
    print(f"Validated train examples: {train_count}")
    print(f"Validated eval examples: {eval_count}")


if __name__ == "__main__":
    main()
