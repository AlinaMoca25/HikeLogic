import json
import random
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "train_osm_mistral.jsonl"
TRAIN_FILE = BASE_DIR / "train_clean.jsonl"
VAL_FILE = BASE_DIR / "val_clean.jsonl"
TEST_FILE = BASE_DIR / "test_clean.jsonl"

SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1


def is_good_example(text: str) -> bool:
    bad_patterns = [
        "Vreau sa merg pe traseul acest traseu",
        "'unknown'",
        "## ",
        "Tip: Traseu de drumeție",
        "Dificultate (SAC Scale): unknown",
        "Marcaj: ",
        "Timp estimat: Nespecificat",
    ]
    return not any(p in text for p in bad_patterns)


def read_jsonl(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        text = obj.get("text", "")
        if isinstance(text, str) and text.strip():
            rows.append({"text": text.strip()})
    return rows


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")

    all_rows = read_jsonl(INPUT_FILE)
    unique_rows = list({r["text"]: r for r in all_rows}.values())
    clean_rows = [r for r in unique_rows if is_good_example(r["text"])]

    random.Random(SEED).shuffle(clean_rows)

    n = len(clean_rows)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)

    train_rows = clean_rows[:train_end]
    val_rows = clean_rows[train_end:val_end]
    test_rows = clean_rows[val_end:]

    write_jsonl(TRAIN_FILE, train_rows)
    write_jsonl(VAL_FILE, val_rows)
    write_jsonl(TEST_FILE, test_rows)

    print(f"Input rows: {len(all_rows)}")
    print(f"Unique rows: {len(unique_rows)}")
    print(f"Clean rows: {len(clean_rows)}")
    print(f"Train: {len(train_rows)} -> {TRAIN_FILE}")
    print(f"Val: {len(val_rows)} -> {VAL_FILE}")
    print(f"Test: {len(test_rows)} -> {TEST_FILE}")


if __name__ == "__main__":
    main()
