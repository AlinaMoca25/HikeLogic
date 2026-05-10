from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import uuid

import frontmatter

from .embedder import upsert_trail_batch

DOCS_PATH = Path(__file__).resolve().parents[2] / "chunking_setup" / "hiking_docs"


@dataclass
class IngestionResult:
    total_files: int
    indexed: int
    failed: int


def _point_id(raw_id: object, filename: str) -> int | str:
    if not raw_id:
        raise ValueError(f"{filename}: missing required frontmatter field 'id'")

    value = str(raw_id)
    if value.startswith("osm_"):
        value = value.removeprefix("osm_")

    if value.isdigit():
        return int(value)

    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"https://www.openstreetmap.org/{value}"))


def _extract_title(content: str, filename: str) -> str:
    for line in content.splitlines():
        if line.startswith("# "):
            return line.removeprefix("# ").strip()
    return Path(filename).stem


def _metadata_from_post(post: frontmatter.Post, filename: str) -> dict:
    content = post.content.strip()
    name = post.get("name") or _extract_title(content, filename)

    return {
        "id": _point_id(post.get("id"), filename),
        "osm_id": str(post.get("id")),
        "osm_type": post.get("osm_type", "unknown"),
        "entity_type": post.get("type", "trail"),
        "poi_type": post.get("poi_type", "none"),
        "name": name,
        "difficulty": post.get("difficulty", "unknown"),
        "region": post.get("region", "unknown"),
        "marking": post.get("marking", "none"),
        "marking_color": post.get("marking_color", "unknown"),
        "marking_shape": post.get("marking_shape", "unknown"),
        "elevation_gain": post.get("elevation_gain", "unknown"),
        "duration": post.get("duration", "unknown"),
        "ele": post.get("ele"),
        "from": post.get("from", "unknown"),
        "to": post.get("to", "unknown"),
        "lat": post.get("lat"),
        "lon": post.get("lon"),
        "source": post.get("source", "openstreetmap"),
        "osm_url": post.get("osm_url"),
    }


def _flush_batch(batch: list[tuple[dict, str]], indexed: int) -> int:
    upsert_trail_batch(batch)
    return indexed + len(batch)


def run_ingestion(docs_path: str | Path = DOCS_PATH, batch_size: int = 32) -> IngestionResult:
    path = Path(docs_path)
    if not path.exists():
        raise FileNotFoundError(f"Docs path not found: {path}")

    files = sorted(path.glob("*.md"))
    if not files:
        raise RuntimeError(f"No markdown documents found in {path}")

    items: list[tuple[dict, str]] = []
    failures: list[str] = []

    print(f"Found {len(files)} hiking documents. Validating metadata...")

    for file_path in files:
        try:
            post = frontmatter.load(file_path)
            content = post.content.strip()
            if not content:
                raise ValueError(f"{file_path.name}: empty document body")

            metadata = _metadata_from_post(post, file_path.name)
            items.append((metadata, content))
        except Exception as exc:
            failures.append(f"{file_path.name}: {exc}")

    if failures:
        preview = "\n".join(failures[:10])
        raise RuntimeError(
            f"Validated {len(items)}/{len(files)} documents; "
            f"{len(failures)} failed before indexing.\n{preview}"
        )

    indexed = 0
    print(f"Starting Qdrant ingestion in batches of {batch_size}...")

    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        try:
            indexed = _flush_batch(batch, indexed)
        except Exception as exc:
            raise RuntimeError(
                f"Indexed {indexed}/{len(items)} documents; "
                f"batch starting at {start} failed: {exc}"
            ) from exc

    print(f"Indexed {indexed} documents in Qdrant.")
    return IngestionResult(total_files=len(files), indexed=indexed, failed=0)
