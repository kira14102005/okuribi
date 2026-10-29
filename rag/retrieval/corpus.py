"""Export the player-facing archive view without leaking hidden world truth."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORLD_BIBLE_PATH = PROJECT_ROOT / "world" / "okuribi_world_bible.json"
CORPUS_PATH = PROJECT_ROOT / "world" / "docs.json"

PUBLIC_FIELDS = ("id", "title", "type", "date", "author", "reliability", "text", "clue_ids")


def load_world_bible(path: Path = WORLD_BIBLE_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def public_document(document: dict[str, Any]) -> dict[str, Any]:
    """Return only fields allowed in the player-facing archive index."""
    return {key: document[key] for key in PUBLIC_FIELDS if key in document}


def build_corpus(
    world_bible_path: Path = WORLD_BIBLE_PATH,
    output_path: Path = CORPUS_PATH,
) -> list[dict[str, Any]]:
    world = load_world_bible(world_bible_path)
    docs = [public_document(doc) for doc in world["documents"]]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    return docs


def load_corpus(path: Path = CORPUS_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        build_corpus(output_path=path)
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
