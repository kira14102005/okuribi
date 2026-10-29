"""Simple lexical baseline for Okuribi's archive RAG.

This is intentionally small and dependency-free. It is a baseline to test corpus
quality and metadata behavior before adding embeddings.
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from typing import Any

from retrieval.corpus import load_corpus

TOKEN_RE = re.compile(r"[a-z0-9']+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "in", "into", "is", "it", "of", "on", "or", "that", "the", "this",
    "to", "was", "were", "will", "with", "why", "does", "do", "what",
    "who", "when", "how", "i", "me", "my", "we", "you", "your",
}


def tokenize(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if token.lower() not in STOPWORDS
    ]


def score_document(query_tokens: list[str], document: dict[str, Any]) -> float:
    """Score exact token overlap, weighting title and metadata more than body text."""
    title_tokens = Counter(tokenize(document.get("title", "")))
    metadata_text = " ".join(
        str(document.get(key, ""))
        for key in ("type", "date", "author", "reliability", "clue_ids")
    )
    metadata_tokens = Counter(tokenize(metadata_text))
    body_tokens = Counter(tokenize(document.get("text", "")))

    score = 0.0
    for token in query_tokens:
        score += 3.0 * title_tokens[token]
        score += 1.5 * metadata_tokens[token]
        score += 1.0 * body_tokens[token]
    return score


def search_documents(
    query: str,
    filters: dict[str, str] | None = None,
    top_k: int = 5,
    corpus: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return ranked archive records; does not reveal hidden `forged` labels."""
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    documents = corpus if corpus is not None else load_corpus()
    filters = filters or {}
    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    candidates = []
    for doc in documents:
        if any(str(doc.get(key, "")).lower() != str(value).lower()
               for key, value in filters.items()):
            continue
        score = score_document(query_tokens, doc)
        if score > 0:
            result = dict(doc)
            result["score"] = round(score, 3)
            candidates.append(result)

    candidates.sort(key=lambda item: (-item["score"], item["id"]))
    return candidates[:top_k]


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Okuribi's archive corpus")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--type", dest="doc_type", help="Filter by exact document type")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    filters = {"type": args.doc_type} if args.doc_type else None
    results = search_documents(args.query, filters=filters, top_k=args.top_k)
    if not results:
        print("No matching records.")
        return

    for result in results:
        print(f"[{result['id']}] {result['title']} (score={result['score']})")
        print(f"  Date: {result['date']} | Type: {result['type']} | Reliability: {result['reliability']}")
        print(f"  {result['text']}")
        print()


if __name__ == "__main__":
    main()
