import argparse
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

load_dotenv()

import retrieval.embeddings as embed


@lru_cache(maxsize=1)
def get_vectorstore() -> FAISS:
    """Load FAISS once and reuse it across queries."""
    path = Path(embed.VECTORSTORE_PATH)

    if not path.exists():
        raise FileNotFoundError(f"FAISS vector store not found at: {path}")

    return FAISS.load_local(
        str(path),
        embed.embeddings,
        allow_dangerous_deserialization=True,
    )


def query_vectorstore(
    query: str,
    k: int = 5,
    filters: dict | None = None,
) -> list[tuple[Document, float]]:
    """Retrieve archive documents with their raw FAISS scores."""
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    if k < 1:
        raise ValueError("k must be at least 1.")

    vectorstore = get_vectorstore()

    return vectorstore.similarity_search_with_score(
        query,
        k=k,
        filter=filters,
        fetch_k=max(20, k * 4),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--type", type=str, default=None)
    args = parser.parse_args()

    filters = {"type": args.type} if args.type else None

    results = query_vectorstore(args.query, args.k, filters)

    for doc, score in results:
        print(f"\nDocument: {doc.metadata.get('doc_id', 'unknown')}")
        print(f"Title: {doc.metadata.get('title', 'Untitled')}")
        print(f"Raw FAISS score: {score:.4f}")
        print(doc.page_content)
