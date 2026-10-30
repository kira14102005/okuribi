from collections import defaultdict

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from retrieval.corpus import load_archive_docs
from retrieval.vector_search import get_vectorstore


def document_key(doc: Document) -> str:
    """Stable identity for deduplicating retrieved documents."""
    return str(
        doc.metadata.get("chunk_id")
        or doc.metadata.get("doc_id")
        or doc.metadata.get("id")
        or doc.page_content
    )


def reciprocal_rank_fusion(
    rankings: list[list[Document]],
    k: int = 60,
) -> list[tuple[Document, float]]:
    """Fuse ranked lists using Reciprocal Rank Fusion."""
    scores = defaultdict(float)
    documents = {}

    for ranking in rankings:
        seen = set()

        for rank, doc in enumerate(ranking, start=1):
            key = document_key(doc)

            if key in seen:
                continue

            seen.add(key)
            scores[key] += 1 / (k + rank)
            documents[key] = doc

    return sorted(
        ((documents[key], score) for key, score in scores.items()),
        key=lambda item: item[1],
        reverse=True,
    )


def hybrid_search(
    query: str,
    top_k: int = 5,
    candidate_k: int = 5,
) -> list[tuple[Document, float]]:
    """Combine dense FAISS search and BM25 keyword search."""
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    # Dense retrieval
    vectorstore = get_vectorstore()
    dense_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": candidate_k},
    )
    dense_docs = dense_retriever.invoke(query)

    # Lexical retrieval
    archive_docs = load_archive_docs()
    bm25_retriever = BM25Retriever.from_documents(
        archive_docs,
        k=candidate_k,
    )
    keyword_docs = bm25_retriever.invoke(query)

    # Merge rankings
    fused = reciprocal_rank_fusion([dense_docs, keyword_docs])
    return fused[:top_k]
