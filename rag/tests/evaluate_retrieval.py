def get_doc_id(doc):
    """Map a retrieved chunk back to its parent document."""
    return str(doc.metadata.get("doc_id") or doc.metadata.get("id"))


def evaluate_retriever(search_fn, dataset, k=5):
    hits = 0
    recalls = []
    precisions = []
    reciprocal_ranks = []

    for item in dataset:
        relevant = set(item["relevant_doc_ids"])
        results = search_fn(item["query"])
        docs = [
            result[0] if isinstance(result, tuple) else result for result in results
        ]

        # Avoid counting multiple chunks from one document twice.
        ranked_ids = list(dict.fromkeys(get_doc_id(d) for d in docs))
        ranked_ids = ranked_ids[:k]

        found = set(ranked_ids) & relevant
        hits += bool(found)
        recalls.append(len(found) / len(relevant))
        precisions.append(len(found) / k)

        first_rank = next(
            (i for i, doc_id in enumerate(ranked_ids, start=1) if doc_id in relevant),
            None,
        )
        reciprocal_ranks.append(1 / first_rank if first_rank is not None else 0)

    n = len(dataset)
    if n == 0:
        raise ValueError("Evaluation dataset cannot be empty.")

    return {
        f"Hit Rate@{k}": hits / n,
        f"Recall@{k}": sum(recalls) / n,
        f"Precision@{k}": sum(precisions) / n,
        "MRR": sum(reciprocal_ranks) / n,
    }


GOLDEN_QUERIES = [
    {
        "query": "What date was Aoi's transfer recorded?",
        "relevant_doc_ids": ["DOC_02"],
    },
    {
        "query": "What happened to the bell clapper?",
        "relevant_doc_ids": ["DOC_03"],
    },
    # Add more queries and verify their relevant IDs.
]

from retrieval.hybrid_search import hybrid_search

if __name__ == "__main__":
    print(evaluate_retriever(hybrid_search, GOLDEN_QUERIES))
