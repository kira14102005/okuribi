import json
import os

from dotenv import load_dotenv
from fastmcp import FastMCP
from retrieval.hybrid_search import hybrid_search

load_dotenv()

mcp = FastMCP(name="okuribi_docs_mcp")


@mcp.tool
def query_docs(query: str, k: int = 5, type: str | None = None) -> list[dict]:
    """Retrieve archive documents relevant to a query."""
    from retrieval.vector_search import query_vectorstore

    filters = {"type": type} if type else None
    results = hybrid_search(query, top_k=k, filters=filters)
    return [dict(doc, score=score) for doc, score in results]
