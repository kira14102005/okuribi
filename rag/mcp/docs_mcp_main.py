from dotenv import load_dotenv
from fastmcp import FastMCP

from rag.retrieval.hybrid_search import hybrid_search

load_dotenv()

mcp = FastMCP(name="okuribi_docs_mcp")


@mcp.tool
def query_docs(query: str, k: int = 5, type: str | None = None) -> list[dict]:
    """Retrieve archive documents relevant to a query."""

    filters = {"type": type} if type else None
    results = hybrid_search(query, top_k=k, filters=filters)
    return [dict(doc, score=score) for doc, score in results]


if __name__ == "__main__":
    mcp.run()
