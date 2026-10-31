import argparse
import asyncio
import json

from langchain.mcp import MCPAdapter

from rag.mcp.docs_mcp_main import mcp

# StdioTransport/`python -m rag.mcp.docs_mcp_main` subprocess only if the game
# and the server must run as separate processes.


def _format_doc(doc):
    metadata = doc.get("metadata", {})
    content = doc.get("page_content", "")
    mod_doc = {**metadata, "content": content}
    return mod_doc


async def search_docs(query: str = "When is the festival?"):
    async with MCPAdapter(mcp) as adapter:
        tools = await adapter.list_tools()
        print(f"Available tools: {sorted(t.name for t in tools)}")
        by_name = {t.name: t for t in tools}
        print(f"Query: {query}")
        result = await by_name["query_docs"].ainvoke({"query": query})
        result = result[0].get("text", [])
        result = json.loads(result)
        docs = [_format_doc(doc) for doc in result]
        for doc in docs:
            print(f"Document: {json.dumps(doc, indent=3)}")
        return docs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search documents via MCP.")

    parser.add_argument(
        "--query",
        type=str,
        default="When is the festival?",
        help="Search query for documents.",
    )
    args = parser.parse_args()
    asyncio.run(search_docs(query=args.query))
