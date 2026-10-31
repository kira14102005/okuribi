import asyncio
import json

from langchain.mcp import MCPAdapter
from rag.mcp.docs_mcp_main import mcp

# ponytail: in-process client imports the server directly; switch to
# StdioTransport/`python -m rag.mcp.docs_mcp_main` subprocess only if the game
# and the server must run as separate processes.


def _format_doc(doc):
    metadata = doc.get("metadata", {})
    content = doc.get("page_content", "")
    mod_doc = {**metadata, "content": content}
    return mod_doc


async def main():
    async with MCPAdapter(mcp) as adapter:
        tools = await adapter.list_tools()
        print(f"Available tools: {sorted(t.name for t in tools)}")
        by_name = {t.name: t for t in tools}
        query = "When is the festival?"
        print(f"Query: {query}")
        result = await by_name["query_docs"].ainvoke({"query": query})
        result = result[0].get("text", [])
        result = json.loads(result)
        docs = [_format_doc(doc) for doc in result]
        for doc in docs:
            print(f"Document: {json.dumps(doc, indent=3)}")


if __name__ == "__main__":
    asyncio.run(main())
