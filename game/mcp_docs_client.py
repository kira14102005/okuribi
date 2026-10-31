import asyncio

from fastmcp import Client

from rag.mcp.docs_mcp_main import mcp

# ponytail: in-process client imports the server directly; switch to
# StdioTransport/`python -m rag.mcp.docs_mcp_main` subprocess only if the game
# and the server must run as separate processes.


async def main():
    client = Client(mcp)
    async with client:
        tools = await client.list_tools()
        print(f"Available tools: {sorted(t.name for t in tools)}")
        query = "When is the festival?"
        print(f"Query: {query}")
        result = await client.call_tool("query_docs", {"query": query})
        print(f"Result: {result.data}")

    print(f"Result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
