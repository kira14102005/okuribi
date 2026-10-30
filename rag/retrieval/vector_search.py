import argparse

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

import retrieval.embeddings as embed

load_dotenv()  # Load environment variables from .env file


def query_vectorstore(query: str) -> list[Document]:
    """Query the FAISS vector store and return the most relevant documents."""
    vectorstore = FAISS.load_local(
        embed.VECTORSTORE_PATH, embed.embeddings, allow_dangerous_deserialization=True
    )
    docs = vectorstore.similarity_search(query, k=2)
    return docs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query the FAISS vector store")
    parser.add_argument("query", help="Search query")
    args = parser.parse_args()

    results = query_vectorstore(args.query)
    if not results:
        print("No matching records.")
    else:
        for result in results:
            print(f"[{result.metadata['id']}] {result.metadata['title']}")
            print(
                f"  Date: {result.metadata['date']} | Type: {result.metadata['type']} | Reliability: {result.metadata['reliability']}"
            )
            print(f"  {result.page_content}")
            print()
