from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

from retrieval.chunking import chunk_doc
from retrieval.corpus import load_corpus

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EMBEDDING_MODEL = "embeddinggemma-2"
VECTORSTORE_PATH = PROJECT_ROOT / "world" / "vectorstore"
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)


def main():
    chunks = chunk_doc(load_corpus())
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTORSTORE_PATH)
