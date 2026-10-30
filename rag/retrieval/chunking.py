from langchain_core.documents import Document
from langchain_text_splitters.character import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
    length_function=len,
)


def chunk_doc(docs: list[dict]):
    docs = [
        Document(
            page_content=doc["text"],
            metadata={key: value for key, value in doc.items() if key != "text"},
        )
        for doc in docs
    ]

    split_chunks = splitter.split_documents(docs)
    for index, chunk in enumerate(split_chunks):
        chunk.metadata["chunk_id"] = f"{chunk.metadata['id']}_chunk_{index}"

    return split_chunks
