
from pathlib import Path

from langchain_chroma import Chroma

from src.embeddings import get_embedding_model
from src.vector_store import CHROMA_DB_DIR, COLLECTION_NAME


def get_retriever():
    """Load the saved vector database and return a retriever."""

    if not Path(CHROMA_DB_DIR).exists():
        raise FileNotFoundError(
            "ChromaDB not found. Run: python -m src.vector_store"
        )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embedding_model(),
        persist_directory=str(CHROMA_DB_DIR),
    )

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )


if __name__ == "__main__":
    retriever = get_retriever()

    question = "What factors increase the risk of developing type 2 diabetes?"
    results = retriever.invoke(question)

    print(f"Question: {question}")
    print(f"Relevant chunks retrieved: {len(results)}")

    for index, document in enumerate(results, start=1):
        print("\n" + "=" * 60)
        print(f"Result: {index}")
        print(f"Source: {document.metadata.get('source', 'Unknown')}")
        print(f"Chunk ID: {document.metadata.get('chunk_id', 'Unknown')}")
        print(f"Content: {document.page_content[:500]}")
