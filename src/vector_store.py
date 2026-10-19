
from pathlib import Path

from langchain_chroma import Chroma

from src.document_processor import (
    load_healthcare_documents,
    split_healthcare_documents,
)
from src.embeddings import get_embedding_model


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "healthcare_documents"


def create_vector_store() -> Chroma:
    """Rebuild the persistent ChromaDB healthcare collection."""

    documents = load_healthcare_documents()
    chunks = split_healthcare_documents(documents)
    embeddings = get_embedding_model()

    store_config = {
        "collection_name": COLLECTION_NAME,
        "embedding_function": embeddings,
        "persist_directory": str(CHROMA_DB_DIR),
    }

    # Remove the previous collection if it exists.
    existing_store = Chroma(**store_config)
    existing_store.delete_collection()

    # Create a fresh collection and insert the current chunks.
    vector_store = Chroma(**store_config)
    vector_store.add_documents(chunks)

    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks stored: {len(chunks)}")
    print(f"Collection name: {COLLECTION_NAME}")
    print(f"Database location: {CHROMA_DB_DIR}")

    return vector_store


if __name__ == "__main__":
    create_vector_store()
