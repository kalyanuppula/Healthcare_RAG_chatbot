
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "healthcare_docs"


def load_healthcare_documents() -> list[Document]:
    """Load healthcare text files and preserve their source filenames."""

    if not DOCUMENTS_DIR.exists():
        raise FileNotFoundError(
            f"Healthcare documents folder not found: {DOCUMENTS_DIR}"
        )

    documents = []

    for file_path in sorted(DOCUMENTS_DIR.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8").strip()

        if not text:
            print(f"Skipping empty document: {file_path.name}")
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": file_path.name,
                    "source_path": str(file_path.relative_to(PROJECT_ROOT)),
                },
            )
        )

    if not documents:
        raise ValueError(
            f"No non-empty .txt files found in {DOCUMENTS_DIR}"
        )

    return documents


def split_healthcare_documents(
    documents: list[Document],
    chunk_size: int = 700,
    chunk_overlap: int = 100,
) -> list[Document]:
    """Split documents into overlapping chunks while preserving metadata."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be non-negative and smaller than chunk_size."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )

    chunks = splitter.split_documents(documents)

    # Remove tiny chunks, such as headings with no useful information.
    chunks = [
    chunk
    for chunk in chunks
    if len(chunk.page_content.strip()) >= 100
    ]

    for chunk_id, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = chunk_id

    return chunks


if __name__ == "__main__":
    documents = load_healthcare_documents()
    chunks = split_healthcare_documents(documents)

    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks created: {len(chunks)}")

    for chunk in chunks[:3]:
        print("\n" + "-" * 50)
        print(f"Source: {chunk.metadata['source']}")
        print(f"Chunk ID: {chunk.metadata['chunk_id']}")
        print(f"Start index: {chunk.metadata['start_index']}")
        print(f"Text preview: {chunk.page_content[:200]}")
