from langchain_huggingface import HuggingFaceEmbeddings

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def get_embedding_model() -> HuggingFaceEmbeddings:
    """Load the local sentence-transformer embedding model."""

    return HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


if __name__ == "__main__":
    embeddings = get_embedding_model()

    sample_texts = [
        "Diabetes is a condition that affects blood glucose.",
        "Hypertension means high blood pressure.",
    ]

    vectors = embeddings.embed_documents(sample_texts)
    query_vector = embeddings.embed_query("What is diabetes?")

    print(f"Model: {MODEL_NAME}")
    print(f"Number of document vectors: {len(vectors)}")
    print(f"Vector dimensions: {len(vectors[0])}")
    print(f"Query vector dimensions: {len(query_vector)}")
