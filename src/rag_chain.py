import os
from langchain_groq import ChatGroq

from src import vector_store
from src.retriever import get_retriever
from dotenv import load_dotenv
load_dotenv()

def get_rag_answer(question: str) -> dict:
    if not question.strip():
        return {
            "question": question,
            "answer": "Please enter a healthcare question.",
            "sources": [],
        }
    retriever = get_retriever()
    vector_store = retriever.vectorstore
    results = vector_store.similarity_search_with_score(
    question, k=5
    )

    documents = [
    doc for doc, score in results
    if score <= 1.0
        ]

    if not documents:
        return {
            "question": question,
            "answer": "I could not find relevant information in the available medical documents.",
            "sources": [],
        }

    context = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}\n"
        f"Content: {doc.page_content}"
        for doc in documents
    )

    llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
)

    prompt = f"""You are a healthcare information assistant.

Rules:
- Answer only using the medical context provided below.
- Do not use outside knowledge or invent medical facts.
- If the context does not contain the answer, clearly say that the information is not available in the provided documents.
- Do not add statistics, medical thresholds, symptoms, treatments, or recommendations unless supported by the context.
- Provide general educational information, not a personal diagnosis.
- Use simple, clear language.
- If possible, identify the source document for the information.

Medical context:
{context}

Question: {question}

Answer:"""

    response = llm.invoke(prompt)

    sources = list(dict.fromkeys(
        doc.metadata.get("source", "Unknown")
        for doc in documents
    ))

    return {
        "question": question,
        "answer": response.content,
        "sources": sources,
    }



if __name__ == "__main__":
    while True:
        question = input("\nAsk a healthcare question (or type 'exit'): ")

        if question.lower().strip() == "exit":
            print("Goodbye!")
            break

        if not question.strip():
            print("Please enter a question.")
            continue

        result = get_rag_answer(question)

        print("\nAnswer:", result["answer"])
        print("\nSources:")

        if result["sources"]:
            for source in result["sources"]:
                print("-", source)
        else:
            print("No relevant sources found.")

