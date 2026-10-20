# Healthcare RAG Chatbot

A healthcare-focused Retrieval-Augmented Generation (RAG) chatbot that
answers questions using information retrieved from a collection of
healthcare documents. It uses semantic search to find relevant passages,
then sends that context to a language model to generate a grounded,
easy-to-understand response with source documents.

> **Important:** This project is for educational and informational
> purposes only. It is not a medical device and does not provide medical
> diagnosis or treatment. For personal medical concerns, consult a
> qualified healthcare professional.

## Live Demo

**Streamlit app:** \ https://healthcareragchatbot-ec3lcpayvh8lo8tgwyafkp.streamlit.app/

## Features

-   **Document-based answers:** Retrieves relevant passages from
    healthcare documents before generating a response.
-   **Semantic search:** Uses Hugging Face sentence embeddings and
    ChromaDB for vector search.
-   **Grounded responses:** Instructs the language model to answer from
    the retrieved context and state when the provided documents do not
    contain the answer.
-   **Source display:** Shows source documents used to support the
    answer.
-   **Simple web interface:** Built with Streamlit.
-   **Hosted LLM:** Uses Groq through LangChain, so cloud deployment
    does not depend on a locally running Ollama server.
-   **Persistent local vector store:** Uses ChromaDB to store document
    embeddings locally.
-   **Automated tests:** Includes project tests run with `pytest`.

## How It Works

1.  Healthcare documents are loaded and split into smaller text chunks.
2.  Hugging Face embeddings convert the chunks into vectors.
3.  ChromaDB stores the vectors and supports similarity search.
4.  When a user asks a question, the app retrieves relevant document
    chunks.
5.  The retrieved context and question are sent to the Groq-hosted
    language model.
6.  The app displays the generated answer and source documents.

## Architecture

``` text
User question
     |
     v
Streamlit UI (app.py)
     |
     v
RAG chain (src/rag_chain.py)
     |
     +----> Retriever (src/retriever.py)
     |              |
     |              v
     |        ChromaDB vector store
     |              ^
     |              |
     |        Hugging Face embeddings
     |
     v
Groq-hosted language model
     |
     v
Answer + source documents
```

## Technology Stack

-   **Python**
-   **Streamlit** --- web interface
-   **LangChain** --- RAG workflow and model integration
-   **Groq** --- hosted language model inference
-   **ChromaDB** --- vector database
-   **Hugging Face Sentence Transformers** --- text embeddings
-   **PyPDF** --- PDF document processing
-   **pytest** --- testing
-   **Streamlit Community Cloud** --- deployment

## Project Structure

``` text
Healthcare_RAG_chatbot/
├── app.py
├── requirements.txt
├── .gitignore
├── src/
│   ├── embeddings.py
│   ├── document_processor.py
│   ├── vector_store.py
│   ├── retriever.py
│   └── rag_chain.py
├── tests/
└── README.md
```

The exact contents of the `tests/` and source-document directories may
vary with your local project.

## Run Locally

### 1. Clone the repository

``` bash
git clone https://github.com/kalyanuppula/Healthcare_RAG_chatbot.git
cd Healthcare_RAG_chatbot
```

### 2. Create and activate a virtual environment

On macOS or Linux:

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

``` bash
python -m pip install -r requirements.txt
```

### 4. Configure the Groq API key

Create a `.env` file in the project root:

``` text
GROQ_API_KEY=your_groq_api_key
```

Get an API key from the [Groq Console](https://console.groq.com/).

**Never commit `.env` or publish your API key.** The repository's
`.gitignore` should keep `.env` out of Git.

### 5. Start the application

``` bash
streamlit run app.py
```

Streamlit will display a local URL in the terminal. Open it in your
browser.

## Build the Vector Store

The application can create the ChromaDB vector store when it is missing.
To rebuild it manually, run:

``` bash
python -m src.vector_store
```

Ensure your healthcare documents are in the location configured by
`src/document_processor.py` before building the store.

## Run Tests

``` bash
pytest -v
```

## Deploy to Streamlit Community Cloud

1.  Push the project to GitHub.

2.  Open [Streamlit Community Cloud](https://share.streamlit.io/).

3.  Create an app from this repository and select `main` as the branch.

4.  Set the main file path to `app.py`.

5.  In the app's **Advanced settings → Secrets**, add:

    ``` toml
    GROQ_API_KEY = "your_groq_api_key"
    ```

6.  Deploy the app.

The `chroma_db/` directory is excluded from Git, so the deployed app
needs access to the source documents and builds the vector store in its
cloud environment. Check your document paths and Streamlit Cloud
resource limits if the initial deployment takes time or fails.

## Security and Responsible Use

-   Keep API keys in `.env` locally and Streamlit Secrets in the cloud.
-   Never hard-code secrets into Python files or commit them to GitHub.
-   The chatbot is intended for general educational information, not
    diagnosis, treatment decisions, or emergency guidance.
-   Answers depend on the quality, accuracy, and coverage of the source
    documents and retrieved passages.
-   A generated answer can still be incomplete or incorrect; verify
    important medical information with reliable sources and qualified
    professionals.
-   Avoid entering personal or identifiable patient information into a
    demo application.

## Future Improvements

-   Add document upload and document-management features.
-   Improve retrieval quality with metadata filters and retrieval
    evaluation.
-   Add citations at passage or page level where available.
-   Expand automated tests for retrieval relevance and answer grounding.
-   Add monitoring for latency, errors, and model usage.

## Author

**Kalyan Uppula**

GitHub: [@kalyanuppula](https://github.com/kalyanuppula)

------------------------------------------------------------------------

If you find this project useful, feel free to explore the code and
suggest improvements.
