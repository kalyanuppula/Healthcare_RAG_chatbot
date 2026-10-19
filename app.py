import os
import streamlit as st

try:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

from src.rag_chain import get_rag_answer

st.set_page_config(
    page_title="Healthcare RAG Chatbot",
    page_icon="🏥",
    layout="centered",
)

st.title("🏥 Healthcare RAG Chatbot")
st.caption("Healthcare information grounded in your medical documents.")

st.info(
    "This chatbot provides general educational information, "
    "not a personal diagnosis or a substitute for professional medical advice."
)

question = st.text_input(
    "Ask a healthcare question",
    placeholder="For example: What is diabetes?",
)

if st.button("Get Answer", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        try:
            with st.spinner("Searching documents and generating your answer..."):
                result = get_rag_answer(question)

            st.subheader("Answer")
            st.write(result["answer"])

            st.subheader("Sources")
            if result["sources"]:
                for source in result["sources"]:
                    st.write(f"- {source}")
            else:
                st.write("No relevant sources found.")

        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.caption(
                "Check that Ollama is running and the required model is available."
            )
