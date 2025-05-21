import streamlit as st
from rag_utils import parse_file, split_text, build_vectorstore, query_with_gemini
import numpy as np

st.set_page_config(page_title="Gemini RAG Chatbot", layout="wide")
st.title("🤖 Chat with Your Documents using Gemini")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_files = st.file_uploader(
    "📁 Upload your documents (PDF, DOCX, TXT)",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

if uploaded_files:
    with st.spinner("Reading and indexing..."):
        full_text = "\n".join([parse_file(file) for file in uploaded_files])
        text_chunks = split_text(full_text)
        index, chunk_list, embed_model = build_vectorstore(text_chunks)
    st.success("Documents processed. Ask a question!")

    user_query = st.text_input("💬 Ask a question about your documents:")

    if user_query:
        query_embedding = embed_model.encode([user_query])
        _, indices = index.search(np.array(query_embedding), k=5)
        top_chunks = [chunk_list[i] for i in indices[0]]
        answer = query_with_gemini(user_query, top_chunks)

        st.session_state.chat_history.append(("You", user_query))
        st.session_state.chat_history.append(("Bot", answer))

    for speaker, msg in st.session_state.chat_history:
        if speaker == "You":
            st.markdown(f"**🧑 {speaker}:** {msg}")
        else:
            st.markdown(f"**🤖 {speaker}:** {msg}")
else:
    st.info("Upload documents to begin.")
