import sys
import os

# Add root folder to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from src.rag_engine import RAGEngine

st.set_page_config(
    page_title="Enterprise AI Assistant",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Enterprise Business Process Assistant")
st.caption("Grounded Document QA System Powered by LangChain, FAISS & Groq Llama 3")

# Initialize RAG Engine
@st.cache_resource
def load_rag_engine():
    return RAGEngine()

try:
    engine = load_rag_engine()
    st.sidebar.success("FAISS Vector Store Loaded")
except Exception as e:
    st.sidebar.error(f"Error loading vector store: {e}")
    st.stop()

st.sidebar.markdown("### 📊 Document Metadata")
st.sidebar.info("📄 Active Document: **WorldEnergyOutlook2025.pdf**\n\n🔍 Model: **Llama-3.1-8b-instant**")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("🔍 View Retrieved Source Context"):
                for src in message["sources"]:
                    st.write(f"**Page {src['page']}**")
                    st.caption(src["content"])

# React to user input
if prompt := st.chat_input("Ask a question about the energy report..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Retrieving document context and generating response..."):
            result = engine.answer_question(prompt)
            response = result["answer"]
            sources = result["sources"]

            st.markdown(response)

            if sources and "Information not available" not in response:
                with st.expander("🔍 View Retrieved Source Context"):
                    for src in sources:
                        st.write(f"**Page {src['page']}**")
                        st.caption(src["content"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "sources": sources if "Information not available" not in response else []
    })