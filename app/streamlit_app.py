import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st

from chains.conversational_rag import create_conversational_rag_chain


st.set_page_config(
    page_title="Multi-PDF RAG Assistant",
    page_icon="📄",
    layout="wide"
)

st.markdown("""
<style>
.main-title {
    font-size: 2.4rem;
    font-weight: 800;
    margin-bottom: 0.2rem;
}
.subtitle {
    color: #6b7280;
    font-size: 1rem;
    margin-bottom: 2rem;
}
.feature-card {
    padding: 1rem;
    border-radius: 14px;
    background: #f8fafc;
    border: 1px solid #e5e7eb;
    margin-bottom: 0.8rem;
}
</style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.title("⚙️ RAG Controls")

    st.markdown("### Project Features")

    features = [
        "Multi-PDF Support",
        "Hybrid Retrieval",
        "BM25 + Vector Search",
        "Reranking",
        "Conversational Memory",
        "Query Rewriting",
        "Source Citations",
        "Strict Grounding"
    ]

    for feature in features:
        st.markdown(f"✅ {feature}")

    st.divider()

    if st.button("🧹 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption("Make sure Ollama is running before asking questions.")


st.markdown('<div class="main-title">📄 Multi-PDF Conversational RAG Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Ask questions from your indexed PDFs with memory, citations, hybrid retrieval, reranking, and grounded answers.</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
    <b>🔍 Retrieval</b><br>
    Hybrid BM25 + vector search
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
    <b>🧠 Memory</b><br>
    Summary + recent chat context
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
    <b>📌 Grounding</b><br>
    PDF-only answers with citations
    </div>
    """, unsafe_allow_html=True)


if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.info("Start by asking something like: **List the project titles mentioned in the PDF.**")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_question = st.chat_input("Ask a question from your PDFs...")

if user_question:
    st.session_state.messages.append({
        "role": "user",
        "content": user_question
    })

    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant PDF context..."):
            response = create_conversational_rag_chain(user_question)

        st.markdown(response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })