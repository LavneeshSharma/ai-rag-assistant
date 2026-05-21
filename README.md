# 🧠 AI RAG Pipeline

A modular Retrieval-Augmented Generation (RAG) system built using LangChain, ChromaDB, HuggingFace embeddings, and Phi-3 via Ollama.

---

# 🚀 Features

- PDF document ingestion
- Text chunking
- Embedding generation
- Chroma vector database
- Semantic similarity search
- Retrieval-Augmented Generation (RAG)
- Local LLM inference using Phi-3
- Fully local and free setup

---

# 🔥 RAG Pipeline

```text
PDF
 ↓
Text Extraction
 ↓
Chunking
 ↓
Embeddings
 ↓
ChromaDB
 ↓
Retriever
 ↓
Prompt Injection
 ↓
LLM Response

## 📂 Project Structure

```bash
utils/
├── pdf_loader.py
├── chunker.py
└── embeddings.py

vector_store/
├── chroma_store.py
└── db/

chains/
├── retriever.py
└── rag_chain.py

prompts/
└── rag_prompt.t

## 📘 What I Learned

Through this project, I understood how modern RAG (Retrieval-Augmented Generation) systems actually work behind the scenes. Instead of simply calling an LLM, I built the complete pipeline step-by-step — from loading PDFs and chunking text to generating embeddings, storing vectors in ChromaDB, retrieving relevant chunks semantically, and finally generating grounded responses using Phi-3.

This project helped me understand:
- semantic search
- embeddings and vector representations
- vector databases
- retrieval pipelines
- prompt injection
- grounded AI responses
- modular GenAI architecture
