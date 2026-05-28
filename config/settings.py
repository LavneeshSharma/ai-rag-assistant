import os

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

RETRIEVAL_K = 24
FETCH_K = 50
MMR_LAMBDA = 0.7
TOP_N = 10

MODEL_NAME = "llama-3.3-70b-versatile"
LLM_PROVIDER = "groq"
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VECTOR_DB_DIR = os.path.abspath(os.path.join(BASE_DIR, "vector_store", "db"))
VECTOR_INDEXES_DIR = os.path.abspath(os.path.join(BASE_DIR, "vector_store", "indexes"))
ACTIVE_INDEX_FILE = os.path.abspath(os.path.join(BASE_DIR, "vector_store", "active_index.json"))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "data"))

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
PARENT_CHUNK_SIZE = 1600
PARENT_CHUNK_OVERLAP = 250
CHILD_CHUNK_SIZE = 450
CHILD_CHUNK_OVERLAP = 80
