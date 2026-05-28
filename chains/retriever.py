import os

from langchain_chroma import Chroma

from utils.embeddings import create_embeddings
from chains.reranker import rerank_documents
from utils.query_classifier import classify_query
from config.settings import FETCH_K, MMR_LAMBDA
from vector_store.chroma_store import get_active_index_path


def load_vector_store():
    embedding_model = create_embeddings()
    active_index_path = get_active_index_path()

    if not active_index_path or not os.path.isdir(active_index_path):
        raise RuntimeError("No active Chroma index found. Upload and index PDFs first.")

    vector_store = Chroma(
        persist_directory=active_index_path,
        embedding_function=embedding_model
    )

    return vector_store


def retrieve_documents(query):
    vector_store = load_vector_store()

    query_type = classify_query(query)

    if query_type == "broad":
        retrieval_k = 24
        top_n = 10
    else:
        retrieval_k = 12
        top_n = 5

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": retrieval_k,
            "fetch_k": FETCH_K,
            "lambda_mult": MMR_LAMBDA
        }
    )

    initial_docs = retriever.invoke(query)

    reranked_docs = rerank_documents(
        query=query,
        documents=initial_docs,
        top_n=top_n
    )

    return reranked_docs


def retrieve_documents_with_scores(query, k=4):
    vector_store = load_vector_store()

    results = vector_store.similarity_search_with_score(
        query=query,
        k=k
    )

    return results
