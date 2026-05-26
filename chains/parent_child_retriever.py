import pickle

from langchain_chroma import Chroma

from utils.embeddings import create_embeddings
from chains.reranker import rerank_documents


def load_parent_child_store():
    embedding_model = create_embeddings()

    child_vector_store = Chroma(
        persist_directory="vector_store/parent_child_db",
        embedding_function=embedding_model
    )

    with open("vector_store/parent_store.pkl", "rb") as file:
        parent_store = pickle.load(file)

    return child_vector_store, parent_store


def parent_child_retrieve_documents(query, top_k=6, top_n=2):
    child_vector_store, parent_store = load_parent_child_store()

    child_docs = child_vector_store.similarity_search(
        query=query,
        k=top_k
    )

    parent_docs = []
    seen_parent_ids = set()

    for child in child_docs:
        parent_id = child.metadata.get("parent_id")

        if parent_id and parent_id not in seen_parent_ids:
            seen_parent_ids.add(parent_id)
            parent_docs.append(parent_store[parent_id])

    reranked_parent_docs = rerank_documents(
        query=query,
        documents=parent_docs,
        top_n=top_n
    )

    return reranked_parent_docs


if __name__ == "__main__":
    query = "Customer Segmentation Analysis customer behavior purchase patterns"

    docs = parent_child_retrieve_documents(query)

    print(f"\nRetrieved Parent Documents: {len(docs)}\n")

    for i, doc in enumerate(docs):
        print(f"\n--- PARENT DOC {i + 1} ---")
        print("Source:", doc.metadata.get("source"))
        print("Page:", doc.metadata.get("page_label"))
        print(doc.page_content[:1000])