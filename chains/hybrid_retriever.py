from rank_bm25 import BM25Okapi

from chains.retriever import load_vector_store
from chains.reranker import rerank_documents
from utils.query_classifier import classify_query


def tokenize(text):
    return text.lower().split()


def get_all_documents_from_vector_store(vector_store):
    collection = vector_store.get()

    documents = []

    texts = collection["documents"]
    metadatas = collection["metadatas"]

    for text, metadata in zip(texts, metadatas):
        from langchain_core.documents import Document

        documents.append(
            Document(
                page_content=text,
                metadata=metadata
            )
        )

    return documents


def merge_documents(vector_docs, bm25_docs):
    seen = set()
    merged_docs = []

    for doc in vector_docs + bm25_docs:
        unique_key = (
            doc.metadata.get("source"),
            doc.metadata.get("page"),
            doc.page_content[:80]
        )

        if unique_key not in seen:
            seen.add(unique_key)
            merged_docs.append(doc)

    return merged_docs


def hybrid_retrieve_documents(query):
    vector_store = load_vector_store()

    query_type = classify_query(query)

    if query_type == "broad":
        vector_k = 24
        bm25_k = 10
        top_n = 10
    else:
        vector_k = 12
        bm25_k = 5
        top_n = 5

    vector_docs = vector_store.max_marginal_relevance_search(
        query=query,
        k=vector_k,
        fetch_k=50,
        lambda_mult=0.7
    )

    all_docs = get_all_documents_from_vector_store(vector_store)

    tokenized_docs = [tokenize(doc.page_content) for doc in all_docs]

    bm25 = BM25Okapi(tokenized_docs)

    bm25_scores = bm25.get_scores(tokenize(query))

    scored_bm25_docs = list(zip(all_docs, bm25_scores))

    scored_bm25_docs = sorted(
        scored_bm25_docs,
        key=lambda x: x[1],
        reverse=True
    )

    bm25_docs = [doc for doc, score in scored_bm25_docs[:bm25_k]]

    merged_docs = merge_documents(vector_docs, bm25_docs)

    reranked_docs = rerank_documents(
        query=query,
        documents=merged_docs,
        top_n=top_n
    )

    return reranked_docs


if __name__ == "__main__":
    query = "List the project titles mentioned in the PDF."

    docs = hybrid_retrieve_documents(query)

    print(f"\nRetrieved Documents: {len(docs)}\n")

    for i, doc in enumerate(docs):
        print(f"\n--- DOC {i + 1} ---")
        print("Source:", doc.metadata.get("source"))
        print("Page:", doc.metadata.get("page_label"))
        print(doc.page_content[:500])