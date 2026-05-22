from sentence_transformers import CrossEncoder


reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank_documents(query, documents, top_n=4):
    """
    Rerank retrieved documents using a cross-encoder reranker.
    """

    if not documents:
        return []

    pairs = [(query, doc.page_content) for doc in documents]

    scores = reranker_model.predict(pairs)

    scored_docs = list(zip(documents, scores))

    scored_docs = sorted(
        scored_docs,
        key=lambda x: x[1],
        reverse=True
    )

    reranked_docs = [doc for doc, score in scored_docs[:top_n]]

    return reranked_docs