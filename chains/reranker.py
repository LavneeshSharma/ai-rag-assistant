from sentence_transformers import CrossEncoder

from config.settings import RERANKER_MODEL


reranker_model = CrossEncoder(RERANKER_MODEL)


def rerank_documents(query, documents, top_n):
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

    return [doc for doc, score in scored_docs[:top_n]]