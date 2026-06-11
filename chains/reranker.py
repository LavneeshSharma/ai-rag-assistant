import time

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from config.settings import RERANKER_MODEL
from utils.tracing import record_trace_step, serialize_document


reranker_model = None


def get_reranker_model():
    global reranker_model
    if reranker_model is None:
        reranker_model = CrossEncoder(RERANKER_MODEL)
    return reranker_model


def rerank_documents(query, documents, top_n):
    if not documents:
        record_trace_step(
            "reranked_chunks",
            inputs={"query": query, "candidate_count": 0, "top_n": top_n},
            outputs={"chunk_count": 0, "chunks": []},
            run_type="retriever",
        )
        return []

    started_at = time.perf_counter()
    pairs = [(query, doc.page_content) for doc in documents]

    scores = get_reranker_model().predict(pairs)

    scored_docs = list(zip(documents, scores))

    scored_docs = sorted(
        scored_docs,
        key=lambda x: x[1],
        reverse=True
    )

    top_scored_docs = [
        Document(
            page_content=doc.page_content,
            metadata={
                **doc.metadata,
                "rerank_score": float(score),
            },
        )
        for doc, score in scored_docs[:top_n]
    ]
    record_trace_step(
        "reranked_chunks",
        inputs={
            "query": query,
            "candidate_count": len(documents),
            "top_n": top_n,
        },
        outputs={
            "chunk_count": len(top_scored_docs),
            "chunks": [
                serialize_document(doc, index)
                for index, doc in enumerate(top_scored_docs)
            ],
        },
        run_type="retriever",
        latency_seconds=time.perf_counter() - started_at,
    )

    return top_scored_docs
