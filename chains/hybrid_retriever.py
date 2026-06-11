import time

from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

from chains.retriever import load_vector_store
from chains.reranker import rerank_documents
from utils.query_classifier import classify_query
from utils.tracing import record_trace_step, serialize_documents

MMR_K = 8
MMR_FETCH_K = 30
BM25_K = 8
RERANK_TOP_N = 5


def tokenize(text):
    return text.lower().split()


def copy_document_with_metadata(doc, metadata):
    return Document(
        page_content=doc.page_content,
        metadata={**doc.metadata, **metadata},
    )


def normalize_file_filters(file_names=None):
    if not file_names:
        return None
    normalized = {
        name.strip()
        for name in file_names
        if isinstance(name, str) and name.strip()
    }
    return normalized or None


def document_file_name(doc):
    return doc.metadata.get("file_name")


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


def filter_documents_by_file(documents, file_names=None):
    file_names = normalize_file_filters(file_names)
    if not file_names:
        return documents
    return [
        doc
        for doc in documents
        if document_file_name(doc) in file_names
    ]


def score_lookup_key(doc):
    return (
        doc.metadata.get("file_name"),
        doc.metadata.get("page"),
        doc.metadata.get("page_label"),
        doc.page_content[:160],
    )


def vector_score_lookup(vector_store, query, fetch_k, file_names=None):
    try:
        scored_docs = vector_store.similarity_search_with_score(query=query, k=fetch_k)
    except Exception:
        return {}

    lookup = {}
    for doc, score in scored_docs:
        if normalize_file_filters(file_names) and document_file_name(doc) not in file_names:
            continue
        lookup[score_lookup_key(doc)] = float(score)
    return lookup


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


def hybrid_retrieve_documents(query, active_index_path=None, file_names=None):
    started_at = time.perf_counter()
    vector_store = load_vector_store(active_index_path)
    file_names = normalize_file_filters(file_names)

    query_type = classify_query(query)

    vector_k = MMR_K
    bm25_k = BM25_K
    top_n = RERANK_TOP_N

    vector_filter = None
    if file_names and len(file_names) == 1:
        vector_filter = {"file_name": next(iter(file_names))}

    vector_kwargs = {
        "query": query,
        "k": vector_k,
        "fetch_k": MMR_FETCH_K,
        "lambda_mult": 0.7,
    }
    if vector_filter:
        vector_kwargs["filter"] = vector_filter

    vector_docs = vector_store.max_marginal_relevance_search(**vector_kwargs)
    vector_docs = filter_documents_by_file(vector_docs, file_names)
    vector_scores = vector_score_lookup(
        vector_store,
        query,
        MMR_FETCH_K,
        file_names=file_names,
    )
    vector_docs = [
        copy_document_with_metadata(
            doc,
            {
                "retrieval_source": "vector_mmr",
                "vector_score": vector_scores.get(score_lookup_key(doc)),
            },
        )
        for doc in vector_docs
    ]
    record_trace_step(
        "vector_retrieval",
        inputs={
            "query": query,
            "active_index_path": active_index_path,
            "file_names": sorted(file_names) if file_names else None,
            "search_type": "max_marginal_relevance",
            "k": vector_k,
            "fetch_k": MMR_FETCH_K,
            "lambda_mult": 0.7,
            "filter": vector_filter,
        },
        outputs={
            "chunk_count": len(vector_docs),
            "chunks": serialize_documents(vector_docs),
        },
        run_type="retriever",
    )

    all_docs = get_all_documents_from_vector_store(vector_store)
    all_docs = filter_documents_by_file(all_docs, file_names)

    if not all_docs:
        record_trace_step(
            "hybrid_retrieval_complete",
            inputs={
                "query": query,
                "file_names": sorted(file_names) if file_names else None,
            },
            outputs={
                "query_type": query_type,
                "top_n": top_n,
                "final_chunk_count": 0,
                "chunks": [],
                "reason": "no_documents_after_filter",
            },
            run_type="retriever",
            latency_seconds=time.perf_counter() - started_at,
        )
        return []

    tokenized_docs = [tokenize(doc.page_content) for doc in all_docs]

    bm25 = BM25Okapi(tokenized_docs)

    bm25_scores = bm25.get_scores(tokenize(query))

    scored_bm25_docs = list(zip(all_docs, bm25_scores))

    scored_bm25_docs = sorted(
        scored_bm25_docs,
        key=lambda x: x[1],
        reverse=True
    )

    bm25_docs = [
        copy_document_with_metadata(
            doc,
            {
                "retrieval_source": "bm25",
                "bm25_score": float(score),
            },
        )
        for doc, score in scored_bm25_docs[:bm25_k]
    ]
    bm25_scores_top = [
        float(score)
        for doc, score in scored_bm25_docs[:bm25_k]
    ]
    record_trace_step(
        "bm25_retrieval",
        inputs={
            "query": query,
            "k": bm25_k,
            "candidate_count": len(all_docs),
            "file_names": sorted(file_names) if file_names else None,
        },
        outputs={
            "chunk_count": len(bm25_docs),
            "scores": bm25_scores_top,
            "chunks": serialize_documents(bm25_docs),
        },
        run_type="retriever",
    )

    merged_docs = merge_documents(vector_docs, bm25_docs)
    record_trace_step(
        "retrieved_chunks",
        inputs={
            "query": query,
            "query_type": query_type,
            "file_names": sorted(file_names) if file_names else None,
        },
        outputs={
            "vector_chunk_count": len(vector_docs),
            "bm25_chunk_count": len(bm25_docs),
            "merged_chunk_count": len(merged_docs),
            "chunks": serialize_documents(merged_docs),
        },
        run_type="retriever",
        latency_seconds=time.perf_counter() - started_at,
    )

    reranked_docs = rerank_documents(
        query=query,
        documents=merged_docs,
        top_n=top_n
    )
    record_trace_step(
        "hybrid_retrieval_complete",
        inputs={"query": query},
        outputs={
            "query_type": query_type,
            "top_n": top_n,
            "final_chunk_count": len(reranked_docs),
            "chunks": serialize_documents(reranked_docs),
        },
        run_type="retriever",
        latency_seconds=time.perf_counter() - started_at,
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
