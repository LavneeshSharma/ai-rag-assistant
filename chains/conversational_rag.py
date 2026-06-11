import os
import re
import time

from dotenv import load_dotenv

from utils.llm import create_llm
from utils.tracing import (
    RAGTrace,
    extract_usage_and_cost,
    record_trace_step,
    serialize_documents,
    truncate_prompt,
)
from langchain_core.prompts import PromptTemplate

from chains.hybrid_retriever import hybrid_retrieve_documents
from chains.query_rewriter import rewrite_query

load_dotenv()


chat_history = []
conversation_summary = ""
MAX_HISTORY = 3
LOW_CONFIDENCE_ANSWER = "I could not find this clearly in the uploaded PDF."
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does",
    "for", "from", "give", "how", "in", "is", "it", "me", "of", "on", "or",
    "pdf", "please", "the", "this", "to", "what", "when", "where", "which",
    "who", "why", "with",
}

def load_prompt():
    with open("prompts/rag_prompt.txt", "r") as file:
        template = file.read()

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )

    return prompt


def format_context(documents):
    formatted_chunks = []
    for index, doc in enumerate(documents, start=1):
        source = doc.metadata.get("file_name") or os.path.basename(
            doc.metadata.get("source", "")
        )
        page = doc.metadata.get("page_label") or doc.metadata.get("page")
        header = f"[Chunk {index} | Source: {source or 'Unknown'}"
        if page is not None:
            header += f" | Page: {page}"
        header += "]"
        formatted_chunks.append(f"{header}\n{doc.page_content}")

    return "\n\n".join(formatted_chunks)


def format_sources(documents):
    sources = {}

    for doc in documents:
        source = doc.metadata.get("file_name") or os.path.basename(
            doc.metadata.get("source", "")
        )
        page = doc.metadata.get("page_label")
        if page is None and doc.metadata.get("page") is not None:
            raw_page = doc.metadata.get("page")
            page = raw_page + 1 if isinstance(raw_page, int) else raw_page
        if not source or page is None:
            continue
        sources.setdefault(source, set()).add(str(page))

    formatted = []
    for source, pages in sorted(sources.items()):
        page_list = ", ".join(
            f"Page {page}"
            for page in sorted(pages, key=lambda p: int(p) if p.isdigit() else p)
        )
        formatted.append(f"{source} — {page_list}")

    return "\n".join(formatted)


def retrieval_page_summary(documents):
    return [
        {
            "rank": index + 1,
            "file_name": doc.metadata.get("file_name")
            or os.path.basename(doc.metadata.get("source", "")),
            "page": doc.metadata.get("page"),
            "page_label": doc.metadata.get("page_label"),
            "retrieval_source": doc.metadata.get("retrieval_source"),
            "vector_score": doc.metadata.get("vector_score"),
            "bm25_score": doc.metadata.get("bm25_score"),
            "rerank_score": doc.metadata.get("rerank_score"),
            "content_length": len(doc.page_content or ""),
        }
        for index, doc in enumerate(documents)
    ]


def query_terms(text):
    terms = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return {
        term
        for term in terms
        if len(term) > 2 and term not in STOP_WORDS
    }


def context_weakness_reason(question, rewritten_question, documents, context):
    if not documents:
        return "no_retrieved_documents"
    if len(context.strip()) < 120:
        return "context_too_short"

    terms = query_terms(f"{question} {rewritten_question}")
    if terms:
        context_lower = context.lower()
        matched_terms = {term for term in terms if term in context_lower}
        overlap = len(matched_terms) / max(len(terms), 1)
        if overlap == 0:
            return "no_query_term_overlap"

    rerank_scores = [
        doc.metadata.get("rerank_score")
        for doc in documents
        if doc.metadata.get("rerank_score") is not None
    ]
    if rerank_scores and max(rerank_scores) < -8:
        return "low_rerank_score"

    return None


def is_fallback_response(response_text):
    return "I could not find" in response_text

def update_conversation_summary():
    global conversation_summary

    if len(chat_history) <= MAX_HISTORY:
        return

    older_history = chat_history[:-MAX_HISTORY]

    history_text = ""

    for item in older_history:
        history_text += f"User: {item['question']}\n"
        history_text += f"Assistant: {item['answer']}\n\n"

    llm = create_llm()

    summary_prompt = f"""
Summarize the following conversation briefly.
Keep only important context needed for future questions.

Existing Summary:
{conversation_summary}

Old Conversation:
{history_text}

Updated Summary:
"""

    response = llm.invoke(summary_prompt)

    conversation_summary = response.content

    del chat_history[:-MAX_HISTORY]

def format_chat_history(chat_messages=None):
    if chat_messages is not None:
        recent_messages = chat_messages[-MAX_HISTORY * 2:]
        if not recent_messages:
            return "No previous conversation."

        history_text = "Recent Conversation:\n"
        for message in recent_messages:
            label = "User" if message.get("role") == "user" else "Assistant"
            history_text += f"{label}: {message.get('content', '')}\n"
        return history_text

    recent_history = chat_history[-MAX_HISTORY:]

    history_text = ""

    if conversation_summary:
        history_text += f"Conversation Summary:\n{conversation_summary}\n\n"

    if recent_history:
        history_text += "Recent Conversation:\n"

        for item in recent_history:
            history_text += f"User: {item['question']}\n"
            history_text += f"Assistant: {item['answer']}\n\n"
    else:
        history_text += "No previous conversation."

    return history_text

def create_conversational_rag_chain(
    question,
    active_index_path=None,
    chat_messages=None,
    file_names=None,
):
    trace_metadata = {
        "active_index_path": active_index_path,
        "has_chat_messages": chat_messages is not None,
        "file_names": file_names,
    }
    with RAGTrace(
        "conversational_rag",
        inputs={"query": question},
        metadata=trace_metadata,
    ):
        return _create_conversational_rag_chain(
            question,
            active_index_path,
            chat_messages,
            file_names,
        )


def _create_conversational_rag_chain(
    question,
    active_index_path=None,
    chat_messages=None,
    file_names=None,
):
    started_at = time.perf_counter()
    record_trace_step(
        "query",
        outputs={"query": question},
        metadata={
            "active_index_path": active_index_path,
            "file_names": file_names,
        },
    )

    if not active_index_path or not os.path.isdir(active_index_path):
        answer = """
Answer:
No active index found. Upload and index PDFs first.
"""
        record_trace_step(
            "llm_response",
            inputs={"query": question},
            outputs={
                "response": answer.strip(),
                "reason": "missing_active_index",
            },
            latency_seconds=time.perf_counter() - started_at,
        )
        return answer

    history = format_chat_history(chat_messages)

    rewrite_started_at = time.perf_counter()
    rewritten_question = rewrite_query(history, question)
    print(f"\nRewritten Query: {rewritten_question}")
    record_trace_step(
        "rewritten_query",
        inputs={
            "query": question,
            "chat_history": history,
        },
        outputs={"rewritten_query": rewritten_question},
        latency_seconds=time.perf_counter() - rewrite_started_at,
    )

    retrieval_started_at = time.perf_counter()
    retrieved_docs = hybrid_retrieve_documents(
        rewritten_question,
        active_index_path,
        file_names=file_names,
    )
    record_trace_step(
        "retrieved_context",
        inputs={"rewritten_query": rewritten_question},
        outputs={
            "chunk_count": len(retrieved_docs),
            "chunks": serialize_documents(retrieved_docs),
        },
        latency_seconds=time.perf_counter() - retrieval_started_at,
    )

    context = format_context(retrieved_docs)
    sources = format_sources(retrieved_docs)
    weakness_reason = context_weakness_reason(
        question,
        rewritten_question,
        retrieved_docs,
        context,
    )

    record_trace_step(
        "retrieval_diagnostics",
        inputs={
            "user_query": question,
            "rewritten_query": rewritten_question,
            "file_names": file_names,
        },
        outputs={
            "retrieved_filenames_pages": retrieval_page_summary(retrieved_docs),
            "reranked_chunks": serialize_documents(retrieved_docs),
            "final_context_length": len(context),
            "sources": sources,
            "weakness_reason": weakness_reason,
        },
        run_type="retriever",
        latency_seconds=time.perf_counter() - retrieval_started_at,
    )

    if weakness_reason:
        final_answer = f"""
Answer:
{LOW_CONFIDENCE_ANSWER}
"""
        record_trace_step(
            "fallback_answer",
            inputs={
                "query": question,
                "rewritten_query": rewritten_question,
            },
            outputs={
                "answer": final_answer.strip(),
                "reason": weakness_reason,
                "final_context_length": len(context),
            },
            latency_seconds=time.perf_counter() - started_at,
        )
        return final_answer

    prompt = load_prompt()

    final_prompt = prompt.format(
        context=context,
        question=question
    )
    record_trace_step(
        "final_prompt",
        inputs={
            "query": question,
            "rewritten_query": rewritten_question,
            "retrieved_chunk_count": len(retrieved_docs),
        },
        outputs={
            "prompt": truncate_prompt(final_prompt),
            "prompt_length": len(final_prompt),
            "sources": sources,
        },
        run_type="prompt",
    )

    llm = create_llm()

    llm_started_at = time.perf_counter()
    response = llm.invoke(final_prompt)
    response_text = response.content
    usage_and_cost = extract_usage_and_cost(
        response=response,
        prompt=final_prompt,
        response_text=response_text,
    )
    record_trace_step(
        "llm_response",
        inputs={"prompt": truncate_prompt(final_prompt)},
        outputs={
            "response": response_text,
            "latency_seconds": round(time.perf_counter() - llm_started_at, 3),
            **usage_and_cost,
        },
        run_type="llm",
        latency_seconds=time.perf_counter() - llm_started_at,
    )

    if is_fallback_response(response_text):
        final_answer = f"""
Answer:
{response_text}
"""
    else:
        final_answer = f"""
Answer:
{response_text}

Sources: {sources}
"""

    if chat_messages is None:
        chat_history.append({
            "question": question,
            "answer": response_text
        })
        update_conversation_summary()

    record_trace_step(
        "final_answer",
        inputs={"query": question},
        outputs={
            "answer": final_answer.strip(),
            "sources": sources,
        },
        latency_seconds=time.perf_counter() - started_at,
    )

    return final_answer


if __name__ == "__main__":
    print("\nConversational RAG started. Type 'exit' to stop.\n")

    while True:
        user_question = input("You: ")

        if user_question.lower() == "exit":
            break

        answer = create_conversational_rag_chain(user_question)

        print("\nAssistant:")
        print(answer)
