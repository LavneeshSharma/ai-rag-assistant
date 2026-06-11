
import time
import re

from utils.llm import create_llm
from config.settings import MODEL_NAME
from utils.tracing import (
    extract_usage_and_cost,
    record_trace_step,
    truncate_prompt,
)


def rewrite_query(chat_history, question):
    vague_words = [
        "it", "this", "that", "these", "those", "more", "elaborate",
        "detail", "mentioned", "above", "same", "first", "second",
    ]

    question_lower = question.lower()
    word_count = len(question.split())

    has_vague_word = any(
        re.search(rf"\b{re.escape(word)}\b", question_lower)
        for word in vague_words
    )
    likely_needs_rewrite = has_vague_word or word_count <= 6

    if not likely_needs_rewrite:
        record_trace_step(
            "query_rewrite",
            inputs={"query": question},
            outputs={
                "rewritten_query": question,
                "rewrite_needed": False,
            },
        )
        return question
    llm = create_llm()

    prompt = f"""
You are a query rewriting assistant for a PDF RAG system.

Your task:
Rewrite the user's latest question into a SHORT standalone retrieval query.

Rules:
1. Do not answer the question.
2. Do not add outside facts.
3. Do not invent new topics.
4. Use conversation history only for vague references like it, this, that, more, elaborate, detail.
5. If the question is already clear, return it unchanged.
6. If the user asks "explain it more", rewrite as:
   "Explain <previous topic> in more detail"
7. Keep the output between 5 and 15 words when possible.
8. Return only the rewritten query.

Conversation History:
{chat_history}

User Question:
{question}

Standalone Retrieval Query:
"""

    started_at = time.perf_counter()
    response = llm.invoke(prompt)
    rewritten_query = response.content.strip()
    usage_and_cost = extract_usage_and_cost(
        response=response,
        prompt=prompt,
        response_text=rewritten_query,
    )
    record_trace_step(
        "query_rewrite",
        inputs={
            "query": question,
            "chat_history": chat_history,
            "prompt": truncate_prompt(prompt),
        },
        outputs={
            "rewritten_query": rewritten_query,
            "rewrite_needed": True,
            **usage_and_cost,
        },
        run_type="llm",
        latency_seconds=time.perf_counter() - started_at,
    )

    return rewritten_query

if __name__ == "__main__":
    history = """
User: List the project titles mentioned in the PDF.
Assistant: The PDF mentions EDA on Retail Sales Data, Customer Segmentation, Sentiment Analysis, Predicting House Prices with Linear Regression, Wine Quality Prediction, Fraud Detection, and others.
"""

    question = "what is car prediction project about"

    rewritten_query = rewrite_query(history, question)

    print("\nRewritten Query:")
    print(rewritten_query)
