from langchain_ollama import ChatOllama

from config.settings import MODEL_NAME


def rewrite_query(chat_history, question):
    vague_words = ["it", "this", "that", "more", "elaborate", "detail"]

    question_lower = question.lower()

    has_vague_word = any(word in question_lower for word in vague_words)

    if not has_vague_word:
        return question

    llm = ChatOllama(
        model=MODEL_NAME,
        temperature=0
    )

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

    response = llm.invoke(prompt)

    return response.content.strip()

if __name__ == "__main__":
    history = """
User: List the project titles mentioned in the PDF.
Assistant: The PDF mentions EDA on Retail Sales Data, Customer Segmentation, Sentiment Analysis, Predicting House Prices with Linear Regression, Wine Quality Prediction, Fraud Detection, and others.
"""

    question = "what is car prediction project about"

    rewritten_query = rewrite_query(history, question)

    print("\nRewritten Query:")
    print(rewritten_query)