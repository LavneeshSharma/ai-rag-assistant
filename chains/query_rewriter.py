from langchain_ollama import ChatOllama


def rewrite_query(chat_history, question):

    llm = ChatOllama(
        model="phi3",
        temperature=0
    )

    rewrite_prompt = f"""
You are a query rewriting assistant.

Your task:
Convert the user's latest question into a clear standalone question.

Use conversation history only if needed.

Conversation:
{chat_history}

User Question:
{question}

Standalone Question:
"""

    response = llm.invoke(rewrite_prompt)

    return response.content.strip()


if __name__ == "__main__":

    history = """
User: What is Data Analytics?
Assistant: Data Analytics is the process of analyzing data...
"""

    question = "Explain it in more detail"

    rewritten = rewrite_query(history, question)

    print("\nRewritten Query:\n")
    print(rewritten)