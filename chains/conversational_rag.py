from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate

from chains.retriever import retrieve_documents


load_dotenv()


chat_history = []
conversation_summary = ""
MAX_HISTORY = 3

def load_prompt():
    with open("prompts/rag_prompt.txt", "r") as file:
        template = file.read()

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )

    return prompt


def format_context(documents):
    return "\n\n".join([doc.page_content for doc in documents])


def format_sources(documents):
    sources = []

    for doc in documents:
        source = doc.metadata.get("source")
        page = doc.metadata.get("page_label")
        sources.append(f"{source} - Page {page}")

    return "\n".join(sorted(set(sources)))

def update_conversation_summary():
    global conversation_summary

    if len(chat_history) <= MAX_HISTORY:
        return

    older_history = chat_history[:-MAX_HISTORY]

    history_text = ""

    for item in older_history:
        history_text += f"User: {item['question']}\n"
        history_text += f"Assistant: {item['answer']}\n\n"

    llm = ChatOllama(
        model="phi3",
        temperature=0
    )

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

def format_chat_history():
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

def create_conversational_rag_chain(question):
    history = format_chat_history()

    enhanced_question = f"""
Previous Conversation:
{history}

Current Question:
{question}
"""

    retrieved_docs = retrieve_documents(enhanced_question)

    context = format_context(retrieved_docs)
    sources = format_sources(retrieved_docs)

    prompt = load_prompt()

    final_prompt = prompt.format(
        context=context,
        question=enhanced_question
    )

    llm = ChatOllama(
        model="phi3",
        temperature=0
    )

    response = llm.invoke(final_prompt)

    if "I could not find this information" in response.content:
        final_answer = f"""
Answer:
{response.content}
"""
    else:
        final_answer = f"""
Answer:
{response.content}

Sources:
{sources}
"""

    chat_history.append({
        "question": question,
        "answer": response.content
    })

    update_conversation_summary()

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