from langchain_groq import ChatGroq

from config.settings import MODEL_NAME


def create_llm():
    return ChatGroq(
        model=MODEL_NAME,
        temperature=0
    )