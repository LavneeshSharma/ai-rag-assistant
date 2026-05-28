import os

from dotenv import load_dotenv

from utils.llm import create_llm
from langchain_core.prompts import PromptTemplate
from chains.hybrid_retriever import hybrid_retrieve_documents
from config.settings import MODEL_NAME


load_dotenv()


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
    sources = {}

    for doc in documents:
        source = doc.metadata.get("file_name") or os.path.basename(
            doc.metadata.get("source", "")
        )
        page = doc.metadata.get("page_label")
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


def create_rag_chain(question):
    retrieved_docs = hybrid_retrieve_documents(question)

    context = format_context(retrieved_docs)
    sources = format_sources(retrieved_docs)

    prompt = load_prompt()

    final_prompt = prompt.format(
        context=context,
        question=question
    )

    llm = create_llm()

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

Sources: {sources}
"""

    return final_answer


def stream_rag_chain(question):
    retrieved_docs = hybrid_retrieve_documents(question)

    context = format_context(retrieved_docs)
    sources = format_sources(retrieved_docs)

    prompt = load_prompt()

    final_prompt = prompt.format(
        context=context,
        question=question
    )

    llm = ChatOllama(
        model=MODEL_NAME,
        temperature=0
    )

    print("\nAnswer:\n")

    full_response = ""

    for chunk in llm.stream(final_prompt):
        print(chunk.content, end="", flush=True)
        full_response += chunk.content

    if "I could not find this information" not in full_response:
        print("\n\nSources:")
        print(sources)

    print()


if __name__ == "__main__":
    user_question = "List the project titles mentioned in the PDF."
    stream_rag_chain(user_question)
