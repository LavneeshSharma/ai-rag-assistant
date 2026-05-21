from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate

from chains.retriever import retrieve_documents


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

    context = "\n\n".join(
        [doc.page_content for doc in documents]
    )

    return context


def format_sources(documents):

    sources = []

    for doc in documents:

        source = doc.metadata.get("source")
        page = doc.metadata.get("page_label")

        sources.append(f"{source} - Page {page}")

    unique_sources = list(set(sources))

    return "\n".join(unique_sources)


def create_rag_chain(question):
    retrieved_docs = retrieve_documents(question)

    context = format_context(retrieved_docs)
    sources = format_sources(retrieved_docs)

    prompt = load_prompt()

    final_prompt = prompt.format(
        context=context,
        question=question
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

    return final_answer


if __name__ == "__main__":

    user_question = "Which project uses clustering algorithms?"

    answer = create_rag_chain(user_question)

    print(answer)