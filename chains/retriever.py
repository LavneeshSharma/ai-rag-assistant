from langchain_chroma import Chroma

from utils.embeddings import create_embeddings


def load_vector_store():

    embedding_model = create_embeddings()

    vector_store = Chroma(
        persist_directory="vector_store/db",
        embedding_function=embedding_model
    )

    return vector_store


def retrieve_documents(query):

    vector_store = load_vector_store()

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )

    results = retriever.invoke(query)

    return results


if __name__ == "__main__":

    user_query = "Which projects involve machine learning?"

    retrieved_docs = retrieve_documents(user_query)

    print(f"\nTotal Retrieved Chunks: {len(retrieved_docs)}\n")

    for i, doc in enumerate(retrieved_docs):

        print(f"\n--- RESULT {i+1} ---\n")

        print(doc.page_content[:500])

        print("\nMetadata:")
        print(doc.metadata)