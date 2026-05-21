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
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 12,
            "lambda_mult": 0.5
        }
    )

    results = retriever.invoke(query)

    return results

def retrieve_documents_with_scores(query, k=4):
    vector_store = load_vector_store()

    results = vector_store.similarity_search_with_score(
        query=query,
        k=k
    )

    return results

if __name__ == "__main__":
    user_query = "Which project uses clustering algorithms?"

    results = retrieve_documents_with_scores(user_query)

    print(f"\nTotal Retrieved Chunks: {len(results)}\n")

    for i, (doc, score) in enumerate(results):
        print(f"\n--- RESULT {i + 1} ---")
        print(f"Similarity Score: {score}")
        print(f"Page: {doc.metadata.get('page_label')}")
        print(f"Source: {doc.metadata.get('source')}\n")
        print(doc.page_content[:500])

