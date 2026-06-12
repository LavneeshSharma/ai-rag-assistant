import pickle

from langchain_chroma import Chroma

from utils.pdf_loader import load_all_pdfs
from utils.parent_child_chunker import create_parent_child_chunks
from utils.embeddings import create_embeddings


def create_parent_child_vector_store():
    documents = load_all_pdfs("data")

    parent_store, child_chunks = create_parent_child_chunks(documents)

    embedding_model = create_embeddings()

    Chroma.from_documents(
        documents=child_chunks,
        embedding=embedding_model,
        persist_directory="vector_store/parent_child_db"
    )

    with open("vector_store/parent_store.pkl", "wb") as file:
        pickle.dump(parent_store, file)

    print("\nParent-child vector DB created successfully!")
    print(f"Total Parent Chunks: {len(parent_store)}")
    print(f"Total Child Chunks Stored: {len(child_chunks)}")


if __name__ == "__main__":
    create_parent_child_vector_store()