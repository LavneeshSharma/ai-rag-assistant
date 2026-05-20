from langchain_huggingface import HuggingFaceEmbeddings

from utils.pdf_loader import load_pdf
from utils.chunker import chunk_documents


def create_embeddings():
    """
    Create embedding model.
    """

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embedding_model


if __name__ == "__main__":

    pdf_path = "data/sample.pdf"

    docs = load_pdf(pdf_path)

    chunks = chunk_documents(docs)

    embedding_model = create_embeddings()

    sample_text = chunks[0].page_content

    embedding_vector = embedding_model.embed_query(sample_text)

    print(f"\nChunk Text:\n{sample_text[:300]}")

    print(f"\nEmbedding Vector Length: {len(embedding_vector)}")

    print(f"\nFirst 10 Values:\n{embedding_vector[:10]}")