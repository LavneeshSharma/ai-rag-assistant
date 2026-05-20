from langchain.text_splitter import RecursiveCharacterTextSplitter

from utils.pdf_loader import load_pdf


def chunk_documents(documents):
    """
    Split documents into smaller chunks.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(documents)

    return chunks


if __name__ == "__main__":

    pdf_path = "data/sample.pdf"

    docs = load_pdf(pdf_path)

    chunks = chunk_documents(docs)

    print(f"\nTotal Chunks Created: {len(chunks)}\n")

    print("FIRST CHUNK:\n")
    print(chunks[0].page_content)

    print("\nCHUNK METADATA:\n")
    print(chunks[0].metadata)