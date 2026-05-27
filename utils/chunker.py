from uuid import uuid4
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(documents):
    """
    Split documents into chunks and add metadata.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = text_splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):

        chunk.metadata["chunk_id"] = str(uuid4())

        chunk.metadata["document_id"] = (
            f"{chunk.metadata.get('file_name')}"
            f"_page_{chunk.metadata.get('page')}"
        )

        chunk.metadata["chunk_index"] = i

    return chunks


if __name__ == "__main__":

    from utils.pdf_loader import load_all_pdfs

    docs = load_all_pdfs("data")

    chunks = chunk_documents(docs)

    print(f"\nTotal Chunks Created: {len(chunks)}\n")

    print("FIRST CHUNK:\n")
    print(chunks[0].page_content)

    print("\nCHUNK METADATA:\n")
    print(chunks[0].metadata)