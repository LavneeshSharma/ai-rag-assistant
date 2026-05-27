from uuid import uuid4

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import (
    PARENT_CHUNK_SIZE,
    PARENT_CHUNK_OVERLAP,
    CHILD_CHUNK_SIZE,
    CHILD_CHUNK_OVERLAP
)


def create_parent_child_chunks(documents):
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=PARENT_CHUNK_OVERLAP
    )

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP
    )

    parent_chunks = parent_splitter.split_documents(documents)

    child_chunks = []
    parent_store = {}

    for parent_index, parent_doc in enumerate(parent_chunks):
        parent_id = str(uuid4())

        parent_doc.metadata["parent_id"] = parent_id
        parent_doc.metadata["parent_index"] = parent_index

        parent_store[parent_id] = parent_doc

        children = child_splitter.split_documents([parent_doc])

        for child_index, child_doc in enumerate(children):
            child_doc.metadata["parent_id"] = parent_id
            child_doc.metadata["parent_index"] = parent_index
            child_doc.metadata["child_index"] = child_index

            child_chunks.append(child_doc)

    return parent_store, child_chunks


if __name__ == "__main__":
    from utils.pdf_loader import load_all_pdfs

    docs = load_all_pdfs("data")

    parent_store, child_chunks = create_parent_child_chunks(docs)

    print(f"\nTotal Parent Chunks: {len(parent_store)}")
    print(f"Total Child Chunks: {len(child_chunks)}")

    print("\nSample Child Metadata:")
    print(child_chunks[0].metadata)

    print("\nSample Parent Content:")
    first_parent_id = child_chunks[0].metadata["parent_id"]
    print(parent_store[first_parent_id].page_content[:1000])