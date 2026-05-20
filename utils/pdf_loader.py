from langchain_community.document_loaders import PyPDFLoader


def load_pdf(file_path: str):
    """
    Load PDF and return document objects.
    """

    loader = PyPDFLoader(file_path)

    documents = loader.load()

    return documents


if __name__ == "__main__":

    pdf_path = "data/sample.pdf"

    docs = load_pdf(pdf_path)

    print(f"\nTotal Pages Loaded: {len(docs)}\n")

    print("FIRST PAGE CONTENT:\n")
    print(docs[0].page_content[:1000])

    print("\nMETADATA:\n")
    print(docs[0].metadata)

    print(type(docs))
    print(type(docs[0]))
    print(docs[0])