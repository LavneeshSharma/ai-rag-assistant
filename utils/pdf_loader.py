import re
import glob
import os

from langchain_community.document_loaders import PyPDFLoader

def clean_text(text):

    # Normalize newlines
    text = re.sub(r"\n+", "\n", text)

    # Fix spaced letters only:
    # D a t a -> Data
    text = re.sub(
        r'\b(?:[A-Za-z]\s){2,}[A-Za-z]\b',
        lambda x: x.group(0).replace(" ", ""),
        text
    )

    # Normalize spaces
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def load_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    for doc in documents:
        doc.page_content = clean_text(doc.page_content)
        doc.metadata["source"] = file_path
        doc.metadata["file_name"] = os.path.basename(file_path)

    return documents


def load_all_pdfs(data_dir: str = "data"):
    pdf_paths = glob.glob(os.path.join(data_dir, "*.pdf"))

    all_documents = []

    for pdf_path in pdf_paths:
        documents = load_pdf(pdf_path)
        all_documents.extend(documents)

    return all_documents


if __name__ == "__main__":
    docs = load_all_pdfs("data")

    print(f"\nTotal Pages Loaded From All PDFs: {len(docs)}\n")

    if docs:
        print("FIRST PAGE CONTENT:\n")
        print(docs[0].page_content[:1000])

        print("\nMETADATA:\n")
        print(docs[0].metadata)
    else:
        print("No PDF files found in data/ folder.")