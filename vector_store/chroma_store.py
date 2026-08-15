import gc
import json
import os
import shutil
import stat
import time
import uuid
from datetime import datetime

from langchain_chroma import Chroma

from utils.pdf_loader import load_all_pdfs, load_pdf
from utils.chunker import chunk_documents
from utils.embeddings import create_embeddings
from config.settings import (
    ACTIVE_INDEX_FILE as REL_ACTIVE_INDEX_FILE,
    DATA_DIR as REL_DATA_DIR,
    VECTOR_INDEXES_DIR as REL_VECTOR_INDEXES_DIR,
)

ACTIVE_INDEX_FILE = os.path.abspath(REL_ACTIVE_INDEX_FILE)
DATA_DIR = os.path.abspath(REL_DATA_DIR)
VECTOR_INDEXES_DIR = os.path.abspath(REL_VECTOR_INDEXES_DIR)


def _ensure_writable_dir(path):
    os.makedirs(path, exist_ok=True)
    os.chmod(path, os.stat(path).st_mode | stat.S_IWUSR | stat.S_IXUSR)


def get_active_index_path():
    if not os.path.exists(ACTIVE_INDEX_FILE):
        return None
    try:
        with open(ACTIVE_INDEX_FILE, "r") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    active_path = data.get("active_index_path")
    if not active_path:
        return None
    return os.path.abspath(active_path)


def _save_active_index_path(index_path):
    _ensure_writable_dir(os.path.dirname(ACTIVE_INDEX_FILE))
    with open(ACTIVE_INDEX_FILE, "w") as f:
        json.dump({"active_index_path": index_path}, f, indent=2)


def reset_vector_store():
    """Clear active Chroma pointer without deleting active DB folders."""
    gc.collect()
    time.sleep(0.3)
    _save_active_index_path(None)


def cleanup_orphaned_indexes(keep_paths):
    """Delete index folders under VECTOR_INDEXES_DIR not referenced in keep_paths.

    Callers must pass every index path still in use (e.g. every chat's active
    index across all users) so we never delete one that's still referenced.
    """
    if not os.path.isdir(VECTOR_INDEXES_DIR):
        return []

    keep_abs = {os.path.abspath(path) for path in keep_paths if path}
    removed = []
    for name in os.listdir(VECTOR_INDEXES_DIR):
        folder_path = os.path.join(VECTOR_INDEXES_DIR, name)
        if not os.path.isdir(folder_path) or os.path.abspath(folder_path) in keep_abs:
            continue
        try:
            shutil.rmtree(folder_path)
            removed.append(folder_path)
        except OSError:
            pass
    return removed


def create_vector_store(reset_existing=False, pdf_paths=None):
    _ensure_writable_dir(DATA_DIR)
    _ensure_writable_dir(VECTOR_INDEXES_DIR)

    if pdf_paths is None:
        documents = load_all_pdfs(DATA_DIR)
    else:
        documents = []
        for pdf_path in pdf_paths:
            if os.path.exists(pdf_path):
                documents.extend(load_pdf(pdf_path))

    chunks = chunk_documents(documents)

    if not chunks:
        _save_active_index_path(None)
        print("\nNo PDF chunks found. Active Chroma index cleared.")
        return None

    embedding_model = create_embeddings()
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    new_index_path = os.path.join(
        VECTOR_INDEXES_DIR,
        f"index_{timestamp}_{uuid.uuid4().hex[:8]}",
    )
    _ensure_writable_dir(new_index_path)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=new_index_path
    )
    _save_active_index_path(new_index_path)

    print("\nChroma Vector DB created successfully!")
    print(f"\nActive Index: {new_index_path}")
    print(f"\nTotal Documents Stored: {len(chunks)}")

    return vector_store


if __name__ == "__main__":
    create_vector_store()
