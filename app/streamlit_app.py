import os
import sys
import gc
import hashlib
import re
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from chains.conversational_rag import create_conversational_rag_chain
from vector_store.chroma_store import create_vector_store, reset_vector_store

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
from config.settings import DATA_DIR as REL_DATA_DIR

# Ensure backend code that uses relative paths (config.settings) resolves correctly.
os.chdir(BASE_DIR)

DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, REL_DATA_DIR))

st.set_page_config(
    page_title="RAG Assistant",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items=None,
)


# ── Authentication (OAuth hooks for later) ──────────────────────────

def get_current_user() -> Optional[Dict[str, Any]]:
    """Return authenticated user dict or None."""
    return st.session_state.get("authenticated_user")


def login_with_google() -> None:
    """
    Placeholder for Google OAuth.
    Wire streamlit-oauth / authlib here when credentials are configured.
    """
    st.session_state.oauth_notice = "Google OAuth not configured yet"
    st.rerun()


def logout_user() -> None:
    """Clear authenticated session."""
    st.session_state.authenticated_user = None
    st.session_state.pop("oauth_notice", None)


def is_authenticated() -> bool:
    return get_current_user() is not None


# ── Session state ───────────────────────────────────────────────────

def _new_chat_record(title: str = "New chat") -> Dict[str, Any]:
    chat_id = uuid.uuid4().hex[:12]
    return {
        "id": chat_id,
        "title": title,
        "messages": [],
        "created_at": datetime.utcnow().isoformat(),
    }


def _chat_title_from_question(question: str) -> str:
    text = " ".join(question.strip().split())
    lower = text.lower()

    if "pdfloader" in lower or "pdf loader" in lower:
        return "PDF Loader Summary"
    if "date" in lower or "deadline" in lower or "timeline" in lower:
        return "Important Dates"
    if "analysis" in lower or "analyze" in lower:
        return "Project Analysis"

    words = re.findall(r"[A-Za-z0-9]+", text)
    stop_words = {
        "a", "an", "are", "about", "can", "could", "detailed", "explain",
        "for", "give", "is", "me", "of", "on", "please", "tell", "the",
        "this", "to", "what", "whats", "which",
    }
    keywords = [word for word in words if word.lower() not in stop_words]
    if not keywords:
        return "New Chat"

    title_words = keywords[:4]
    if len(title_words) == 1:
        title_words.append("Summary")

    return " ".join(
        word.upper() if word.isupper() else word.title() for word in title_words
    )


def _persist_current_chat() -> None:
    chat_id = st.session_state.current_chat_id
    first_user = next(
        (m["content"] for m in st.session_state.messages if m["role"] == "user"),
        None,
    )

    if chat_id and chat_id in st.session_state.chats:
        st.session_state.chats[chat_id]["messages"] = list(st.session_state.messages)
        if first_user and st.session_state.chats[chat_id]["title"] == "New chat":
            st.session_state.chats[chat_id]["title"] = _chat_title_from_question(
                first_user
            )
        return

    if not first_user:
        return

    chat = _new_chat_record(_chat_title_from_question(first_user))
    st.session_state.chats[chat["id"]] = chat
    st.session_state.current_chat_id = chat["id"]
    chat_id = chat["id"]
    st.session_state.chats[chat_id]["messages"] = list(st.session_state.messages)


def init_session_state() -> None:
    defaults = {
        "messages": [],
        "chats": {},
        "current_chat_id": None,
        "uploaded_pdfs": [],
        "show_uploader": False,
        "authenticated_user": None,
        "selected_pdf_ids": [],
        "uploader_key": 0,
        "chat_text_key": 0,
        "is_indexing": False,
        "needs_reindex": False,
        "oauth_notice": None,
        "chat_search": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    os.makedirs(DATA_DIR, exist_ok=True)

    if (
        st.session_state.current_chat_id is not None
        and st.session_state.current_chat_id not in st.session_state.chats
    ):
        st.session_state.current_chat_id = None
        st.session_state.messages = []

    _sync_pdfs_from_disk()


def _sync_pdfs_from_disk() -> None:
    if not os.path.isdir(DATA_DIR):
        return
    on_disk = sorted(f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf"))
    for name in on_disk:
        if name not in st.session_state.uploaded_pdfs:
            st.session_state.uploaded_pdfs.append(name)


def create_new_chat() -> None:
    _persist_current_chat()
    st.session_state.current_chat_id = None
    st.session_state.messages = []


def switch_chat(chat_id: str) -> None:
    if chat_id == st.session_state.current_chat_id:
        return
    if chat_id not in st.session_state.chats:
        return
    _persist_current_chat()
    st.session_state.current_chat_id = chat_id
    st.session_state.messages = list(st.session_state.chats[chat_id]["messages"])


def clear_current_chat() -> None:
    st.session_state.messages = []
    _persist_current_chat()


def save_uploaded_files(uploaded_files: Optional[List[Any]]) -> bool:
    if not uploaded_files:
        return False

    new_files = False
    for uf in uploaded_files:
        name = uf.name
        if not name.lower().endswith(".pdf"):
            continue
        save_path = os.path.join(DATA_DIR, name)
        # Prevent duplicates across both session and disk state.
        if name in st.session_state.uploaded_pdfs or os.path.exists(save_path):
            continue
        with open(save_path, "wb") as f:
            f.write(uf.getbuffer())
        st.session_state.uploaded_pdfs.append(name)
        new_files = True

    if new_files:
       st.session_state.needs_reindex = True
       st.session_state.uploader_key += 1

    return new_files



def remove_pdf(filename: str) -> None:
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        os.remove(filepath)
    if filename in st.session_state.uploaded_pdfs:
        st.session_state.uploaded_pdfs.remove(filename)
    if filename in st.session_state.selected_pdf_ids:
        st.session_state.selected_pdf_ids.remove(filename)
    _clear_vector_refs()
    st.session_state.needs_reindex = True


def _clear_vector_store_dir() -> None:
    """Remove persisted Chroma DB to prevent stale retrieval results."""
    if st.session_state.is_indexing:
        return
    reset_vector_store()


def _clear_vector_refs() -> None:
    st.session_state.pop("vectorstore", None)
    st.session_state.pop("retriever", None)
    st.session_state.pop("rag_chain", None)


def safe_reindex() -> bool:
    if st.session_state.is_indexing:
        return False

    st.session_state.is_indexing = True
    try:
        _clear_vector_refs()
        gc.collect()
        time.sleep(0.3)
        with st.spinner("Documents changed. Re-indexing..."):
            if st.session_state.uploaded_pdfs:
                create_vector_store()
            else:
                reset_vector_store()
        st.session_state.needs_reindex = False
        return True
    finally:
        st.session_state.is_indexing = False


def reindex_documents() -> None:
    """Mark documents for re-indexing."""
    if not st.session_state.uploaded_pdfs:
        st.info("No PDFs found in `data/` to index.")
        return
    st.session_state.needs_reindex = True
    


def clear_all_pdfs() -> None:
    """Delete all PDFs + vector index; reset upload-related session state."""
    if os.path.isdir(DATA_DIR):
        for f in os.listdir(DATA_DIR):
            if f.lower().endswith(".pdf"):
                try:
                    os.remove(os.path.join(DATA_DIR, f))
                except OSError:
                    pass
    _clear_vector_store_dir()

    st.session_state.uploaded_pdfs = []
    st.session_state.selected_pdf_ids = []
    st.session_state.show_uploader = False
    st.session_state.needs_reindex = False
    _clear_vector_refs()
    st.session_state.uploader_key += 1


# ── CSS ─────────────────────────────────────────────────────────────

CUSTOM_CSS = """
<style>
    :root {
        --bg-main: #1f1f1f;
        --bg-sidebar: #171717;
        --bg-input: #2b2b2b;
        --bg-hover: #242424;
        --border-color: #333333;
        --text-primary: #ececec;
        --text-secondary: #b4b4b4;
        --text-muted: #8b8b8b;
        --danger: #ef4444;
    }

    .stApp { background: var(--bg-main); }
    [data-testid="collapsedControl"] { display: none !important; }

    section[data-testid="stSidebar"] {
        background: var(--bg-sidebar) !important;
        min-width: 260px !important;
        max-width: 260px !important;
        border-right: none !important;
    }
    section[data-testid="stSidebar"] > div { padding: 0 !important; }
    section[data-testid="stSidebar"] .block-container {
        padding: 14px 10px 10px !important;
        display: flex;
        flex-direction: column;
        min-height: 100vh;
    }
    section[data-testid="stSidebar"] .element-container {
        margin-bottom: 4px !important;
    }
    section[data-testid="stSidebar"] .stMarkdown {
        margin-bottom: 0 !important;
    }

    section[data-testid="stSidebar"] .stButton > button {
        width: 100%;
        text-align: left;
        background: transparent !important;
        border: none !important;
        color: #e8e8e8 !important;
        padding: 7px 9px !important;
        border-radius: 8px !important;
        font-size: 13px !important;
        font-weight: 400 !important;
        box-shadow: none !important;
        min-height: 32px !important;
        line-height: 1.15 !important;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: var(--bg-hover) !important;
        color: #ffffff !important;
    }

    .sb-header {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 2px 8px 10px;
        font-size: 15px;
        font-weight: 600;
        color: #ffffff;
    }
    .sb-logo {
        width: 20px;
        height: 20px;
        border-radius: 6px;
        background: #2f2f2f;
        color: #f4f4f5;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 12px;
    }
    .sb-section-title {
        padding: 14px 8px 6px;
        font-size: 11px;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .sb-spacer { flex: 1; min-height: 12px; }
    .sb-profile-box {
        padding: 8px;
        margin-top: auto;
    }
    .sb-user-name { font-size: 13px; font-weight: 500; color: #f2f2f2; line-height: 1.2; }
    .sb-user-email { font-size: 12px; color: #a6a6a6; line-height: 1.2; }
    .sb-avatar {
        width: 28px; height: 28px; border-radius: 50%;
        background: #e8605f; color: #fff;
        display: flex; align-items: center; justify-content: center;
        font-size: 12px; font-weight: 600;
    }

    .main .block-container {
        max-width: 760px;
        margin: 0 auto;
        padding: 72px 1rem 150px 1rem;
    }

    .empty-state { text-align: center; padding: 18vh 1rem 0; }
    .empty-title { font-size: 26px; font-weight: 600; color: var(--text-primary); margin-bottom: 8px; }
    .empty-sub { font-size: 14px; color: var(--text-secondary); }

    .stChatMessage { padding: 4px 0 !important; }
    [data-testid="stChatMessageContent"] {
        background: transparent !important;
        border: none !important;
        color: var(--text-primary) !important;
        font-size: 15px !important;
        line-height: 1.7 !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
    [data-testid="stChatMessageContent"] {
        background: var(--bg-input) !important;
        border-radius: 16px !important;
        padding: 10px 14px !important;
        max-width: 70%;
        margin-left: auto;
    }

    .st-key-input_panel {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: var(--bg-main);
        padding: 4px 16px 16px;
        z-index: 100;
        max-width: 760px;
        margin-left: auto;
        margin-right: auto;
    }
    .st-key-pdf_chips_area {
        margin-bottom: 4px;
    }
    .st-key-pdf_chips_area [data-testid="stHorizontalBlock"] {
        width: fit-content !important;
        max-width: 100%;
        background: #292929;
        border: none;
        border-radius: 16px;
        padding: 1px 4px 1px 8px;
        align-items: center !important;
        gap: 0 !important;
    }
    .st-key-pdf_chips_area .stButton > button {
        min-height: 22px !important;
        height: 22px !important;
        width: 22px !important;
        padding: 0 !important;
        font-size: 13px !important;
        line-height: 1 !important;
        background: transparent !important;
        border: none !important;
        color: var(--text-muted) !important;
        border-radius: 50% !important;
    }
    .st-key-pdf_chips_area .stButton > button:hover {
        background: rgba(255,255,255,0.10) !important;
        color: var(--danger) !important;
    }
    .st-key-input_bar {
        background: var(--bg-input);
        border-radius: 24px;
        padding: 4px 8px;
    }
    .st-key-input_bar [data-testid="stHorizontalBlock"] {
        align-items: center !important;
        gap: 0.25rem !important;
    }
    .st-key-input_bar .element-container,
    .st-key-input_bar .stTextInput,
    .st-key-input_bar .stButton,
    .st-key-input_bar [data-testid="stFileUploader"] {
        margin: 0 !important;
    }
    .st-key-input_bar [data-testid="stFileUploader"] section {
        padding: 0 !important;
    }
    .st-key-input_bar [data-testid="stFileUploader"] label,
    .st-key-input_bar [data-testid="stFileUploaderDropzoneInstructions"],
    .st-key-input_bar [data-testid="stFileUploaderDropzone"] svg,
    .st-key-input_bar [data-testid="stFileUploaderDropzone"] small,
    .st-key-input_bar [data-testid="stFileUploaderDropzone"] p {
        display: none !important;
    }
    .st-key-input_bar [data-testid="stFileUploaderDropzone"] {
        width: 32px !important;
        height: 32px !important;
        min-height: 32px !important;
        padding: 0 !important;
        background: transparent !important;
        border: none !important;
        border-radius: 50% !important;
    }
    .st-key-input_bar [data-testid="stFileUploaderDropzone"] button,
    .st-key-input_bar .stButton > button {
        width: 32px !important;
        height: 32px !important;
        min-height: 32px !important;
        padding: 0 !important;
        border: none !important;
        border-radius: 50% !important;
        background: transparent !important;
        box-shadow: none !important;
        color: var(--text-secondary) !important;
    }
    .st-key-input_bar [data-testid="stFileUploaderDropzone"] button {
        color: transparent !important;
        font-size: 0 !important;
    }
    .st-key-input_bar [data-testid="stFileUploaderDropzone"] button::before {
        content: "+";
        color: var(--text-secondary);
        font-size: 22px;
        line-height: 1;
        font-weight: 300;
    }
    .st-key-input_bar [data-testid="stFileUploaderDropzone"] button:hover,
    .st-key-input_bar .stButton > button:hover {
        background: var(--bg-hover) !important;
    }

    .st-key-input_bar .stTextInput input {
        background: transparent !important;
        color: var(--text-primary) !important;
        border: none !important;
        padding: 8px 4px !important;
        font-size: 15px !important;
    }
    .st-key-input_bar .stTextInput div[data-baseweb="input"],
    .st-key-input_bar .stTextInput div[data-baseweb="input"]:focus-within {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
    }
    .st-key-input_bar .stTextInput input::placeholder { color: var(--text-muted) !important; }

    .oauth-notice { font-size: 12px; color: var(--text-muted); padding: 6px 8px 0; }
</style>
"""


# ── UI components ─────────────────────────────────────────────────────

def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            '<div class="sb-header"><span class="sb-logo">A</span><span>RAG Assistant</span></div>',
            unsafe_allow_html=True,
        )

        if st.button("+  New chat", key="sidebar_new_chat", use_container_width=True):
            create_new_chat()
            st.rerun()

        if st.button("Re-index documents", key="sidebar_reindex_docs", use_container_width=True):
            st.session_state.needs_reindex = True
            st.rerun()

        st.markdown('<div class="sb-section-title">Recents</div>', unsafe_allow_html=True)

        chats_sorted = sorted(
            st.session_state.chats.values(),
            key=lambda c: c.get("created_at", ""),
            reverse=True,
        )

        visible_recent_limit = 6
        shown = 0
        for chat in chats_sorted:
            first_user = next(
                (
                    m.get("content", "")
                    for m in chat.get("messages", [])
                    if m.get("role") == "user"
                ),
                "",
            )
            if not first_user:
                continue
            title = _chat_title_from_question(first_user)
            if shown >= visible_recent_limit:
                break

            chat_id = chat["id"]
            is_active = chat_id == st.session_state.current_chat_id
            prefix = "• " if is_active else ""
            if st.button(
                f"{prefix}{title}",
                key=f"switch_chat_{chat_id}",
                use_container_width=True,
            ):
                switch_chat(chat_id)
                st.rerun()
            shown += 1

        st.markdown('<div class="sb-spacer"></div>', unsafe_allow_html=True)

        st.markdown('<div class="sb-profile-box">', unsafe_allow_html=True)
        left, right = st.columns([0.22, 0.78], gap="small")
        with left:
            st.markdown('<div class="sb-avatar">LS</div>', unsafe_allow_html=True)
        with right:
            st.markdown('<div class="sb-user-name">Lavneesh Sharma</div>', unsafe_allow_html=True)
            st.markdown('<div class="sb-user-email">Local session</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


def render_chat() -> None:
    if not st.session_state.messages:
        st.markdown(
            """
            <div class="empty-state">
                <div class="empty-title">How can I help you today?</div>
                <div class="empty-sub">Upload PDFs below and ask questions about them.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def render_pdf_chips() -> None:
    pdfs = st.session_state.uploaded_pdfs
    if not pdfs:
        return

    with st.container(key="pdf_chips_area"):
        for filename in pdfs:
            digest = hashlib.md5(filename.encode("utf-8")).hexdigest()[:10]
            icon_col, name_col, remove_col = st.columns(
                [0.10, 0.72, 0.18], gap="small"
            )
            with icon_col:
                st.caption("📄")
            with name_col:
                st.caption(filename)
            with remove_col:
                if st.button(
                    "×",
                    key=f"remove_pdf_{digest}",
                    help=f"Remove {filename}",
                ):
                    remove_pdf(filename)
                    st.rerun()


def render_input_area() -> None:
    user_question = None
    with st.container(key="input_panel"):
        render_pdf_chips()

        with st.container(key="input_bar"):
            upload_col, text_col, send_col = st.columns([0.08, 0.84, 0.08], gap="small")
            with upload_col:
                uploaded = st.file_uploader(
                    "Upload PDFs",
                    type=["pdf"],
                    accept_multiple_files=True,
                    key=f"pdf_uploader_{st.session_state.uploader_key}",
                    label_visibility="collapsed",
                )
            with text_col:
                pending_question = st.text_input(
                    "Ask about your documents",
                    key=f"chat_text_{st.session_state.chat_text_key}",
                    placeholder="Ask about your documents…",
                    label_visibility="collapsed",
                )
            with send_col:
                send_clicked = st.button("➤", key="chat_send")

        if uploaded:
            saved = save_uploaded_files(uploaded)
            if saved:
                 st.rerun()

        if send_clicked:
            user_question = pending_question.strip()

    if user_question:
        if not st.session_state.uploaded_pdfs:
            st.info("Upload at least one PDF to ask questions.")
            return

        st.session_state.messages.append({"role": "user", "content": user_question})
        _persist_current_chat()

        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Searching…"):
                response = create_conversational_rag_chain(user_question)
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
        _persist_current_chat()
        st.session_state.chat_text_key += 1
        st.rerun()


def main() -> None:
    init_session_state()
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    render_sidebar()
    if st.session_state.needs_reindex:
        safe_reindex()
    render_chat()
    render_input_area()


if __name__ == "__main__":
    main()
