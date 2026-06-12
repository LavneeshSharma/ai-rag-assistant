import os
import sys
import gc
import hashlib
import html
import re
import time
from typing import Any, Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv
try:
    from authlib.integrations.requests_client import OAuth2Session
except ImportError:
    OAuth2Session = None

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from chains.conversational_rag import create_conversational_rag_chain
from vector_store.chroma_store import (
    create_vector_store,
    get_active_index_path,
    reset_vector_store,
)
from db.database import (
    add_document,
    create_chat as db_create_chat,
    get_chats,
    get_documents,
    get_messages,
    init_db,
    remove_document,
    save_message,
    upsert_user,
    update_chat_active_index_path,
    update_chat_title,
)

from config.settings import DATA_DIR as REL_DATA_DIR

# Ensure backend code that uses relative paths (config.settings) resolves correctly.
os.chdir(BASE_DIR)
load_dotenv(os.path.join(BASE_DIR, ".env"))

DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, REL_DATA_DIR))

st.set_page_config(
    page_title="RAG Assistant",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items=None,
)


# ── Authentication (OAuth hooks for later) ──────────────────────────

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"
GOOGLE_OAUTH_STATE = "rag_assistant_google_auth"


def _secret(name: str) -> Optional[str]:
    try:
        value = st.secrets[name]
    except Exception:
        value = None
    if value is None:
        try:
            value = st.secrets["google"][name.removeprefix("GOOGLE_").lower()]
        except Exception:
            value = None
    return value or os.environ.get(name)


def _google_auth_config() -> Dict[str, Optional[str]]:
    return {
        "client_id": _secret("GOOGLE_CLIENT_ID"),
        "client_secret": _secret("GOOGLE_CLIENT_SECRET"),
        "redirect_uri": _secret("GOOGLE_REDIRECT_URI") or "http://localhost:8501",
    }


def _google_auth_enabled() -> bool:
    config = _google_auth_config()
    return bool(config["client_id"] and config["client_secret"] and OAuth2Session)


def _local_fallback_enabled() -> bool:
    config = _google_auth_config()
    return not config["client_id"] or not config["client_secret"] or OAuth2Session is None


def get_current_user() -> Optional[Dict[str, Any]]:
    """Return authenticated user dict or None."""
    return st.session_state.get("authenticated_user")


def _current_user_id() -> str:
    user = get_current_user() or {}
    return user.get("id", "local")


def _build_google_auth_url() -> Optional[str]:
    if not _google_auth_enabled():
        return None

    config = _google_auth_config()
    st.session_state.oauth_state = GOOGLE_OAUTH_STATE
    client = OAuth2Session(
        config["client_id"],
        config["client_secret"],
        scope="openid email profile",
        redirect_uri=config["redirect_uri"],
    )
    auth_url, _ = client.create_authorization_url(
        GOOGLE_AUTH_URL,
        state=GOOGLE_OAUTH_STATE,
        prompt="select_account",
    )
    return auth_url


def _handle_google_callback() -> None:
    if not _google_auth_enabled():
        return

    code = st.query_params.get("code")
    state = st.query_params.get("state")
    if not code:
        return
    if isinstance(state, list):
        state = state[0] if state else None
    if state != GOOGLE_OAUTH_STATE:
        st.session_state.oauth_notice = "Google sign-in state did not match."
        st.query_params.clear()
        return

    config = _google_auth_config()
    client = OAuth2Session(
        config["client_id"],
        config["client_secret"],
        redirect_uri=config["redirect_uri"],
    )
    client.fetch_token(GOOGLE_TOKEN_URL, code=code)
    userinfo = client.get(GOOGLE_USERINFO_URL).json()
    st.session_state.authenticated_user = upsert_user(
        userinfo["email"],
        userinfo.get("name") or userinfo["email"],
        userinfo.get("picture"),
    )
    st.session_state.pop("oauth_state", None)
    st.query_params.clear()
    st.rerun()


def logout_user() -> None:
    """Clear authenticated session."""
    st.session_state.authenticated_user = None
    st.session_state.active_user_id = None
    st.session_state.chats = {}
    st.session_state.current_chat_id = None
    st.session_state.messages = []
    st.session_state.pop("oauth_notice", None)
    st.session_state.pop("oauth_state", None)


def is_authenticated() -> bool:
    return get_current_user() is not None


def set_guest_user() -> None:
    st.session_state.authenticated_user = upsert_user(
        "local@session",
        "Guest",
        None,
        user_id="local",
    )


def is_cloud_user() -> bool:
    user = get_current_user() or {}
    return user.get("id") not in (None, "local")


# ── Session state ───────────────────────────────────────────────────

def _chat_title_from_question(question: str) -> str:
    text = " ".join(question.strip().split())
    lower = text.lower()

    if ("summarize" in lower or "summary" in lower) and "pdf" in lower:
        return "PDF Summary"
    if ("summarize" in lower or "summary" in lower) and "resume" in lower:
        return "Resume Summary"
    if "id card" in lower and "name" in lower:
        return "ID Card Name"
    if "how many" in lower and ("people" in lower or "person" in lower):
        return "People Mentioned"
    if "people mentioned" in lower or "persons mentioned" in lower:
        return "People Mentioned"
    if "flight" in lower:
        return "Flight Details"
    if "answer" in lower and "question" in lower:
        return "Question Answer"
    if "pdfloader" in lower or "pdf loader" in lower:
        return "PDF Loader Summary"
    if "important dates" in lower or "date" in lower or "deadline" in lower or "timeline" in lower:
        return "Important Dates"
    if "analysis" in lower or "analyze" in lower:
        return "Project Analysis"

    words = re.findall(r"[A-Za-z0-9]+", text)
    stop_words = {
        "a", "an", "are", "about", "can", "could", "detailed", "explain",
        "for", "give", "in", "is", "me", "mentioned", "of", "on", "please",
        "tell", "the", "this", "to", "what", "whats", "which",
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
    if not chat_id or chat_id not in st.session_state.chats:
        return

    st.session_state.chats[chat_id]["messages"] = list(st.session_state.messages)
    first_user = next(
        (m["content"] for m in st.session_state.messages if m["role"] == "user"),
        None,
    )
    if first_user and st.session_state.chats[chat_id]["title"] == "New chat":
        title = _chat_title_from_question(first_user)
        st.session_state.chats[chat_id]["title"] = title
        update_chat_title(chat_id, title, _current_user_id())


def _messages_to_history_pairs(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    pairs = []
    pending_question = None

    for message in messages:
        if message.get("role") == "user":
            pending_question = message.get("content", "")
        elif message.get("role") == "assistant" and pending_question is not None:
            pairs.append({
                "question": pending_question,
                "answer": message.get("content", ""),
            })
            pending_question = None

    return pairs


def _ask_current_chat(
    question: str,
    active_index_path: Optional[str],
    previous_messages: List[Dict[str, str]],
) -> str:
    if "chat_messages" in create_conversational_rag_chain.__code__.co_varnames:
        return create_conversational_rag_chain(
            question,
            active_index_path,
            chat_messages=previous_messages,
        )

    create_conversational_rag_chain.__globals__["chat_history"] = (
        _messages_to_history_pairs(previous_messages)
    )
    create_conversational_rag_chain.__globals__["conversation_summary"] = ""
    return create_conversational_rag_chain(question, active_index_path)


def init_session_state() -> None:
    init_db()
    defaults = {
        "messages": [],
        "chats": {},
        "current_chat_id": None,
        "uploaded_pdfs": [],
        "uploaded_pdf_docs": [],
        "show_uploader": False,
        "authenticated_user": None,
        "active_user_id": None,
        "oauth_state": None,
        "selected_pdf_ids": [],
        "uploader_key": 0,
        "chat_text_key": 0,
        "is_indexing": False,
        "needs_reindex": False,
        "oauth_notice": None,
        "chat_search": "",
        "show_auth_dialog": False,
        "pending_user_question": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    _handle_google_callback()
    if not st.session_state.authenticated_user:
        set_guest_user()
    if not is_authenticated():
        return

    user_id = _current_user_id()
    if st.session_state.active_user_id != user_id:
        st.session_state.active_user_id = user_id
        st.session_state.messages = []
        st.session_state.chats = {}
        st.session_state.current_chat_id = None
        st.session_state.uploaded_pdfs = []
        st.session_state.uploaded_pdf_docs = []

    os.makedirs(DATA_DIR, exist_ok=True)

    db_chats = get_chats(user_id)
    if db_chats:
        st.session_state.chats = {
            chat["id"]: {**chat, "messages": get_messages(chat["id"], user_id)}
            for chat in db_chats
        }
    elif st.session_state.chats:
        migrated_chats = {}
        migrated_current_chat_id = None
        for old_chat_id, old_chat in st.session_state.chats.items():
            chat = db_create_chat(old_chat.get("title", "New chat"), user_id)
            messages = []
            for message in old_chat.get("messages", []):
                saved = save_message(
                    chat["id"],
                    message["role"],
                    message["content"],
                    user_id,
                )
                messages.append(saved)
            migrated_chats[chat["id"]] = {**chat, "messages": messages}
            if old_chat_id == st.session_state.current_chat_id:
                migrated_current_chat_id = chat["id"]
        st.session_state.chats = migrated_chats
        st.session_state.current_chat_id = migrated_current_chat_id
    else:
        chat = db_create_chat("New chat", user_id)
        st.session_state.chats[chat["id"]] = {**chat, "messages": []}
        st.session_state.current_chat_id = chat["id"]

    if (
        st.session_state.current_chat_id is not None
        and st.session_state.current_chat_id not in st.session_state.chats
    ):
        st.session_state.current_chat_id = None
        st.session_state.messages = []

    if st.session_state.current_chat_id is None and st.session_state.chats:
        st.session_state.current_chat_id = next(iter(st.session_state.chats))

    if st.session_state.current_chat_id:
        st.session_state.messages = list(
            st.session_state.chats[st.session_state.current_chat_id]["messages"]
        )

    _sync_pdfs_for_current_chat()

def _sync_pdfs_for_current_chat() -> None:
    chat_id = st.session_state.current_chat_id
    if not chat_id:
        st.session_state.uploaded_pdf_docs = []
        st.session_state.uploaded_pdfs = []
        return
    docs = get_documents(chat_id, _current_user_id())
    st.session_state.uploaded_pdf_docs = docs
    st.session_state.uploaded_pdfs = [doc["filename"] for doc in docs]


def create_new_chat() -> None:
    _persist_current_chat()
    chat = db_create_chat("New chat", _current_user_id())
    st.session_state.chats[chat["id"]] = {**chat, "messages": []}
    st.session_state.current_chat_id = chat["id"]
    st.session_state.messages = []
    st.session_state.uploaded_pdf_docs = []
    st.session_state.uploaded_pdfs = []


def switch_chat(chat_id: str) -> None:
    if chat_id == st.session_state.current_chat_id:
        return
    if chat_id not in st.session_state.chats:
        return
    _persist_current_chat()
    st.session_state.current_chat_id = chat_id
    st.session_state.messages = get_messages(chat_id, _current_user_id())
    st.session_state.chats[chat_id]["messages"] = list(st.session_state.messages)
    _sync_pdfs_for_current_chat()


def clear_current_chat() -> None:
    st.session_state.messages = []
    _persist_current_chat()


def save_uploaded_files(uploaded_files: Optional[List[Any]]) -> bool:
    if not uploaded_files:
        return False

    changed = False
    existing_docs = {
        doc["filename"]
        for doc in get_documents(
            st.session_state.current_chat_id,
            _current_user_id(),
        )
    }
    for uf in uploaded_files:
        name = uf.name
        if not name.lower().endswith(".pdf"):
            continue
        save_path = os.path.join(DATA_DIR, name)
        if name in existing_docs:
            continue
        if not os.path.exists(save_path):
            with open(save_path, "wb") as f:
                f.write(uf.getbuffer())
        add_document(
            st.session_state.current_chat_id,
            name,
            save_path,
            _current_user_id(),
        )
        existing_docs.add(name)
        changed = True

    if changed:
        _sync_pdfs_for_current_chat()
        st.session_state.needs_reindex = True
        st.session_state.uploader_key += 1

    return changed



def remove_pdf(filename: str) -> None:
    document = next(
        (
            doc for doc in st.session_state.uploaded_pdf_docs
            if doc["filename"] == filename
        ),
        None,
    )
    if document:
        remove_document(document["id"], _current_user_id())
    _sync_pdfs_for_current_chat()
    if filename in st.session_state.selected_pdf_ids:
        st.session_state.selected_pdf_ids.remove(filename)
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
            docs = get_documents(
                st.session_state.current_chat_id,
                _current_user_id(),
            )
            if docs:
                create_vector_store(pdf_paths=[doc["file_path"] for doc in docs])
                active_index_path = get_active_index_path()
            else:
                reset_vector_store()
                active_index_path = None
            update_chat_active_index_path(
                st.session_state.current_chat_id,
                active_index_path,
                _current_user_id(),
            )
            st.session_state.chats[st.session_state.current_chat_id][
                "active_index_path"
            ] = active_index_path
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
        --sidebar-width: 260px;
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
        padding: 12px 8px 84px !important;
        display: flex;
        flex-direction: column;
        min-height: 100vh;
    }
    section[data-testid="stSidebar"] .element-container {
        margin-bottom: 2px !important;
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
        padding: 7px 10px !important;
        border-radius: 8px !important;
        font-size: 15px !important;
        font-weight: 400 !important;
        box-shadow: none !important;
        min-height: 34px !important;
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
        padding: 4px 8px 12px;
        font-size: 16px;
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
        padding: 10px 10px 4px;
        font-size: 13px;
        font-weight: 500;
        color: var(--text-muted);
        text-transform: none;
        letter-spacing: 0;
    }
    .st-key-recents_list .element-container {
        margin-bottom: 0 !important;
    }
    .st-key-recents_list[data-testid="stVerticalBlock"],
    .st-key-recents_list [data-testid="stVerticalBlock"],
    .st-key-recents_list [data-testid="stVerticalBlockBorderWrapper"] {
        gap: 1px !important;
    }
    .st-key-recents_list .stButton > button {
        min-height: 36px !important;
        height: 36px !important;
        padding: 7px 10px !important;
        font-size: 14px !important;
        line-height: 1.15 !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        white-space: nowrap !important;
        justify-content: flex-start !important;
    }
    .st-key-recents_list button {
        font-size: 14px !important;
    }
    section[data-testid="stSidebar"] .st-key-recents_list [data-testid="stBaseButton-secondary"] {
        font-size: 14px !important;
    }
    .st-key-recent_active_chat .stButton > button {
        background: #242424 !important;
        color: #ffffff !important;
        border-radius: 8px !important;
    }
    section[data-testid="stSidebar"] .st-key-recent_active_chat [data-testid="stBaseButton-secondary"] {
        background: #242424 !important;
        color: #ffffff !important;
        border-radius: 8px !important;
    }
    .st-key-profile_area {
        position: fixed;
        left: 8px;
        bottom: 8px;
        width: 244px;
        background: var(--bg-sidebar);
        padding-top: 6px;
        z-index: 20;
    }
    .st-key-profile_area .stButton > button {
        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
        gap: 8px !important;
        min-height: 42px !important;
        padding: 8px 10px !important;
        font-size: 14px !important;
    }
    .sb-user-name { font-size: 14px; font-weight: 500; color: #f2f2f2; line-height: 1.25; }
    .sb-user-email { font-size: 12px; color: #a6a6a6; line-height: 1.25; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .sb-avatar {
        width: 28px; height: 28px; border-radius: 50%;
        background: #e8605f; color: #fff;
        display: flex; align-items: center; justify-content: center;
        font-size: 12px; font-weight: 600;
    }
    .sb-avatar-img {
        width: 28px; height: 28px; border-radius: 50%;
        object-fit: cover; display: block;
    }

    .main .block-container {
        max-width: 760px;
        margin: 0 auto;
        padding: 46px 1rem 150px 1rem;
    }

    .empty-state { text-align: center; padding: 18vh 1rem 0; }
    .empty-title { font-size: 26px; font-weight: 600; color: var(--text-primary); margin-bottom: 8px; }
    .empty-sub { font-size: 14px; color: var(--text-secondary); }

    .stChatMessage { padding: 8px 0 !important; }
    [data-testid="stChatMessageContent"] {
        background: transparent !important;
        border: none !important;
        color: var(--text-primary) !important;
        font-size: 15px !important;
        line-height: 1.7 !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        display: flex !important;
        justify-content: flex-end !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        display: flex !important;
        justify-content: flex-start !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])
    [data-testid="stChatMessageContent"] {
        max-width: 78%;
        margin-right: auto;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
    [data-testid="stChatMessageContent"] {
        background: var(--bg-input) !important;
        border-radius: 16px !important;
        padding: 9px 13px !important;
        max-width: 70%;
        margin-left: auto !important;
        margin-right: 0 !important;
        line-height: 1.45 !important;
    }
    [data-testid="stChatMessageAvatarUser"],
    [data-testid="stChatMessageAvatarAssistant"] {
        display: none !important;
    }
    .assistant-answer {
        color: var(--text-primary);
        font-size: 15px;
        line-height: 1.65;
    }
    .assistant-sources {
        margin-top: 10px;
        color: var(--text-secondary);
        font-size: 13px;
        line-height: 1.45;
    }

    .st-key-input_panel {
        position: fixed;
        bottom: 0;
        left: var(--sidebar-width);
        right: 0;
        background: var(--bg-main);
        padding: 4px 16px 16px;
        z-index: 100;
        width: auto;
        max-width: none;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        margin: 0 !important;
        box-sizing: border-box !important;
    }
    .st-key-pdf_chips_area {
        margin-bottom: 4px;
        width: min(760px, calc(100vw - var(--sidebar-width) - 32px)) !important;
        max-width: min(760px, calc(100vw - var(--sidebar-width) - 32px)) !important;
    }
    .st-key-pdf_chips_area [class*="st-key-pdf_chip_"][data-testid="stVerticalBlock"] {
        position: relative !important;
        width: min(100%, 320px) !important;
        height: 28px !important;
        min-height: 28px !important;
        background: #292929 !important;
        border-radius: 16px !important;
        overflow: hidden !important;
        padding: 0 !important;
        gap: 0 !important;
    }
    .st-key-pdf_chips_area [class*="st-key-pdf_chip_"] .element-container {
        margin: 0 !important;
    }
    .pdf-chip-label {
        position: absolute;
        inset: 3px 30px 3px 8px;
        display: flex;
        align-items: center;
        gap: 6px;
        min-width: 0;
        color: var(--text-secondary);
        font-size: 13px;
        line-height: 22px;
        pointer-events: none;
    }
    .pdf-chip-name {
        min-width: 0;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    .st-key-pdf_chips_area [class*="st-key-remove_pdf_"] {
        position: absolute !important;
        top: 3px !important;
        right: 4px !important;
        width: 22px !important;
        height: 22px !important;
        margin: 0 !important;
        z-index: 2 !important;
    }
    .st-key-pdf_chips_area [class*="st-key-remove_pdf_"] .stButton {
        position: static !important;
        width: 22px !important;
        height: 22px !important;
        margin: 0 !important;
    }
    .st-key-pdf_chips_area [data-testid="stHorizontalBlock"] {
        width: min(100%, 320px) !important;
        max-width: min(100%, 320px);
        background: #292929;
        border: none;
        border-radius: 16px;
        padding: 1px 4px 1px 8px;
        align-items: center !important;
        gap: 0 !important;
        flex-wrap: nowrap !important;
        overflow: hidden !important;
        min-height: 28px !important;
    }
    .st-key-pdf_chips_area [data-testid="stHorizontalBlock"] > [data-testid="column"] {
        min-width: 0 !important;
        flex-grow: 0 !important;
    }
    .st-key-pdf_chips_area [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(1) {
        flex: 0 0 22px !important;
        width: 22px !important;
    }
    .st-key-pdf_chips_area [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(2) {
        flex: 1 1 auto !important;
        width: calc(100% - 54px) !important;
        max-width: calc(100% - 54px) !important;
        overflow: hidden !important;
    }
    .st-key-pdf_chips_area [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(3) {
        flex: 0 0 28px !important;
        width: 28px !important;
    }
    .st-key-pdf_chips_area [data-testid="stCaptionContainer"],
    .st-key-pdf_chips_area [data-testid="stCaptionContainer"] p {
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        max-width: 100% !important;
        line-height: 22px !important;
        margin: 0 !important;
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
    .st-key-pdf_chips_area [data-testid="stBaseButton-secondary"] {
        min-height: 22px !important;
        height: 22px !important;
        width: 22px !important;
        padding: 0 !important;
        font-size: 13px !important;
        line-height: 1 !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: var(--text-muted) !important;
        border-radius: 50% !important;
    }
    .st-key-pdf_chips_area .stButton > button:hover {
        background: rgba(255,255,255,0.10) !important;
        color: var(--danger) !important;
    }
    .st-key-input_bar {
        width: min(760px, calc(100vw - var(--sidebar-width) - 32px)) !important;
        max-width: min(760px, calc(100vw - var(--sidebar-width) - 32px)) !important;
        background: var(--bg-input);
        border-radius: 24px;
        padding: 5px 8px;
        box-shadow: none !important;
        box-sizing: border-box !important;
    }
    .st-key-input_bar [data-testid="stHorizontalBlock"] {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        align-items: center !important;
        gap: 0.25rem !important;
        width: 100% !important;
        min-height: 42px !important;
        background: transparent !important;
    }
    .st-key-input_bar [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        min-width: 0 !important;
        flex: 0 0 auto !important;
        width: auto !important;
    }
    .st-key-input_bar [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(1),
    .st-key-input_bar [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(3) {
        flex: 0 0 34px !important;
        width: 34px !important;
        max-width: 34px !important;
    }
    .st-key-input_bar [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(2) {
        flex: 1 1 auto !important;
        width: auto !important;
        max-width: none !important;
        min-width: 0 !important;
    }
    .st-key-input_bar .element-container,
    .st-key-input_bar .stTextInput,
    .st-key-input_bar .stButton,
    .st-key-input_bar [data-testid="stFileUploader"] {
        margin: 0 !important;
        width: 100% !important;
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
    .st-key-input_bar .stTextInput div[data-baseweb="base-input"],
    .st-key-input_bar .stTextInput div[data-baseweb="input"] > div,
    .st-key-input_bar .stTextInput div[data-baseweb="input"]:focus-within {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
    }
    .st-key-input_bar .stTextInput input:focus {
        outline: none !important;
        box-shadow: none !important;
    }
    .st-key-input_bar .stTextInput input::placeholder { color: var(--text-muted) !important; }

    @media (max-width: 760px) {
        .st-key-input_panel {
            left: 0;
            right: 0;
            padding-left: 16px;
            padding-right: 16px;
        }
        .st-key-pdf_chips_area,
        .st-key-input_bar {
            width: calc(100vw - 32px) !important;
            max-width: calc(100vw - 32px) !important;
        }
    }

    .oauth-notice { font-size: 12px; color: var(--text-muted); padding: 6px 8px 0; }
    .auth-dialog-copy {
        color: var(--text-secondary);
        font-size: 14px;
        line-height: 1.45;
        margin-bottom: 12px;
    }
    .account-menu-user {
        padding: 6px 4px 8px;
        border-bottom: 1px solid #303030;
        margin-bottom: 6px;
    }
    div[data-testid="stDialog"] div[role="dialog"] {
        background: #202020 !important;
        border: 1px solid #303030 !important;
        color: var(--text-primary) !important;
    }
</style>
"""


# ── UI components ─────────────────────────────────────────────────────

def render_sign_in_dialog() -> None:
    @st.dialog("RAG Assistant", width="small")
    def sign_in_dialog() -> None:
        user = get_current_user() or {}
        if is_cloud_user():
            name = user.get("name") or "Account"
            email = user.get("email") or ""
            st.markdown(
                f"""
                <div class="account-menu-user">
                    <div class="sb-user-name">{html.escape(name)}</div>
                    <div class="sb-user-email">{html.escape(email)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Sign out", key="account_modal_sign_out", use_container_width=True):
                st.session_state.show_auth_dialog = False
                logout_user()
                st.rerun()
            return

        st.markdown(
            '<div class="auth-dialog-copy">Sign in with Google, or continue as guest.</div>',
            unsafe_allow_html=True,
        )
        auth_url = _build_google_auth_url()
        if st.session_state.get("oauth_notice"):
            st.warning(st.session_state.oauth_notice)
        if auth_url:
            st.link_button("Continue with Google", auth_url, use_container_width=True)
        else:
            st.markdown(
                '<div class="oauth-notice">Google OAuth not configured yet.</div>',
                unsafe_allow_html=True,
            )
            st.button(
                "Continue with Google",
                key="google_disabled",
                use_container_width=True,
                disabled=True,
            )
        if st.button("Continue as guest", key="continue_as_guest", use_container_width=True):
            set_guest_user()
            st.session_state.show_auth_dialog = False
            st.rerun()

    sign_in_dialog()


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

        with st.container(key="recents_list"):
            visible_recent_limit = 6
            shown = 0
            title_counts: Dict[str, int] = {}
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
                title_counts[title] = title_counts.get(title, 0) + 1
                display_title = (
                    f"{title} {title_counts[title]}"
                    if title_counts[title] > 1
                    else title
                )
                if shown >= visible_recent_limit:
                    break

                chat_id = chat["id"]
                is_active = chat_id == st.session_state.current_chat_id
                row_key = "recent_active_chat" if is_active else f"recent_chat_{chat_id}"
                with st.container(key=row_key):
                    if st.button(
                        display_title,
                        key=f"switch_chat_{chat_id}",
                        use_container_width=True,
                    ):
                        switch_chat(chat_id)
                        st.rerun()
                shown += 1

        with st.container(key="profile_area"):
            user = get_current_user() or {}
            name = user.get("name") or ""
            email = user.get("email") or ""

            if user.get("id") == "local":
                profile_label = "Guest session"
            elif not user:
                profile_label = "Sign in"
            elif email:
                profile_label = f"{name or 'Account'} · {email}"
            else:
                profile_label = name or "Account"

            if st.button(profile_label, key="sidebar_profile_action", use_container_width=True):
                st.session_state.show_auth_dialog = True
                st.rerun()


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
            render_message_content(message["role"], message["content"])


def _split_answer_sources(content: str) -> Dict[str, str]:
    text = content.strip()
    if text.startswith("Answer:"):
        text = text[len("Answer:"):].strip()

    if "\nSources:" not in text:
        return {"answer": text, "sources": ""}

    answer, sources = text.split("\nSources:", 1)
    cleaned_sources = "\n".join(
        line.strip()
        for line in sources.strip().splitlines()
        if line.strip()
    )
    return {"answer": answer.strip(), "sources": cleaned_sources}


def render_message_content(role: str, content: str) -> None:
    if role != "assistant":
        st.markdown(html.escape(content))
        return

    parts = _split_answer_sources(content)
    answer_html = html.escape(parts["answer"]).replace("\n", "<br>")
    sources_html = html.escape(parts["sources"]).replace("\n", "<br>")
    st.markdown(
        f'<div class="assistant-answer">{answer_html}</div>',
        unsafe_allow_html=True,
    )
    if parts["sources"]:
        st.markdown(
            f'<div class="assistant-sources"><strong>Sources:</strong> {sources_html}</div>',
            unsafe_allow_html=True,
        )


def render_pdf_chips() -> None:
    pdfs = st.session_state.uploaded_pdfs
    if not pdfs:
        return

    with st.container(key="pdf_chips_area"):
        for filename in pdfs:
            digest = hashlib.md5(filename.encode("utf-8")).hexdigest()[:10]
            with st.container(key=f"pdf_chip_{digest}"):
                st.markdown(
                    f"""
                    <div class="pdf-chip-label">
                        <span>📄</span>
                        <span class="pdf-chip-name">{html.escape(filename)}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button(
                    "×",
                    key=f"remove_pdf_{digest}",
                    help=f"Remove {filename}",
                ):
                    remove_pdf(filename)
                    st.rerun()


def queue_user_question(input_key: str) -> None:
    question = st.session_state.get(input_key, "").strip()
    if question:
        st.session_state.pending_user_question = question


def handle_pending_question() -> bool:
    user_question = st.session_state.get("pending_user_question")
    if not user_question:
        return False

    st.session_state.pending_user_question = None
    if not st.session_state.uploaded_pdfs:
        st.info("Upload at least one PDF to ask questions.")
        return False

    previous_messages = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": user_question})
    save_message(
        st.session_state.current_chat_id,
        "user",
        user_question,
        _current_user_id(),
    )
    _persist_current_chat()

    render_chat()

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = _ask_current_chat(
                user_question,
                st.session_state.chats[st.session_state.current_chat_id].get(
                    "active_index_path"
                ),
                previous_messages,
            )
        render_message_content("assistant", response)

    st.session_state.messages.append({"role": "assistant", "content": response})
    save_message(
        st.session_state.current_chat_id,
        "assistant",
        response,
        _current_user_id(),
    )
    _persist_current_chat()
    st.session_state.chat_text_key += 1
    st.rerun()


def render_input_area() -> None:
    text_key = f"chat_text_{st.session_state.chat_text_key}"
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
                    key=text_key,
                    placeholder="Ask about your documents…",
                    label_visibility="collapsed",
                    on_change=queue_user_question,
                    args=(text_key,),
                )
            with send_col:
                send_clicked = st.button("➤", key="chat_send")

        if uploaded:
            saved = save_uploaded_files(uploaded)
            if saved:
                 st.rerun()

        if send_clicked:
            st.session_state.pending_user_question = pending_question.strip()
            st.rerun()


def main() -> None:
    init_session_state()
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    render_sidebar()
    if st.session_state.show_auth_dialog:
        render_sign_in_dialog()
    if st.session_state.needs_reindex:
        safe_reindex()
    if not handle_pending_question():
        render_chat()
    render_input_area()


if __name__ == "__main__":
    main()
