try:
    __import__("pysqlite3")
    import sys as _sys
    _sys.modules["sqlite3"] = _sys.modules.pop("pysqlite3")
except ImportError:
    pass

import os
import sys
import gc
import hashlib
import html
import json
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
try:
    from authlib.integrations.requests_client import OAuth2Session
except ImportError:
    OAuth2Session = None

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from chains.conversational_rag import stream_conversational_rag_chain
from vector_store.chroma_store import (
    cleanup_orphaned_indexes,
    create_vector_store,
    get_active_index_path,
    reset_vector_store,
)
from db.database import (
    add_document,
    create_chat as db_create_chat,
    delete_chat as db_delete_chat,
    delete_message,
    get_all_active_index_paths,
    get_chats,
    get_documents,
    get_messages,
    get_message_activity,
    get_usage_counts,
    init_db,
    remove_document,
    save_message,
    upsert_user,
    update_chat_active_index_path,
    update_chat_title,
)
from utils.stats import aggregate_usage_stats, read_eval_summary, read_trace_events, trace_file_fingerprint

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


def _date_bucket(iso_timestamp: str) -> str:
    """Bucket a chat's timestamp into a ChatGPT-style sidebar group label."""
    try:
        dt = datetime.fromisoformat(iso_timestamp)
    except (ValueError, TypeError):
        return "Older"
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    delta_days = (datetime.now(timezone.utc).date() - dt.astimezone(timezone.utc).date()).days
    if delta_days <= 0:
        return "Today"
    if delta_days == 1:
        return "Yesterday"
    if delta_days <= 7:
        return "Previous 7 Days"
    if delta_days <= 30:
        return "Previous 30 Days"
    return "Older"


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
        "is_indexing": False,
        "needs_reindex": False,
        "oauth_notice": None,
        "chat_search": "",
        "show_auth_dialog": False,
        "pending_user_question": None,
        "active_view": "chat",
        "confirm_delete_chat_id": None,
        "regenerate_question": None,
        "is_responding": False,
        "sidebar_collapsed": False,
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
    st.session_state.active_view = "chat"


def switch_chat(chat_id: str) -> None:
    st.session_state.active_view = "chat"
    if chat_id == st.session_state.current_chat_id:
        return
    if chat_id not in st.session_state.chats:
        return
    _persist_current_chat()
    st.session_state.current_chat_id = chat_id
    st.session_state.messages = get_messages(chat_id, _current_user_id())
    st.session_state.chats[chat_id]["messages"] = list(st.session_state.messages)
    _sync_pdfs_for_current_chat()


def delete_chat_and_switch(chat_id: str) -> None:
    db_delete_chat(chat_id, _current_user_id())
    st.session_state.chats.pop(chat_id, None)

    if st.session_state.current_chat_id != chat_id:
        return

    remaining = sorted(
        st.session_state.chats.values(),
        key=lambda c: c.get("updated_at", ""),
        reverse=True,
    )
    if remaining:
        switch_chat(remaining[0]["id"])
    else:
        create_new_chat()


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
            keep_paths = set(get_all_active_index_paths())
            if active_index_path:
                keep_paths.add(active_index_path)
            cleanup_orphaned_indexes(keep_paths)
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

def build_custom_css(sidebar_collapsed: bool) -> str:
    sidebar_width = "0px" if sidebar_collapsed else "260px"
    return f"""
<style>
    :root {{
        --sidebar-width: {sidebar_width};
        --bg-main: #212121;
        --bg-sidebar: #181818;
        --bg-input: #2f2f2f;
        --bg-hover: #2a2a2a;
        --bg-elevated: #303030;
        --border-color: #3a3a3a;
        --text-primary: #ececec;
        --text-secondary: #b4b4b4;
        --text-muted: #8e8ea0;
        --accent: #10a37f;
        --danger: #ef4444;
        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 16px;
        --radius-pill: 999px;
    }}

    * {{ scrollbar-color: #4a4a4a transparent; }}
    ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
    ::-webkit-scrollbar-thumb {{ background: #4a4a4a; border-radius: 4px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: #5a5a5a; }}
    ::-webkit-scrollbar-track {{ background: transparent; }}

    .stApp {{ background: var(--bg-main); }}
    [data-testid="collapsedControl"] {{ display: none !important; }}
    [data-testid="stSidebarCollapseButton"] {{ display: none !important; }}
    [data-testid="stExpandSidebarButton"] {{ display: none !important; }}
    header[data-testid="stHeader"] {{ background: transparent !important; height: 2.2rem !important; }}

    /* ── Sidebar shell ─────────────────────────────────────────── */
    section[data-testid="stSidebar"] {{
        background: var(--bg-sidebar) !important;
        width: var(--sidebar-width) !important;
        min-width: var(--sidebar-width) !important;
        max-width: var(--sidebar-width) !important;
        border-right: none !important;
        overflow: hidden !important;
        transform: none !important;
        transition: min-width 0.18s ease, max-width 0.18s ease, width 0.18s ease;
    }}
    @media (max-width: 768px) {{
        section[data-testid="stSidebar"] {{
            position: fixed !important;
            top: 0;
            bottom: 0;
            left: 0;
            z-index: 999 !important;
            box-shadow: {"none" if sidebar_collapsed else "2px 0 24px rgba(0,0,0,0.5)"};
        }}
    }}
    section[data-testid="stSidebar"] > div {{ padding: 0 !important; }}
    section[data-testid="stSidebar"] .block-container {{
        padding: 8px 8px 10px !important;
        display: flex;
        flex-direction: column;
        min-height: 100vh;
        width: 260px;
    }}
    section[data-testid="stSidebar"] .element-container {{ margin-bottom: 2px !important; }}
    section[data-testid="stSidebar"] .stMarkdown {{ margin-bottom: 0 !important; }}

    /* `.stButton button` (descendant), not `.stButton > button` (direct child):
       a button with help= gets wrapped in extra tooltip divs by Streamlit,
       breaking direct-child selectors throughout this stylesheet. */
    section[data-testid="stSidebar"] .stButton button {{
        width: 100%;
        text-align: left;
        background: transparent !important;
        border: none !important;
        color: #e8e8e8 !important;
        padding: 8px 10px !important;
        border-radius: var(--radius-sm) !important;
        font-size: 14px !important;
        font-weight: 400 !important;
        box-shadow: none !important;
        min-height: 36px !important;
        line-height: 1.2 !important;
        transition: background 0.12s ease;
    }}
    section[data-testid="stSidebar"] .stButton button:hover {{
        background: var(--bg-hover) !important;
        color: #ffffff !important;
    }}

    .st-key-sidebar_header_row {{ margin-bottom: 8px; }}
    .st-key-sidebar_header_row [data-testid="stHorizontalBlock"] {{
        align-items: center !important;
        gap: 0 !important;
    }}
    .st-key-sidebar_header_row .stButton button {{
        width: auto !important;
        background: transparent !important;
        border: none !important;
        color: var(--text-secondary) !important;
        font-size: 16px !important;
        padding: 6px 10px !important;
        min-height: 34px !important;
        border-radius: var(--radius-sm) !important;
        box-shadow: none !important;
    }}
    .st-key-sidebar_header_row .stButton button:hover {{
        background: var(--bg-hover) !important;
        color: #ffffff !important;
    }}
    .sb-header {{
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 8px 10px;
        font-size: 15px;
        font-weight: 600;
        color: #ffffff;
    }}
    .sb-logo {{
        width: 24px; height: 24px; border-radius: 7px;
        background: linear-gradient(135deg, #10a37f, #0d8a6c);
        color: #fff;
        display: inline-flex; align-items: center; justify-content: center;
        font-size: 13px; font-weight: 700;
        flex-shrink: 0;
    }}
    .sb-section-title {{
        padding: 14px 10px 6px;
        font-size: 11px;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    .sb-date-group {{
        padding: 12px 10px 4px;
        font-size: 11px;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    .st-key-sidebar_search {{ padding: 2px 2px 8px; }}
    .st-key-sidebar_search .stTextInput input {{
        background: var(--bg-elevated) !important;
        color: var(--text-primary) !important;
        border: 1px solid transparent !important;
        border-radius: var(--radius-pill) !important;
        font-size: 13px !important;
        padding: 8px 14px !important;
    }}
    .st-key-sidebar_search .stTextInput div[data-baseweb="input"],
    .st-key-sidebar_search .stTextInput div[data-baseweb="base-input"] {{
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }}
    .st-key-sidebar_search .stTextInput input::placeholder {{ color: var(--text-muted) !important; }}
    .st-key-sidebar_search .stTextInput input:focus {{
        outline: none !important;
        box-shadow: 0 0 0 1px var(--accent) !important;
    }}

    .st-key-recents_list .element-container {{ margin-bottom: 0 !important; }}
    .st-key-recents_list[data-testid="stVerticalBlock"],
    .st-key-recents_list [data-testid="stVerticalBlock"],
    .st-key-recents_list [data-testid="stVerticalBlockBorderWrapper"] {{ gap: 1px !important; }}
    .st-key-recents_list [data-testid="stHorizontalBlock"] {{
        gap: 0.15rem !important;
        align-items: center !important;
    }}
    .st-key-recents_list .stButton button {{
        min-height: 34px !important;
        height: 34px !important;
        padding: 7px 10px !important;
        font-size: 13.5px !important;
        line-height: 1.2 !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        white-space: nowrap !important;
        justify-content: flex-start !important;
        border-radius: var(--radius-sm) !important;
    }}
    .st-key-recent_active_chat .stButton button,
    section[data-testid="stSidebar"] .st-key-recent_active_chat [data-testid="stBaseButton-secondary"] {{
        background: var(--bg-elevated) !important;
        color: #ffffff !important;
        border-radius: var(--radius-sm) !important;
    }}

    .st-key-profile_area {{
        position: sticky;
        bottom: 0;
        background: var(--bg-sidebar);
        padding: 8px 0 2px;
        margin-top: auto;
        border-top: 1px solid var(--border-color);
    }}
    .st-key-profile_area .stButton button {{
        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
        gap: 8px !important;
        min-height: 42px !important;
        padding: 8px 10px !important;
        font-size: 14px !important;
    }}
    .sb-user-name {{ font-size: 14px; font-weight: 500; color: #f2f2f2; line-height: 1.25; }}
    .sb-user-email {{ font-size: 12px; color: #a6a6a6; line-height: 1.25; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
    .sb-avatar {{
        width: 28px; height: 28px; border-radius: 50%;
        background: #e8605f; color: #fff;
        display: flex; align-items: center; justify-content: center;
        font-size: 12px; font-weight: 600;
    }}
    .sb-avatar-img {{
        width: 28px; height: 28px; border-radius: 50%;
        object-fit: cover; display: block;
    }}

    /* ── Top app header (main content) ────────────────────────────── */
    .st-key-app_header {{
        position: sticky;
        top: 0;
        z-index: 40;
        background: var(--bg-main);
        padding: 10px 2px 12px;
    }}
    .st-key-app_header [data-testid="stHorizontalBlock"] {{
        align-items: center !important;
        gap: 4px !important;
    }}
    .st-key-app_header .stButton button {{
        background: transparent !important;
        border: none !important;
        color: var(--text-secondary) !important;
        font-size: 16px !important;
        padding: 6px 10px !important;
        min-height: 34px !important;
        border-radius: var(--radius-sm) !important;
        box-shadow: none !important;
        transition: background 0.12s ease;
    }}
    .st-key-app_header .stButton button:hover {{
        background: var(--bg-hover) !important;
        color: #fff !important;
    }}
    .app-header-title {{
        font-size: 14px;
        font-weight: 600;
        color: var(--text-secondary);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }}

    /* ── Main content ──────────────────────────────────────────── */
    .main .block-container {{
        max-width: 760px;
        margin: 0 auto;
        padding: 0 1rem 140px 1rem;
    }}

    .empty-state {{ text-align: center; padding: 12vh 1rem 0; }}
    .empty-title {{ font-size: 28px; font-weight: 600; color: var(--text-primary); margin-bottom: 8px; letter-spacing: -0.01em; }}
    .empty-sub {{ font-size: 14px; color: var(--text-secondary); margin-bottom: 26px; }}

    .st-key-starter_prompts {{ margin-bottom: 12px; }}
    .st-key-starter_prompts [data-testid="stHorizontalBlock"] {{ gap: 10px !important; }}
    .st-key-starter_prompts .stButton button {{
        background: var(--bg-elevated) !important;
        border: 1px solid var(--border-color) !important;
        color: var(--text-primary) !important;
        border-radius: var(--radius-md) !important;
        padding: 12px 14px !important;
        font-size: 13px !important;
        text-align: left !important;
        white-space: normal !important;
        min-height: 64px !important;
        box-shadow: none !important;
        transition: background 0.12s ease, border-color 0.12s ease;
    }}
    .st-key-starter_prompts .stButton button:hover {{
        background: var(--bg-hover) !important;
        border-color: #4a4a4a !important;
    }}

    /* ── Active documents bar ─────────────────────────────────────── */
    .st-key-active_docs_bar {{ margin: 0 0 14px; }}
    .st-key-active_docs_bar [data-testid="stHorizontalBlock"] {{ flex-wrap: wrap !important; gap: 6px !important; }}
    .st-key-active_docs_bar [class*="st-key-pdf_chip_"][data-testid="stVerticalBlock"] {{
        position: relative !important;
        width: min(100%, 260px) !important;
        height: 30px !important;
        min-height: 30px !important;
        background: var(--bg-elevated) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: var(--radius-pill) !important;
        overflow: hidden !important;
        padding: 0 !important;
        gap: 0 !important;
    }}
    .st-key-active_docs_bar [class*="st-key-pdf_chip_"] .element-container {{ margin: 0 !important; }}
    .pdf-chip-label {{
        position: absolute;
        inset: 3px 30px 3px 10px;
        display: flex;
        align-items: center;
        gap: 6px;
        min-width: 0;
        color: var(--text-secondary);
        font-size: 12.5px;
        line-height: 24px;
        pointer-events: none;
    }}
    .pdf-chip-name {{
        min-width: 0;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }}
    .st-key-active_docs_bar [class*="st-key-remove_pdf_"] {{
        position: absolute !important;
        top: 3px !important;
        right: 4px !important;
        width: 22px !important;
        height: 22px !important;
        margin: 0 !important;
        z-index: 2 !important;
    }}
    .st-key-active_docs_bar [class*="st-key-remove_pdf_"] .stButton {{
        position: static !important;
        width: 22px !important;
        height: 22px !important;
        margin: 0 !important;
    }}
    .st-key-active_docs_bar [data-testid="stHorizontalBlock"] [data-testid="column"] {{
        min-width: 0 !important;
        flex-grow: 0 !important;
        width: auto !important;
    }}
    .st-key-active_docs_bar [class*="st-key-pdf_chip_"] [data-testid="stCaptionContainer"] {{
        display: none;
    }}
    .st-key-active_docs_bar .stButton button {{
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
        box-shadow: none !important;
    }}
    .st-key-active_docs_bar .stButton button:hover {{
        background: rgba(255,255,255,0.10) !important;
        color: var(--danger) !important;
    }}

    /* ── Chat messages ─────────────────────────────────────────── */
    .stChatMessage {{ padding: 10px 0 !important; }}
    [data-testid="stChatMessageContent"] {{
        background: transparent !important;
        border: none !important;
        color: var(--text-primary) !important;
        font-size: 15px !important;
        line-height: 1.7 !important;
    }}
    [data-testid="stChatMessageContent"] p {{ margin-bottom: 0.6em !important; }}
    [data-testid="stChatMessageContent"] code {{
        background: var(--bg-elevated);
        padding: 2px 5px;
        border-radius: 4px;
        font-size: 13px;
    }}
    [data-testid="stChatMessageContent"] pre {{
        background: var(--bg-elevated) !important;
        border-radius: var(--radius-sm) !important;
        border: 1px solid var(--border-color) !important;
    }}
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {{
        display: flex !important;
        justify-content: flex-end !important;
    }}
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {{
        display: flex !important;
        justify-content: flex-start !important;
    }}
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])
    [data-testid="stChatMessageContent"] {{
        max-width: 100%;
        margin-right: auto;
    }}
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
    [data-testid="stChatMessageContent"] {{
        background: var(--bg-input) !important;
        border-radius: var(--radius-lg) !important;
        padding: 10px 15px !important;
        max-width: 70%;
        margin-left: auto !important;
        margin-right: 0 !important;
        line-height: 1.5 !important;
    }}
    [data-testid="stChatMessageAvatarUser"],
    [data-testid="stChatMessageAvatarAssistant"] {{
        display: none !important;
    }}
    .assistant-sources {{
        margin-top: 10px;
        padding: 8px 12px;
        background: var(--bg-elevated);
        border-radius: var(--radius-sm);
        color: var(--text-secondary);
        font-size: 12.5px;
        line-height: 1.5;
    }}
    .assistant-sources strong {{
        color: var(--text-muted);
        font-weight: 600;
        text-transform: uppercase;
        font-size: 11px;
        letter-spacing: 0.03em;
        display: block;
        margin-bottom: 4px;
    }}

    [class*="st-key-msg_actions_"] {{
        opacity: 0;
        transition: opacity 0.12s ease;
        margin-top: 2px;
    }}
    [data-testid="stChatMessage"]:hover [class*="st-key-msg_actions_"] {{ opacity: 1; }}
    @media (hover: none) {{
        [class*="st-key-msg_actions_"] {{ opacity: 1 !important; }}
    }}
    [class*="st-key-msg_actions_"] .stButton button {{
        min-height: 28px !important;
        height: 28px !important;
        padding: 0 10px !important;
        font-size: 12.5px !important;
        white-space: nowrap !important;
        background: transparent !important;
        border: none !important;
        color: var(--text-muted) !important;
        border-radius: var(--radius-sm) !important;
        box-shadow: none !important;
        transition: background 0.12s ease, color 0.12s ease;
    }}
    [class*="st-key-msg_actions_"] .stButton button:hover {{
        background: var(--bg-hover) !important;
        color: var(--text-primary) !important;
    }}
    [class*="st-key-msg_actions_"] iframe {{ display: block; }}

    /* ── Chat input (native st.chat_input) ────────────────────────── */
    [data-testid="stChatInput"] {{
        background: var(--bg-input) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 26px !important;
    }}
    [data-testid="stChatInput"] textarea {{
        color: var(--text-primary) !important;
        /* Streamlit 1.50's chat_input auto-resize misfires on this page
           (renders ~260px tall) unless height is explicitly constrained. */
        height: auto !important;
        max-height: 200px !important;
    }}
    [data-testid="stChatInput"] textarea::placeholder {{
        color: var(--text-muted) !important;
    }}
    [data-testid="stChatInputSubmitButton"] button,
    [data-testid="stChatInputSubmitButton"] {{
        background: var(--text-primary) !important;
        border-radius: 50% !important;
    }}
    [data-testid="stBottomBlockContainer"] {{
        background: var(--bg-main) !important;
    }}

    /* ── Popovers / dialogs ────────────────────────────────────── */
    [class*="st-key-chat_row_menu_"] [data-testid="stPopover"] > button,
    [class*="st-key-chat_row_menu_"] button[data-testid="stPopoverButton"] {{
        min-height: 34px !important;
        height: 34px !important;
        padding: 0 !important;
        background: transparent !important;
        border: none !important;
        color: var(--text-muted) !important;
        box-shadow: none !important;
    }}
    [class*="st-key-chat_row_menu_"] [data-testid="stPopover"] > button:hover {{
        background: var(--bg-hover) !important;
        color: #ffffff !important;
    }}
    div[data-testid="stPopoverBody"] {{
        background: var(--bg-elevated) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: var(--radius-md) !important;
        color: var(--text-primary) !important;
    }}
    div[data-testid="stPopoverBody"] .stTextInput input {{
        background: var(--bg-hover) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
    }}
    div[data-testid="stDialog"] div[role="dialog"] {{
        background: var(--bg-elevated) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: var(--radius-lg) !important;
        color: var(--text-primary) !important;
    }}

    /* ── Stats page ────────────────────────────────────────────── */
    .stats-title {{ font-size: 22px; font-weight: 600; color: var(--text-primary); }}
    .stats-caption {{ color: var(--text-muted); font-size: 13px; margin: 2px 0 10px; }}
    div[data-testid="stMetric"] {{
        background: var(--bg-elevated);
        border: 1px solid var(--border-color);
        border-radius: var(--radius-md);
        padding: 14px 16px !important;
    }}
    div[data-testid="stMetricLabel"] {{ color: var(--text-muted) !important; }}

    /* ── Misc ──────────────────────────────────────────────────── */
    .oauth-notice {{ font-size: 12px; color: var(--text-muted); padding: 6px 8px 0; }}
    .auth-dialog-copy {{
        color: var(--text-secondary);
        font-size: 14px;
        line-height: 1.45;
        margin-bottom: 12px;
    }}
    .account-menu-user {{
        padding: 6px 4px 8px;
        border-bottom: 1px solid var(--border-color);
        margin-bottom: 6px;
    }}
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


def render_delete_chat_dialog() -> None:
    chat_id = st.session_state.confirm_delete_chat_id
    chat = st.session_state.chats.get(chat_id)
    if not chat:
        st.session_state.confirm_delete_chat_id = None
        return

    @st.dialog("Delete chat", width="small")
    def confirm_delete() -> None:
        title = chat.get("title") or "this chat"
        st.markdown(
            f'<div class="auth-dialog-copy">Delete "{html.escape(title)}"? '
            "This can't be undone.</div>",
            unsafe_allow_html=True,
        )
        cancel_col, delete_col = st.columns(2)
        with cancel_col:
            if st.button("Cancel", key="cancel_delete_chat", use_container_width=True):
                st.session_state.confirm_delete_chat_id = None
                st.rerun()
        with delete_col:
            if st.button(
                "Delete",
                key="confirm_delete_chat",
                use_container_width=True,
                type="primary",
            ):
                delete_chat_and_switch(chat_id)
                st.session_state.confirm_delete_chat_id = None
                st.rerun()

    confirm_delete()


def render_chat_row_menu(chat_id: str, current_title: str) -> None:
    with st.popover("⋯", use_container_width=False):
        new_title = st.text_input(
            "Rename chat",
            value=current_title,
            key=f"rename_input_{chat_id}",
        )
        if st.button("Save name", key=f"rename_save_{chat_id}", use_container_width=True):
            clean_title = new_title.strip() or "New chat"
            update_chat_title(chat_id, clean_title, _current_user_id())
            st.session_state.chats[chat_id]["title"] = clean_title
            st.rerun()
        if st.button("Delete chat", key=f"delete_trigger_{chat_id}", use_container_width=True):
            st.session_state.confirm_delete_chat_id = chat_id
            st.rerun()


def render_sidebar() -> None:
    with st.sidebar:
        with st.container(key="sidebar_header_row"):
            logo_col, toggle_col = st.columns([0.78, 0.22])
            with logo_col:
                st.markdown(
                    '<div class="sb-header"><span class="sb-logo">A</span><span>RAG Assistant</span></div>',
                    unsafe_allow_html=True,
                )
            with toggle_col:
                if st.button("«", key="sidebar_collapse_toggle", help="Collapse sidebar"):
                    st.session_state.sidebar_collapsed = True
                    st.rerun()

        if st.button("✎  New chat", key="sidebar_new_chat", use_container_width=True):
            create_new_chat()
            st.rerun()

        stats_active = st.session_state.active_view == "stats"
        with st.container(key="recent_active_chat" if stats_active else "sidebar_stats_row"):
            if st.button("📊  Stats", key="sidebar_stats_nav", use_container_width=True):
                st.session_state.active_view = "stats"
                st.rerun()

        if st.button("⟳  Re-index documents", key="sidebar_reindex_docs", use_container_width=True):
            st.session_state.needs_reindex = True
            st.rerun()

        with st.container(key="sidebar_search"):
            st.session_state.chat_search = st.text_input(
                "Search chats",
                value=st.session_state.chat_search,
                key="chat_search_input",
                placeholder="Search chats",
                label_visibility="collapsed",
            )

        chats_sorted = sorted(
            st.session_state.chats.values(),
            key=lambda c: c.get("created_at", ""),
            reverse=True,
        )

        search_query = st.session_state.chat_search.strip().lower()
        if search_query:
            st.markdown('<div class="sb-section-title">Search results</div>', unsafe_allow_html=True)

        with st.container(key="recents_list"):
            visible_recent_limit = 50 if not search_query else 20
            shown = 0
            title_counts: Dict[str, int] = {}
            current_bucket = None
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
                stored_title = chat.get("title") or "New chat"
                title = (
                    stored_title
                    if stored_title != "New chat"
                    else _chat_title_from_question(first_user)
                )

                if search_query:
                    haystack = " ".join(
                        [title] + [m.get("content", "") for m in chat.get("messages", [])]
                    ).lower()
                    if search_query not in haystack:
                        continue

                title_counts[title] = title_counts.get(title, 0) + 1
                display_title = (
                    f"{title} {title_counts[title]}"
                    if title_counts[title] > 1
                    else title
                )
                if shown >= visible_recent_limit:
                    break

                if not search_query:
                    bucket = _date_bucket(chat.get("created_at", ""))
                    if bucket != current_bucket:
                        st.markdown(
                            f'<div class="sb-date-group">{bucket}</div>',
                            unsafe_allow_html=True,
                        )
                        current_bucket = bucket

                chat_id = chat["id"]
                is_active = (
                    chat_id == st.session_state.current_chat_id
                    and st.session_state.active_view == "chat"
                )
                row_key = "recent_active_chat" if is_active else f"recent_chat_{chat_id}"
                with st.container(key=row_key):
                    switch_col, menu_col = st.columns([0.82, 0.18], gap="small")
                    with switch_col:
                        if st.button(
                            display_title,
                            key=f"switch_chat_{chat_id}",
                            use_container_width=True,
                        ):
                            switch_chat(chat_id)
                            st.rerun()
                    with menu_col:
                        with st.container(key=f"chat_row_menu_{chat_id}"):
                            render_chat_row_menu(chat_id, chat.get("title", display_title))
                shown += 1

            if search_query and shown == 0:
                st.caption("No chats match your search.")

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


def render_top_header() -> None:
    if not st.session_state.sidebar_collapsed:
        return
    with st.container(key="app_header"):
        toggle_col, title_col = st.columns([0.08, 0.92])
        with toggle_col:
            if st.button("☰", key="sidebar_expand_toggle", help="Expand sidebar"):
                st.session_state.sidebar_collapsed = False
                st.rerun()
        with title_col:
            if st.session_state.active_view == "stats":
                title = "Stats"
            else:
                current_chat = st.session_state.chats.get(st.session_state.current_chat_id) or {}
                title = current_chat.get("title") or "New chat"
            st.markdown(
                f'<div class="app-header-title">{html.escape(title)}</div>',
                unsafe_allow_html=True,
            )


STARTER_PROMPTS = [
    "Summarize this document",
    "What are the key takeaways?",
    "List any important dates or deadlines",
    "Who or what is mentioned most?",
]


def render_empty_state() -> None:
    has_docs = bool(st.session_state.uploaded_pdfs)
    subtitle = (
        "Ask anything about your uploaded documents."
        if has_docs
        else "Upload a PDF below and ask anything about it."
    )
    st.markdown(
        f"""
        <div class="empty-state">
            <div class="empty-title">What can I help with?</div>
            <div class="empty-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not has_docs:
        return

    with st.container(key="starter_prompts"):
        cols = st.columns(len(STARTER_PROMPTS), gap="small")
        for index, (col, prompt) in enumerate(zip(cols, STARTER_PROMPTS)):
            with col:
                if st.button(prompt, key=f"starter_{index}", use_container_width=True):
                    st.session_state.pending_user_question = prompt
                    st.rerun()


def render_chat() -> None:
    if not st.session_state.messages:
        render_empty_state()
        return

    last_assistant_index = None
    for index, message in enumerate(st.session_state.messages):
        if message["role"] == "assistant":
            last_assistant_index = index

    for index, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            render_message_content(message["role"], message["content"])
            if message["role"] == "assistant":
                render_message_actions(
                    message["content"],
                    message_index=index,
                    show_regenerate=(index == last_assistant_index),
                )


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
        st.markdown(content)
        return

    parts = _split_answer_sources(content)
    st.markdown(parts["answer"])
    if parts["sources"]:
        sources_html = html.escape(parts["sources"]).replace("\n", "<br>")
        st.markdown(
            f'<div class="assistant-sources"><strong>Sources</strong>{sources_html}</div>',
            unsafe_allow_html=True,
        )


def render_copy_button(text: str, key: str) -> None:
    payload = json.dumps(text)
    components.html(
        f"""
        <style>html, body {{ margin:0; padding:0; background:transparent; }}</style>
        <button id="{key}" title="Copy"
          style="background:transparent;border:none;color:#8b8b8b;cursor:pointer;
                 font-size:14px;padding:3px 8px;border-radius:6px;line-height:1.4;
                 font-family:inherit;">⧉ Copy</button>
        <script>
          const btn = document.getElementById("{key}");
          async function copyText(text) {{
            try {{
              await navigator.clipboard.writeText(text);
              return true;
            }} catch (err) {{
              try {{
                const ta = document.createElement("textarea");
                ta.value = text;
                ta.style.position = "fixed";
                ta.style.opacity = "0";
                document.body.appendChild(ta);
                ta.focus();
                ta.select();
                document.execCommand("copy");
                document.body.removeChild(ta);
                return true;
              }} catch (err2) {{
                return false;
              }}
            }}
          }}
          btn.addEventListener("click", async () => {{
            const ok = await copyText({payload});
            btn.textContent = ok ? "✓ Copied" : "Copy failed";
            btn.style.color = ok ? "#22c55e" : "#ef4444";
            setTimeout(() => {{ btn.textContent = "⧉ Copy"; btn.style.color = "#8b8b8b"; }}, 1500);
          }});
        </script>
        """,
        height=30,
    )


def render_message_actions(content: str, message_index: int, show_regenerate: bool) -> None:
    parts = _split_answer_sources(content)
    with st.container(key=f"msg_actions_{message_index}"):
        cols = st.columns([0.18, 0.32, 0.50] if show_regenerate else [0.18, 0.82])
        with cols[0]:
            render_copy_button(parts["answer"], key=f"copy_btn_{message_index}")
        if show_regenerate:
            with cols[1]:
                if st.button(
                    "↻ Regenerate",
                    key=f"regen_{message_index}",
                    disabled=st.session_state.get("is_responding", False),
                ):
                    regenerate_last_response()
                    st.rerun()


def regenerate_last_response() -> None:
    messages = st.session_state.messages
    last_assistant_pos = None
    for pos in range(len(messages) - 1, -1, -1):
        if messages[pos]["role"] == "assistant":
            last_assistant_pos = pos
            break
    if last_assistant_pos is None:
        return

    last_user_question = None
    for pos in range(last_assistant_pos - 1, -1, -1):
        if messages[pos]["role"] == "user":
            last_user_question = messages[pos]["content"]
            break
    if last_user_question is None:
        return

    removed = messages.pop(last_assistant_pos)
    message_id = removed.get("id")
    if message_id:
        delete_message(message_id, _current_user_id())
    _persist_current_chat()
    st.session_state.regenerate_question = last_user_question


def render_pdf_chips() -> None:
    pdfs = st.session_state.uploaded_pdfs
    if not pdfs:
        return

    with st.container(key="active_docs_bar"):
        chip_cols = st.columns(len(pdfs), gap="small")
        for filename, col in zip(pdfs, chip_cols):
            digest = hashlib.md5(filename.encode("utf-8")).hexdigest()[:10]
            with col:
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


def _stream_and_save_answer(
    question: str,
    previous_messages: List[Dict[str, str]],
) -> None:
    """Stream a fresh assistant answer for `question` into the current chat."""
    active_index_path = st.session_state.chats[st.session_state.current_chat_id].get(
        "active_index_path"
    )
    with st.chat_message("assistant"):
        full_answer = st.write_stream(
            stream_conversational_rag_chain(
                question,
                active_index_path,
                chat_messages=previous_messages,
            )
        )

    saved_assistant_msg = save_message(
        st.session_state.current_chat_id,
        "assistant",
        full_answer,
        _current_user_id(),
    )
    st.session_state.messages.append(saved_assistant_msg)
    _persist_current_chat()


def handle_pending_question() -> bool:
    if st.session_state.get("is_responding"):
        return False

    user_question = st.session_state.get("pending_user_question")
    if not user_question:
        return False

    st.session_state.pending_user_question = None
    if not st.session_state.uploaded_pdfs:
        st.info("Upload at least one PDF to ask questions.")
        return False

    st.session_state.is_responding = True
    try:
        previous_messages = list(st.session_state.messages)
        saved_user_msg = save_message(
            st.session_state.current_chat_id,
            "user",
            user_question,
            _current_user_id(),
        )
        st.session_state.messages.append(saved_user_msg)
        _persist_current_chat()

        render_chat()
        _stream_and_save_answer(user_question, previous_messages)
    finally:
        st.session_state.is_responding = False

    st.rerun()
    return True


def handle_regenerate() -> bool:
    if st.session_state.get("is_responding"):
        return False

    question = st.session_state.get("regenerate_question")
    if not question:
        return False

    st.session_state.regenerate_question = None
    st.session_state.is_responding = True
    try:
        previous_messages = list(st.session_state.messages)
        render_chat()
        _stream_and_save_answer(question, previous_messages)
    finally:
        st.session_state.is_responding = False

    st.rerun()
    return True


def render_input_area():
    """Render the chat_input widget and return its raw submission (or None).

    Must be called before any `unsafe_allow_html` markup is injected earlier
    in the script — Streamlit's chat_input auto-resize measurement breaks
    (renders ~260px tall) if raw HTML was rendered before it on the same run.
    """
    return st.chat_input(
        "Message RAG Assistant…",
        key="chat_input_main",
        accept_file="multiple",
        file_type=["pdf"],
        disabled=st.session_state.get("is_responding", False),
    )


def handle_input_submission(submission) -> None:
    if not submission:
        return

    files = list(submission.files or [])
    text = (submission.text or "").strip()

    reindex_triggered = False
    if files:
        reindex_triggered = save_uploaded_files(files)

    if (
        text
        and not st.session_state.get("is_responding")
        and not st.session_state.get("pending_user_question")
    ):
        st.session_state.pending_user_question = text
        st.rerun()
    elif reindex_triggered:
        st.rerun()


@st.cache_data(show_spinner=False)
def _cached_trace_stats(mtime: float, size: int) -> Dict[str, Any]:
    """Cached on the trace file's (mtime, size) so it only re-parses on change."""
    return aggregate_usage_stats(read_trace_events())


def _format_percent(value: Optional[float]) -> str:
    return f"{value * 100:.1f}%" if value is not None else "—"


def _format_seconds(value: Optional[float]) -> str:
    return f"{value:.2f}s" if value is not None else "—"


def render_stats_view() -> None:
    back_col, title_col = st.columns([0.15, 0.85])
    with back_col:
        if st.button("← Back", key="stats_back", use_container_width=True):
            st.session_state.active_view = "chat"
            st.rerun()
    with title_col:
        st.markdown('<div class="stats-title">Stats</div>', unsafe_allow_html=True)

    user_id = _current_user_id()
    usage_counts = get_usage_counts(user_id)
    activity = get_message_activity(user_id, days=30)
    mtime, size = trace_file_fingerprint()
    usage_stats = _cached_trace_stats(mtime, size)

    st.markdown('<div class="sb-section-title">Usage</div>', unsafe_allow_html=True)
    k1, k2, k3 = st.columns(3)
    k1.metric("Chats", usage_counts["chat_count"])
    k2.metric("Messages", usage_counts["message_count"])
    k3.metric("Documents indexed", usage_counts["document_count"])

    st.markdown('<div class="sb-section-title">Performance &amp; cost</div>', unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    p1.metric("Traced queries", usage_stats["total_queries"])
    p2.metric("Avg response time", _format_seconds(usage_stats["avg_latency_seconds"]))
    cost = usage_stats["estimated_cost_usd"]
    p3.metric("Estimated cost to date", f"${cost:.4f}" if cost is not None else "—")

    if activity:
        activity_df = pd.DataFrame(activity).set_index("date")
        st.markdown(
            '<div class="sb-section-title">Messages per day (last 30 days)</div>',
            unsafe_allow_html=True,
        )
        st.bar_chart(activity_df["count"], height=220)
    else:
        st.markdown('<div class="stats-caption">No message activity yet.</div>', unsafe_allow_html=True)

    daily_series = usage_stats["daily_series"]
    if daily_series:
        cost_df = pd.DataFrame(daily_series).set_index("date")
        st.markdown(
            '<div class="sb-section-title">Estimated cost per day</div>',
            unsafe_allow_html=True,
        )
        st.line_chart(cost_df["cost_usd"], height=220)

    st.markdown('<div class="sb-section-title">Retrieval quality</div>', unsafe_allow_html=True)
    q1, q2 = st.columns(2)
    q1.metric(
        "Fallback rate",
        _format_percent(usage_stats["fallback_rate"]),
        help="Share of answers where the assistant couldn't find the info in your documents.",
    )
    avg_rerank = usage_stats["avg_rerank_score"]
    q2.metric("Avg rerank score", f"{avg_rerank:.2f}" if avg_rerank is not None else "—")

    st.markdown('<div class="sb-section-title">Evaluation snapshot</div>', unsafe_allow_html=True)
    summary = read_eval_summary()
    if not summary:
        st.markdown(
            '<div class="stats-caption">No evaluation run yet. '
            'Run <code>python -m evaluation.run_eval</code> to generate one.</div>',
            unsafe_allow_html=True,
        )
    else:
        metrics = summary.get("metrics", {})
        e1, e2, e3 = st.columns(3)
        e1.metric("Citation coverage", _format_percent(metrics.get("citation_coverage")))
        e2.metric("Source-match accuracy", _format_percent(metrics.get("source_match_accuracy")))
        e3.metric("Failure rate", _format_percent(metrics.get("failure_rate")))
        e4, e5 = st.columns(2)
        e4.metric("Avg eval latency", _format_seconds(metrics.get("average_latency_seconds")))
        avg_words = metrics.get("average_answer_length_words")
        e5.metric("Avg answer length", f"{avg_words:.1f} words" if avg_words is not None else "—")

        generated_at = summary.get("_generated_at", "")
        display_time = generated_at[:19].replace("T", " ") if generated_at else "unknown"
        st.markdown(
            f'<div class="stats-caption">As of last eval run: {html.escape(display_time)} · '
            "re-run <code>python -m evaluation.run_eval</code> to refresh.</div>",
            unsafe_allow_html=True,
        )


def main() -> None:
    init_session_state()

    # chat_input must be the first widget rendered — see render_input_area's
    # docstring for why. It's a no-op on the stats page (nothing to submit).
    show_composer = st.session_state.active_view != "stats"
    submission = render_input_area() if show_composer else None

    st.markdown(build_custom_css(st.session_state.sidebar_collapsed), unsafe_allow_html=True)
    render_sidebar()
    if st.session_state.show_auth_dialog:
        render_sign_in_dialog()
    if st.session_state.confirm_delete_chat_id:
        render_delete_chat_dialog()
    if st.session_state.needs_reindex:
        safe_reindex()

    render_top_header()

    if st.session_state.active_view == "stats":
        render_stats_view()
        return

    render_pdf_chips()
    if not handle_pending_question() and not handle_regenerate():
        render_chat()
    handle_input_submission(submission)


if __name__ == "__main__":
    main()
