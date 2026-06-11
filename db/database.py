import os
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.path.join(BASE_DIR, "db", "chat_history.db")
LOCAL_USER_ID = "local"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT UNIQUE,
                name TEXT,
                avatar_url TEXT,
                created_at TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT OR IGNORE INTO users (id, email, name, avatar_url, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (LOCAL_USER_ID, "local@session", "Local session", None, _now()),
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chats (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                title TEXT,
                created_at TEXT,
                updated_at TEXT,
                active_index_path TEXT
            )
            """
        )
        columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(chats)").fetchall()
        }
        if "active_index_path" not in columns:
            conn.execute("ALTER TABLE chats ADD COLUMN active_index_path TEXT")
        if "user_id" not in columns:
            conn.execute("ALTER TABLE chats ADD COLUMN user_id TEXT")
        conn.execute(
            "UPDATE chats SET user_id = ? WHERE user_id IS NULL",
            (LOCAL_USER_ID,),
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                chat_id TEXT,
                role TEXT,
                content TEXT,
                created_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                chat_id TEXT,
                filename TEXT,
                file_path TEXT,
                uploaded_at TEXT
            )
            """
        )


def upsert_user(
    email: str,
    name: str,
    avatar_url: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Dict[str, str]:
    created_at = _now()
    with _connect() as conn:
        existing = conn.execute(
            "SELECT id, email, name, avatar_url, created_at FROM users WHERE email = ?",
            (email,),
        ).fetchone()
        if existing:
            conn.execute(
                "UPDATE users SET name = ?, avatar_url = ? WHERE id = ?",
                (name, avatar_url, existing["id"]),
            )
            return {
                **dict(existing),
                "name": name,
                "avatar_url": avatar_url,
            }

        new_user_id = user_id or uuid.uuid4().hex
        conn.execute(
            """
            INSERT INTO users (id, email, name, avatar_url, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (new_user_id, email, name, avatar_url, created_at),
        )
    return {
        "id": new_user_id,
        "email": email,
        "name": name,
        "avatar_url": avatar_url,
        "created_at": created_at,
    }


def create_chat(title: str, user_id: str = LOCAL_USER_ID) -> Dict[str, str]:
    chat_id = uuid.uuid4().hex[:12]
    created_at = _now()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO chats (
                id, user_id, title, created_at, updated_at, active_index_path
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (chat_id, user_id, title, created_at, created_at, None),
        )
    return {
        "id": chat_id,
        "user_id": user_id,
        "title": title,
        "created_at": created_at,
        "updated_at": created_at,
        "active_index_path": None,
    }


def get_chats(user_id: str = LOCAL_USER_ID) -> List[Dict[str, str]]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, user_id, title, created_at, updated_at, active_index_path
            FROM chats
            WHERE user_id = ?
            ORDER BY updated_at DESC
            """,
            (user_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def save_message(
    chat_id: str,
    role: str,
    content: str,
    user_id: str = LOCAL_USER_ID,
) -> Dict[str, str]:
    message_id = uuid.uuid4().hex
    created_at = _now()
    with _connect() as conn:
        chat = conn.execute(
            "SELECT id FROM chats WHERE id = ? AND user_id = ?",
            (chat_id, user_id),
        ).fetchone()
        if not chat:
            raise ValueError("Chat not found for user")
        conn.execute(
            """
            INSERT INTO messages (id, chat_id, role, content, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (message_id, chat_id, role, content, created_at),
        )
        conn.execute(
            "UPDATE chats SET updated_at = ? WHERE id = ? AND user_id = ?",
            (created_at, chat_id, user_id),
        )
    return {
        "id": message_id,
        "chat_id": chat_id,
        "role": role,
        "content": content,
        "created_at": created_at,
    }


def get_messages(
    chat_id: str,
    user_id: str = LOCAL_USER_ID,
) -> List[Dict[str, str]]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT messages.id, messages.chat_id, messages.role,
                   messages.content, messages.created_at
            FROM messages
            JOIN chats ON chats.id = messages.chat_id
            WHERE messages.chat_id = ? AND chats.user_id = ?
            ORDER BY messages.created_at ASC
            """,
            (chat_id, user_id),
        ).fetchall()
    return [dict(row) for row in rows]


def update_chat_title(
    chat_id: str,
    title: str,
    user_id: str = LOCAL_USER_ID,
) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE chats SET title = ?, updated_at = ? WHERE id = ? AND user_id = ?",
            (title, _now(), chat_id, user_id),
        )


def update_chat_active_index_path(
    chat_id: str,
    active_index_path: Optional[str],
    user_id: str = LOCAL_USER_ID,
) -> None:
    with _connect() as conn:
        conn.execute(
            """
            UPDATE chats
            SET active_index_path = ?, updated_at = ?
            WHERE id = ? AND user_id = ?
            """,
            (active_index_path, _now(), chat_id, user_id),
        )


def delete_chat(chat_id: str, user_id: str = LOCAL_USER_ID) -> None:
    with _connect() as conn:
        chat = conn.execute(
            "SELECT id FROM chats WHERE id = ? AND user_id = ?",
            (chat_id, user_id),
        ).fetchone()
        if not chat:
            return
        conn.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
        conn.execute("DELETE FROM documents WHERE chat_id = ?", (chat_id,))
        conn.execute(
            "DELETE FROM chats WHERE id = ? AND user_id = ?",
            (chat_id, user_id),
        )


def add_document(
    chat_id: str,
    filename: str,
    file_path: str,
    user_id: str = LOCAL_USER_ID,
) -> Dict[str, str]:
    document_id = uuid.uuid4().hex
    uploaded_at = _now()
    with _connect() as conn:
        chat = conn.execute(
            "SELECT id FROM chats WHERE id = ? AND user_id = ?",
            (chat_id, user_id),
        ).fetchone()
        if not chat:
            raise ValueError("Chat not found for user")
        conn.execute(
            """
            INSERT INTO documents (id, chat_id, filename, file_path, uploaded_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (document_id, chat_id, filename, file_path, uploaded_at),
        )
    return {
        "id": document_id,
        "chat_id": chat_id,
        "filename": filename,
        "file_path": file_path,
        "uploaded_at": uploaded_at,
    }


def get_documents(
    chat_id: str,
    user_id: str = LOCAL_USER_ID,
) -> List[Dict[str, str]]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT documents.id, documents.chat_id, documents.filename,
                   documents.file_path, documents.uploaded_at
            FROM documents
            JOIN chats ON chats.id = documents.chat_id
            WHERE documents.chat_id = ? AND chats.user_id = ?
            ORDER BY documents.uploaded_at ASC
            """,
            (chat_id, user_id),
        ).fetchall()
    return [dict(row) for row in rows]


def remove_document(document_id: str, user_id: str = LOCAL_USER_ID) -> None:
    with _connect() as conn:
        conn.execute(
            """
            DELETE FROM documents
            WHERE id = ?
            AND chat_id IN (SELECT id FROM chats WHERE user_id = ?)
            """,
            (document_id, user_id),
        )


def remove_documents_for_chat(chat_id: str, user_id: str = LOCAL_USER_ID) -> None:
    with _connect() as conn:
        conn.execute(
            """
            DELETE FROM documents
            WHERE chat_id = ?
            AND chat_id IN (SELECT id FROM chats WHERE user_id = ?)
            """,
            (chat_id, user_id),
        )
