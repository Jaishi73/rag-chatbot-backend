from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from rag.paths import chats_dir


@dataclass
class ChatMessage:
    role: str  # "user" | "assistant"
    content: str
    created_at: float
    sources: Optional[List[Dict[str, Any]]] = None


def _chat_path(chat_id: str) -> Path:
    return chats_dir() / f"{chat_id}.json"


def new_chat_id() -> str:
    return uuid.uuid4().hex


def list_chats() -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for p in sorted(chats_dir().glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            items.append(
                {
                    "chat_id": data.get("chat_id") or p.stem,
                    "title": data.get("title") or "Untitled",
                    "updated_at": data.get("updated_at") or p.stat().st_mtime,
                }
            )
        except Exception:
            continue
    return items


def load_chat(chat_id: str) -> Dict[str, Any]:
    p = _chat_path(chat_id)
    if not p.exists():
        return {"chat_id": chat_id, "title": "New Chat", "messages": [], "updated_at": time.time()}
    return json.loads(p.read_text(encoding="utf-8"))


def save_chat(chat: Dict[str, Any]) -> None:
    chat["updated_at"] = time.time()
    p = _chat_path(str(chat["chat_id"]))
    p.write_text(json.dumps(chat, ensure_ascii=False, indent=2), encoding="utf-8")


def ensure_title(chat: Dict[str, Any]) -> None:
    title = (chat.get("title") or "").strip()

    if title and title != "New Chat":
        return
 
    for m in chat.get("messages", []):
        if m.get("role") == "user":
            t = (m.get("content") or "").strip()
            if t:
                chat["title"] = (t[:48] + "…") if len(t) > 49 else t
            return

def delete_chat(chat_id: str):
    p=_chat_path(chat_id)

    if p.exists():
        p.unlink()

# def clear_all_chats():
#     for p in chats_dir().glob(".json"):
#         p.unlink

def clear_all_chats():
    dir_path = chats_dir()

    print("🔥 Clearing chats from:", dir_path)

    files = list(dir_path.glob("*.json"))   # ✅ FIX HERE
    print("Files found:", files)

    for p in files:
        print("Deleting:", p)
        p.unlink()