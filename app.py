from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

from rag.chat_store import ensure_title, list_chats, load_chat, new_chat_id, save_chat
from rag.ingest import ingest_file
from rag.paths import uploads_dir
from rag.retrieve import retrieve
from rag.llm import DEFAULT_GROQ_MODEL, generate_answer
from rag.store import delete_by_source_file

from dotenv import load_dotenv

load_dotenv()


st.set_page_config(page_title="AI Chatbot (RAG)", layout="wide")


def _format_source(meta: Dict[str, Any]) -> str:
    f = meta.get("source_file", "unknown")
    if meta.get("type") == "pdf" and meta.get("page"):
        return f"{f}, Page {meta.get('page')}"
    return str(f)


def _load_uploaded_files() -> List[str]:
    p = uploads_dir()
    if not p.exists():
        return []
    return sorted([x.name for x in p.iterdir() if x.is_file()])


def sidebar_ui() -> Dict[str, Any]:
    st.sidebar.markdown("### Upload Documents")
    uploads = st.sidebar.file_uploader(
        "Upload Files",
        type=["pdf", "docx", "csv"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if st.sidebar.button("Upload Files", use_container_width=True, disabled=not uploads):
        saved = 0
        with st.sidebar:
            with st.spinner("Indexing..."):
                for up in uploads or []:
                    dst = uploads_dir() / up.name
                    dst.write_bytes(up.getbuffer())
                    ingest_file(dst)
                    saved += 1
        st.sidebar.success(f"Indexed {saved} file(s).")

    st.sidebar.markdown("### Uploaded Files")
    files = _load_uploaded_files()
    if not files:
        st.sidebar.caption("No files uploaded yet.")
    else:
        for fn in files:
            c1, c2 = st.sidebar.columns([0.82, 0.18])
            c1.write(fn)
            if c2.button("🗑️", key=f"del_{fn}"):
                try:
                    (uploads_dir() / fn).unlink(missing_ok=True)
                except Exception:
                    pass
                try:
                    delete_by_source_file(fn)
                except Exception:
                    pass
                st.rerun()

    st.sidebar.markdown("### Chat History")
    selected_chat_id = None

    if st.sidebar.button("➕ New chat", use_container_width=True):
        selected_chat_id = new_chat_id()
        # Create the chat file immediately so it shows up in history
        save_chat({"chat_id": selected_chat_id, "title": "New Chat", "messages": []})
        st.rerun()

    chats = list_chats()
    if not chats:
        st.sidebar.caption("No chats yet.")
    else:
        # GPT-style: headings in a list; content hidden until clicked.
        for c in chats:
            cid = c["chat_id"]
            title = c["title"]
            is_current = ("chat_id" in st.session_state) and (st.session_state.chat_id == cid)
            header = f"{title}{' (current)' if is_current else ''}"

            with st.sidebar.expander(header, expanded=False):
                col1, col2 = st.columns([0.7, 0.3])
                if col1.button("Open", key=f"open_{cid}", use_container_width=True):
                    selected_chat_id = cid
                    st.session_state.chat_id = cid
                    st.rerun()
                if col2.button("🗑️", key=f"del_chat_{cid}", use_container_width=True):
                    # Delete chat file (best-effort)
                    try:
                        from rag.paths import chats_dir

                        (chats_dir() / f"{cid}.json").unlink(missing_ok=True)
                    except Exception:
                        pass
                    # If we deleted the current chat, start a new one
                    if is_current:
                        st.session_state.chat_id = new_chat_id()
                        save_chat({"chat_id": st.session_state.chat_id, "title": "New Chat", "messages": []})
                    st.rerun()

                # Show a quick preview of the last messages (hidden until expanded)
                past = load_chat(cid)
                msgs = past.get("messages", [])
                if not msgs:
                    st.caption("No messages yet.")
                else:
                    preview = msgs[-6:]
                    st.caption("Click an item to open this chat:")
                    for j, m in enumerate(preview):
                        role = m.get("role", "")
                        content = (m.get("content") or "").strip()
                        if not content:
                            continue
                        label = "You" if role == "user" else "Assistant"
                        snippet = content[:140] + ("…" if len(content) > 140 else "")
                        if st.button(
                            f"{label}: {snippet}",
                            key=f"pv_{cid}_{j}",
                            use_container_width=True,
                        ):
                            selected_chat_id = cid
                            st.session_state.chat_id = cid
                            st.rerun()

    st.sidebar.markdown("---")
    model = st.sidebar.text_input("Groq model", value=DEFAULT_GROQ_MODEL)
    top_k = st.sidebar.slider("Top-K chunks", min_value=3, max_value=12, value=6, step=1)

    return {"uploads": uploads, "selected_chat_id": selected_chat_id, "model": model, "top_k": top_k}


def main():
    ui = sidebar_ui()

    st.markdown("### AI Chatbot (RAG)")
    st.caption("Ask questions based on your documents.")

    if "chat_id" not in st.session_state:
        st.session_state.chat_id = new_chat_id()

    if ui["selected_chat_id"] and ui["selected_chat_id"] != st.session_state.chat_id:
        st.session_state.chat_id = ui["selected_chat_id"]

    chat = load_chat(st.session_state.chat_id)
    messages = chat.get("messages", [])

    # Render messages
    for m in messages:
        with st.chat_message(m.get("role", "assistant")):
            st.write(m.get("content", ""))
            srcs = m.get("sources") or []
            if srcs:
                st.markdown("**Source**")
                for s in srcs:
                    st.caption(str(s))

    prompt = st.chat_input("Type your question…")
    if not prompt:
        return

    # Append user message
    messages.append({"role": "user", "content": prompt})
    chat["messages"] = messages
    ensure_title(chat)
    save_chat(chat)

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        status = st.status("Searching documents...", expanded=False)
        chunks = retrieve(question=prompt, top_k=int(ui["top_k"]))
        status.update(label="Generating answer...", state="running")

        context_texts = [c.text for c in chunks]
        ans = generate_answer(question=prompt, context_chunks=context_texts, model=str(ui["model"]))

        # Build simple sources list
        sources = []
        for c in chunks[: min(4, len(chunks))]:
            meta = c.metadata or {}
            sources.append(
                {
                    "source": _format_source(meta),
                    "chunk_index": meta.get("chunk_index"),
                }
            )

        status.update(state="complete")
        st.write(ans.answer or "I couldn't generate an answer.")
        if sources:
            st.markdown("**Source**")
            for s in sources:
                st.caption(f"{s['source']} (chunk {s.get('chunk_index')})")

    # Persist assistant message
    chat = load_chat(st.session_state.chat_id)
    messages = chat.get("messages", [])
    messages.append({"role": "assistant", "content": ans.answer, "sources": sources})
    chat["messages"] = messages
    ensure_title(chat)
    save_chat(chat)


if __name__ == "__main__":
    main()

