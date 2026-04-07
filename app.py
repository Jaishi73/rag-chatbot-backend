from __future__ import annotations

import streamlit as st
from typing import Any, Dict, List
from rag.llm import generate_chat_title
from rag.chat_store import ensure_title, list_chats, load_chat, new_chat_id, save_chat
from rag.ingest import ingest_file
from rag.paths import uploads_dir
from rag.retrieve import retrieve, _cached_retrieval
from rag.llm import DEFAULT_GROQ_MODEL, generate_answer, _cached_llm
from rag.store import delete_by_source_file
from rag.semantic_cache import _CACHE
import time
import traceback
from dotenv import load_dotenv
from rag.logger import get_logger
logger = get_logger()

load_dotenv()

st.set_page_config(page_title="AI Chatbot (RAG)", layout="wide")

import sys

class StreamToLogger:
    def __init__(self, logger):
        self.logger = logger

    def write(self, message):
        message = message.strip()
        if message:
            self.logger.info(message)

    def flush(self):
        pass


# 🔥 Redirect print → logger
sys.stdout = StreamToLogger(logger)
sys.stderr = StreamToLogger(logger)
# -------------------------------
# 🎨 Styling (ChatGPT-like)
# -------------------------------

st.markdown("""
<style>

div[data-testid="stVerticalBlock"] {
    gap: 0rem !important;
}

div[data-testid="element-container"] {
    margin-bottom: 0px !important;
}


div[data-testid="column"] {
    padding: 0px !important;
    gap: 0px !important;
}


.stButton {
    width: 100% !important;
}


.stButton > button {
    width: 100% !important;
    display: flex !important;
    align-items: center;
    justify-content: flex-start;

    height: 36px !important;
    padding: 6px 10px !important;

    border: none;
    background: transparent;
    border-radius: 6px;
}


.stButton > button div {
    width: 100% !important;
    display: block !important;
}


.stButton > button:hover {
    background-color: #d0e9f7 !important;
    color: black !important;
}


.delete-btn > button {
    width: 100% !important;
    height: 36px !important;
    padding: 0px !important;
    text-align: center !important;
    color: #ff4b4b;
}

.delete-btn > button:hover {
    background-color: #F27405 !important;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# Helpers
# -------------------------------
def _format_source(meta: Dict[str, Any]) -> str:
    f = meta.get("source_file", "unknown")
    if meta.get("type") == "pdf" and meta.get("page"):
        return f"{f}, Page {meta.get('page')}"
    return str(f)


def _load_uploaded_files() -> List[str]:
    p = uploads_dir()
    return sorted([x.name for x in p.iterdir() if x.is_file()]) if p.exists() else []


# ✅ ADD IT HERE
def shorten(text, max_len=35):
    return text[:max_len] + "..." if len(text) > max_len else text

# -------------------------------
# Sidebar UI
# -------------------------------
def sidebar_ui():

    # -------------------------------
    # Upload Section (UNCHANGED)
    # -------------------------------
    st.sidebar.markdown("### 📤 Upload Documents")

    uploads = st.sidebar.file_uploader(
        "Upload",
        type=["pdf", "docx", "csv"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if st.sidebar.button("⬆️ Upload Files", use_container_width=True, disabled=not uploads):
        with st.sidebar:
            with st.spinner("Indexing..."):
                for up in uploads or []:
                    dst = uploads_dir() / up.name
                    dst.write_bytes(up.getbuffer())
                    ingest_file(dst)
        st.sidebar.success("Files indexed")

    # -------------------------------
    # Uploaded Files
    # -------------------------------
    st.sidebar.markdown("### 📂 Files")
    files = _load_uploaded_files()

    if not files:
        st.sidebar.caption("No files uploaded")
    else:
        for fn in files:
            c1, c2 = st.sidebar.columns([0.85, 0.15])
            c1.markdown(f"📄 {fn}")

            if c2.button("✕", key=f"del_file_{fn}"):
                try:
                    (uploads_dir() / fn).unlink(missing_ok=True)
                    delete_by_source_file(fn)

                    # Clear cache when data changes
                    _CACHE.clear()
                    _cached_retrieval.cache_clear()
                    _cached_llm.cache_clear()

                except Exception:
                    pass

                st.rerun()

    st.sidebar.markdown("---")

    # -------------------------------
    # 💬 Chat Panel
    # -------------------------------
    st.sidebar.markdown("## 💬 Chats")

    # New chat
    if st.sidebar.button("➕ New Chat", use_container_width=True):
        new_id = new_chat_id()
        st.session_state.chat_id = new_id
        save_chat({"chat_id": new_id, "title": "New Chat", "messages": []})
        st.rerun()

    # Clear current chat
    if st.sidebar.button("🧹 Clear Current Chat", use_container_width=True):
        chat = load_chat(st.session_state.chat_id)
        chat["messages"] = []
        chat["title"] = "New Chat"
        save_chat(chat)
        st.rerun()

    st.sidebar.markdown("---")

    chats = list_chats()

    if chats:
        st.sidebar.markdown(
            "<span style='color:gray;'>Your conversations</span>",
            unsafe_allow_html=True
        )

    # -------------------------------
    # Chat List (NO RENAME)
    # -------------------------------
    for c in chats:
        cid = c["chat_id"]
        title = shorten(c["title"])

        is_current = (
            "chat_id" in st.session_state and
            st.session_state.chat_id == cid
        )

        display_title = f"🔵  {shorten(title)}" if is_current else shorten(title)

        row = st.sidebar.container()
        col1, col2 = row.columns([0.7, 0.2])

        # Open chat
        if col1.button(display_title, key=f"chat_{cid}"):
            st.session_state.chat_id = cid
            st.rerun()

        # Delete chat
        with col2:
            st.markdown('<div class="delete-btn">', unsafe_allow_html=True)
            if st.button("✕", key=f"del_chat_{cid}"):
                from rag.paths import chats_dir
                (chats_dir() / f"{cid}.json").unlink(missing_ok=True)

                if is_current:
                    new_id = new_chat_id()
                    st.session_state.chat_id = new_id
                    save_chat({
                        "chat_id": new_id,
                        "title": "New Chat",
                        "messages": []
                    })

                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    st.sidebar.markdown("---")

    model = st.sidebar.text_input("Model", value=DEFAULT_GROQ_MODEL)
    top_k = st.sidebar.slider("Top-K", 3, 12, 6)

    return {"model": model, "top_k": top_k}



# -------------------------------
# Main App
# -------------------------------
def main():
    ui = sidebar_ui()

    st.markdown("### 🤖 AI Chatbot (RAG)")
    st.caption("Ask questions based on your documents")

    # Initialize chat
    if "chat_id" not in st.session_state:
        st.session_state.chat_id = new_chat_id()

    chat = load_chat(st.session_state.chat_id)
    messages = chat.get("messages", [])

    # -------------------------------
    # ✅ DISPLAY CHAT (FIXED)
    # -------------------------------
    for m in messages:
        role = m.get("role", "assistant")

        with st.chat_message(role):
            st.write(m.get("content", ""))

              # 🔥 SHOW CACHE + TIME FROM HISTORY
            if role == "assistant":
                cache_status = m.get("cache_status")
                response_time = m.get("response_time")
                
                if cache_status and response_time is not None:
                    if cache_status == "SEMANTIC_HIT":
                        st.caption(f"⚡ Semantic HIT ({response_time} sec)")
                    elif cache_status == "LLM_HIT":
                        st.caption(f"⚡ LLM HIT ({response_time} sec)")
                    elif cache_status == "MISS":
                        st.caption(f"🐢 MISS ({response_time} sec)")
                    elif cache_status == "NO_CONTEXT":
                        st.caption(f"🚫 No Context ({response_time} sec)")

            if role == "assistant" and m.get("sources"):
                st.markdown("**Sources**")
                for s in m["sources"]:
                    st.caption(str(s))

    # -------------------------------
    # INPUT
    # -------------------------------
    prompt = st.chat_input("Type your question...")
    if not prompt:
        return
        logger.info(f"User question: {prompt}")

    # -------------------------------
    # SAVE USER MESSAGE
    # -------------------------------
    chat["messages"].append({
        "role": "user",
        "content": prompt
    })

    # 🔥 AUTO TITLE (ONLY FIRST MESSAGE)
    if len(chat["messages"]) == 1:
        try:
            title = generate_chat_title(prompt)
            chat["title"] = title
        except Exception:
            chat["title"] = prompt[:40]

    save_chat(chat)

    # Show user message instantly
    with st.chat_message("user"):
        st.write(prompt)

    # -------------------------------
    # GENERATE ANSWER
    # -------------------------------
    with st.chat_message("assistant"):
        # logger.info(f"User question: {prompt}")

        # start_time = time.time()

        # status = st.status("Searching documents...", expanded=False)

        # chunks = retrieve(
        #     question=prompt,
        #     top_k=int(ui["top_k"])
        # )

        # status.update(label="Generating answer...", state="running")

        # context_texts = [c.text for c in chunks]

        # ans = generate_answer(
        #     question=prompt,
        #     context_chunks=context_texts,
        #     model=ui["model"]
        # )

        # end_time = time.time()
        # response_time = round(end_time - start_time, 2)
        # logger.info(f"Response time: {response_time}s")
        # status.update(state="complete")

        import traceback
        logger.info(f"User question: {prompt}")
        start_time = time.time()
        try:
            logger.info("Retrieval started")
            chunks = retrieve(
                question=prompt,
                top_k=int(ui["top_k"])
                )
            retrieval_time = round(time.time() - start_time, 3)
            logger.info(f"Retrieved {len(chunks)} chunks")
            logger.info(f"Retrieval time: {retrieval_time}s")

            llm_start = time.time()
            context_texts = [c.text for c in chunks]
            ans = generate_answer(
                question=prompt,
                context_chunks=context_texts,
                model=ui["model"]
                )
            llm_time = round(time.time() - llm_start, 3)
            logger.info(f"LLM time: {llm_time}s")
            
        except Exception as e:
            logger.error("Error during pipeline")
            logger.error(str(e))
            logger.error(traceback.format_exc())
            
            ans = type("obj", (), {
                "answer": "Something went wrong. Check logs.",
                "cache_status": "ERROR"
            })()
            
        end_time = time.time()
        response_time = round(end_time - start_time, 2)
        logger.info(f"Response time: {response_time}s")


        # -------------------------------
        # SHOW ANSWER
        # -------------------------------
        st.write(ans.answer or "No answer found")

        # 🔥 CACHE + TIME
        cache_status = getattr(ans, "cache_status", "MISS")

        if cache_status == "SEMANTIC_HIT":
            st.caption(f"⚡ Semantic HIT ({response_time} sec)")
        elif cache_status == "LLM_HIT":
            st.caption(f"⚡ LLM HIT ({response_time} sec)")
        elif cache_status == "MISS":
            st.caption(f"🐢 MISS ({response_time} sec)")
        elif cache_status == "NO_CONTEXT":
            st.caption(f"🚫 No Context ({response_time} sec)")
        else:
            st.caption(f"⏱ {response_time} sec")

        # -------------------------------
        # SOURCES
        # -------------------------------
        sources = []
        for c in chunks[:4]:
            src = {
                "source": _format_source(c.metadata),
                "chunk_index": c.metadata.get("chunk_index"),
            }
            sources.append(src)

        if sources:
            st.markdown("**Sources**")
            for s in sources:
                st.caption(f"{s['source']} (chunk {s['chunk_index']})")

    # -------------------------------
    # SAVE ASSISTANT MESSAGE (FIXED)
    # -------------------------------
    chat["messages"].append({
        "role": "assistant",
        "content": ans.answer,
        "sources": sources,
        "cache_status": cache_status,
        "response_time": response_time 
    })

    save_chat(chat)


if __name__ == "__main__":
    main()