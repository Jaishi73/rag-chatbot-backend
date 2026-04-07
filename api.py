from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel,Field
from rag.chat_store import list_chats, load_chat
import time
import traceback
from rag.llm import generate_chat_title
from rag.store import delete_by_source_file
from rag.semantic_cache import _CACHE
from rag.llm import _cached_llm
from rag.retrieve import _cached_retrieval
from fastapi import FastAPI
from dotenv import load_dotenv
import os
from rag.chat_store import delete_chat
from rag.chat_store import clear_all_chats
from rag.retrieve import retrieve
from rag.llm import generate_answer
from rag.ingest import ingest_file
from rag.paths import uploads_dir
from rag.chat_store import (
    new_chat_id,
    load_chat,
    save_chat,
    ensure_title
)
from rag.logger import get_logger

logger = get_logger()
load_dotenv() 
app = FastAPI()

# -------------------------------
# 🔥 CORS (Angular connection)
# -------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Request Models
# -------------------------------
class QueryRequest(BaseModel):
    question: str
    chat_id: str | None = None
    top_k: int = 6
    model: str = "llama-3.1-8b-instant"


# -------------------------------
# 🔹 HEALTH CHECK
# -------------------------------




@app.get("/")
def root():
    return {"status": "API running"}

# -------------------------------
# 🔹 ASK QUESTION (MAIN API)
# -------------------------------
@app.post("/ask")
def ask(req: QueryRequest):

    start_time = time.time()

    chat_id = req.chat_id or new_chat_id()
    chat = load_chat(chat_id)

    logger.info(f"User question: {req.question}")

    try:
        # -------------------------------
        # RETRIEVAL
        # -------------------------------
        logger.info("Retrieval started")
        chunks = retrieve(
            question=req.question,
            top_k=req.top_k
        )

        logger.info(f"Retrieved {len(chunks)} chunks")

        # -------------------------------
        # LLM
        # -------------------------------
        context_texts = [c.text for c in chunks]

        ans = generate_answer(
            question=req.question,
            context_chunks=context_texts,
            model=req.model
        )

    except Exception as e:
        logger.error("Pipeline error")
        logger.error(str(e))
        logger.error(traceback.format_exc())

        if ans.cache_status != "NO_CONTEXT":
            store(req.question, ans.answer)

        return {
            "answer": "Something went wrong",
            "sources": [],
            "cache_status": "ERROR"
        }

    response_time = round(time.time() - start_time, 2)
    logger.info(f"Response time: {response_time}s")

    # -------------------------------
    # SOURCES FORMAT
    # -------------------------------
    sources = []
    for c in chunks[:4]:
        sources.append({
            "source": c.metadata.get("source_file"),
            "chunk_index": c.metadata.get("chunk_index")
        })

    # -------------------------------
    # SAVE CHAT
    # -------------------------------
    chat["messages"].append({
        "role": "user",
        "content": req.question
    })

    chat["messages"].append({
        "role": "assistant",
        "content": ans.answer,
        "sources": sources,
        "cache_status": ans.cache_status,
        "response_time": response_time
    })

    # 🔥 Generate AI title ONLY for first message
    if chat.get("title") == "New Chat":
        try:
            chat["title"] = generate_chat_title(req.question)
        except:
            ensure_title(chat)  # fallback
    print("🔥 Chat title:", chat.get("title"))
    save_chat(chat)

    # -------------------------------
    # RESPONSE
    # -------------------------------
    return {
        "chat_id": chat_id,
        "answer": ans.answer,
        "sources": sources,
        "cache_status": ans.cache_status,
        "response_time": response_time
    }

# -------------------------------
# 🔹 FILE UPLOAD API
# -------------------------------
@app.post("/upload")
async def upload(file: UploadFile = File(...)):

    path = uploads_dir() / file.filename

    with open(path, "wb") as f:
        f.write(await file.read())

    file_hash, chunks = ingest_file(path)

    _CACHE.clear()
    _cached_llm.cache_clear()
    _cached_retrieval.cache_clear()

    return {
        "filename": file.filename,
        "chunks_indexed": chunks
    }
    

# -------------------------------
# 🔹chat history  API
# ------------------------------

@app.get("/chats")
def get_chats():
    return list_chats()

@app.get("/chat/{chat_id}")
def get_chat(chat_id: str):
    return load_chat(chat_id)

@app.delete("/delete-file/{filename}")
def delete_file(filename: str):
    from rag.paths import uploads_dir

    path = uploads_dir() / filename
    print("Before delete - cache size:", len(_CACHE))
    print("Deleting:", filename)
    # 1. delete file
    if path.exists():
        path.unlink()

    # 2. delete vectors
    delete_by_source_file(filename)

    # 3. clear semantic cache
    _CACHE.clear()
    print("Semantic cache cleared")

    # 4. clear LLM + retrieval cache
    _cached_llm.cache_clear()
    print("LLM cache cleared")
    _cached_retrieval.cache_clear()
    print("After delete - cache cleared")
    return {"status": "deleted"}


@app.delete("/chat/{chat_id}")
def remove_chat(chat_id: str):
    delete_chat(chat_id)
    return {"status": "deleted"}


@app.delete("/chats")
def remove_all_chats():
    print("🔥 CLEAR ALL CHATS CALLED")
    clear_all_chats()
    return {"status": "all deleted"}


@app.put("/chat/{chat_id}/rename")
def rename_chat(chat_id: str, body: dict):
    chat = load_chat(chat_id)

    chat["title"] = body.get("title", "Untitled")

    save_chat(chat)

    return {"status": "renamed"}

from rag.paths import uploads_dir

@app.get("/files")
def list_files():
    files = []

    for f in uploads_dir().glob("*"):
        if f.is_file():
            files.append(f.name)

    return files

# class UserRequest(BaseModel):
#     email: str
#     password: str = Field(...,max_length=72)

#     class Config:
#         min_anystr_length = 3

# @app.post("/signup")
# def signup(user: UserRequest):
#     users = load_users()

#     if any(u["email"] == user.email for u in users):
#         return {"error": "User already exists"}

#     users.append({
#         "email": user.email,
#         "password": hash_password(user.password)
#     })

#     save_users(users)

#     return {"message": "User created"}

# @app.post("/login")
# def login(user: UserRequest):
#     users = load_users()

#     db_user = next((u for u in users if u["email"] == user.email), None)

#     if not db_user:
#         return {"error": "User not found"}

#     if not verify_password(user.password, db_user["password"]):
#         return {"error": "Invalid password"}

#     token = create_token({"sub": user.email})

#     return {"token": token}

